"""
Full hand replay analyzer: replays entire hands using the MODEL's decisions
instead of the real player's, computing EV at each divergence point.

Uses showdown hands where target player's cards are known. At each decision
point, compares the model's preferred action to the player's actual action.
Divergences are classified and their EV contribution is estimated.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, _PROJECT_ROOT)

import eval7

from src.hand_history import ObservationBuilder
from src.hand_history.models import ParsedHand, DOLLARS_TO_CHIPS
from src.hand_history.action_mapper import (
    FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN,
    map_action, get_legal_actions,
)
from scripts.analysis.analysis_db import AnalysisDB
from scripts.analysis.model_wrapper import ModelWrapper

# ── Constants ───────────────────────────────────────────────────────────────

NUM_PLAYERS = 6
ACTION_NAMES = ['Fold', 'Call', 'RaiseHalf', 'RaisePot', 'AllIn']
STREET_ORDER = ['preflop', 'flop', 'turn', 'river']


# ── Equity calculation ──────────────────────────────────────────────────────

def compute_equity_vs_hand(hero_cards, opp_cards, board_cards):
    """Compute hero's equity vs a specific opponent hand on a given board."""
    try:
        hero = [eval7.Card(c) for c in hero_cards]
        opp = [eval7.Card(c) for c in opp_cards]
        board = [eval7.Card(c) for c in board_cards] if board_cards else []
    except Exception:
        return 0.5

    remaining = 5 - len(board)
    if remaining == 0:
        hero_score = eval7.evaluate(hero + board)
        opp_score = eval7.evaluate(opp + board)
        if hero_score > opp_score:
            return 1.0
        elif hero_score < opp_score:
            return 0.0
        else:
            return 0.5

    import random
    deck = [eval7.Card(r + s) for r in '23456789TJQKA' for s in 'shdc']
    used = set(str(c) for c in hero + opp + board)
    available = [c for c in deck if str(c) not in used]

    if len(available) < remaining:
        return 0.5

    n_samples = 500
    wins = 0
    ties = 0
    for _ in range(n_samples):
        random.shuffle(available)
        full_board = board + available[:remaining]
        hero_score = eval7.evaluate(hero + full_board)
        opp_score = eval7.evaluate(opp + full_board)
        if hero_score > opp_score:
            wins += 1
        elif hero_score == opp_score:
            ties += 1

    return (wins + ties * 0.5) / n_samples


# ── Player state tracker ───────────────────────────────────────────────────

class PlayerState:
    __slots__ = ('name', 'stake', 'pot_chips', 'bet_chips', 'active',
                 'hole_cards', 'seat', 'initial_stake')

    def __init__(self, name, stake, seat, hole_cards=None):
        self.name = name
        self.stake = stake
        self.pot_chips = 0
        self.bet_chips = 0
        self.active = True
        self.hole_cards = hole_cards
        self.seat = seat
        self.initial_stake = stake


# ── Hand replayer ──────────────────────────────────────────────────────────

class HandReplayer:
    """Replays a parsed hand, building observations and querying the model."""

    def __init__(self, model: ModelWrapper, obs_builder: ObservationBuilder):
        self.model = model
        self.obs_builder = obs_builder
        # Preflop ranges for preflop decisions
        try:
            from src.preflop_ranges import PreflopRanges
            self.preflop_ranges = PreflopRanges(num_players=NUM_PLAYERS)
        except Exception:
            self.preflop_ranges = None

    def replay_hand(self, hand: ParsedHand) -> List[dict]:
        """Replay one hand, return list of divergence records."""
        active_players = hand.active_players
        if len(active_players) != NUM_PLAYERS:
            return []

        if len(hand.shown_hands) < 2:
            return []

        # Build player states
        name_to_idx = {}
        players = []
        for i, pp in enumerate(active_players):
            name_to_idx[pp.name] = i
            players.append(PlayerState(
                name=pp.name,
                stake=pp.stack_chips,
                seat=pp.seat,
                hole_cards=pp.hole_cards,
            ))

        button_idx = 0
        for i, pp in enumerate(active_players):
            if pp.seat == hand.button_seat:
                button_idx = i
                break

        cc = hand.community_cards_at_street
        current_street = 'preflop'
        community_cards = []

        player_nets = self._compute_player_nets(hand)
        shown_players = set(hand.shown_hands.keys())

        invested = defaultdict(int)

        # Behavioral stats tracking
        avg_stack = max(1, sum(p.stake + p.bet_chips for p in players) / len(players))
        bh = {
            'pot_committed': 0,
            'street_bets': defaultdict(int),
            'player_total': defaultdict(int),
            'player_agg': defaultdict(int),
            'player_fold': defaultdict(int),
            'sizing_ratios': [],
            'raise_count': 0,
            'street_actions': [],
            'avg_stack': avg_stack,
        }

        divergences = []

        for action in hand.actions:
            if action.player_name not in name_to_idx:
                continue

            pidx = name_to_idx[action.player_name]
            player = players[pidx]

            # Street transition
            if action.street != current_street:
                for p in players:
                    p.pot_chips += p.bet_chips
                    p.bet_chips = 0
                current_street = action.street
                community_cards = cc.get(current_street, [])
                bh['pot_committed'] += sum(bh['street_bets'].values())
                bh['street_bets'].clear()
                bh['street_actions'] = []

            # Blinds
            if action.is_blind:
                chips = action.amount_chips
                actual = min(chips, player.stake)
                player.bet_chips += actual
                player.stake -= actual
                invested[player.name] += actual
                bh['street_bets'][action.player_name] += action.amount_chips
                continue

            if not player.active or player.stake == 0:
                self._track_behavioral(action, bh)
                continue

            if player.name in shown_players and player.hole_cards:
                pot = sum(p.pot_chips + p.bet_chips for p in players)
                max_bet = max(p.bet_chips for p in players)

                legal = get_legal_actions(
                    max_bet=max_bet,
                    player_bet_chips=player.bet_chips,
                    player_stake=player.stake,
                    pot=pot,
                )

                mapped_action = map_action(
                    action=action,
                    pot=pot,
                    max_bet=max_bet,
                    player_bet_chips=player.bet_chips,
                    player_stake=player.stake,
                )

                obs = self.obs_builder._build_observation(
                    players=[self._to_obs_player(p) for p in players],
                    player_idx=pidx,
                    button_idx=button_idx,
                    street=current_street,
                    community_cards=community_cards,
                )

                if obs is not None:
                    from src.hand_history.obs_builder import _bh_compute
                    bh_features = _bh_compute(bh, player.name)
                    obs = np.concatenate([obs, bh_features])

                    # Preflop: use ranges
                    if current_street == 'preflop' and self.preflop_ranges and player.hole_cards:
                        hole = list(player.hole_cards[:2])
                        rel_pos = (pidx - button_idx) % NUM_PLAYERS
                        pot_bb = pot / (hand.big_blind_dollars * DOLLARS_TO_CHIPS)
                        call_bb = max_bet - player.bet_chips
                        if call_bb > 0:
                            eff_pot = pot_bb + call_bb / (hand.big_blind_dollars * DOLLARS_TO_CHIPS)
                        else:
                            eff_pot = pot_bb
                        pf_action = self.preflop_ranges.get_action(
                            hole, rel_pos, eff_pot, legal)
                        model_action = pf_action if pf_action is not None else (
                            0 if 0 in legal else 1)
                        strategy = np.zeros(5, dtype=np.float32)
                        strategy[model_action] = 1.0
                    else:
                        # Postflop: use model_wrapper
                        model_action, strategy = self.model.get_action(
                            obs, legal, player_idx=pidx)

                    # Compute divergence
                    pot_dollars = pot / DOLLARS_TO_CHIPS
                    invested_dollars = invested[player.name] / DOLLARS_TO_CHIPS

                    opp_hands = []
                    for oname, ocards in hand.shown_hands.items():
                        if oname != player.name and len(ocards) >= 2:
                            opp_hands.append(ocards)

                    div_record = self._compute_divergence(
                        player_action=mapped_action,
                        model_action=model_action,
                        model_strategy=strategy,
                        hero_cards=player.hole_cards,
                        opp_hands=opp_hands,
                        community_cards=community_cards,
                        pot_dollars=pot_dollars,
                        invested_dollars=invested_dollars,
                        player_net=player_nets.get(player.name, 0.0),
                        street=current_street,
                        hand_id=hand.hand_id,
                        player_name=player.name,
                        player=player,
                        max_bet=max_bet,
                        legal=legal,
                    )
                    divergences.append(div_record)

            self._track_behavioral(action, bh)
            self._apply_action(player, action, max(p.bet_chips for p in players), invested)

        return divergences

    def _to_obs_player(self, ps):
        from src.hand_history.obs_builder import _PlayerState
        p = _PlayerState(ps.name, ps.stake, ps.seat, ps.hole_cards)
        p.pot_chips = ps.pot_chips
        p.bet_chips = ps.bet_chips
        p.active = ps.active
        return p

    def _track_behavioral(self, action, bh):
        from src.hand_history.obs_builder import _bh_track_action
        _bh_track_action(action, bh)

    def _apply_action(self, player, action, max_bet, invested):
        if action.action_type == 'fold':
            player.active = False
            return
        if action.action_type == 'check':
            return
        if action.action_type == 'call':
            call_cost = max_bet - player.bet_chips
            actual = min(call_cost, player.stake)
            player.bet_chips += actual
            player.stake -= actual
            invested[player.name] += actual
            if action.is_all_in:
                player.stake = 0
            return
        if action.action_type in ('bet', 'raise'):
            new_total_bet = action.amount_chips
            cost = new_total_bet - player.bet_chips
            actual = min(cost, player.stake)
            player.bet_chips += actual
            player.stake -= actual
            invested[player.name] += actual
            if action.is_all_in:
                player.stake = 0
            return

    def _compute_player_nets(self, hand):
        nets = {}
        for pp in hand.active_players:
            name = pp.name
            won = sum(w['amount_dollars'] for w in hand.result.winners
                      if w['name'] == name)
            nets[name] = won - self._total_invested(hand, name)
        return nets

    def _total_invested(self, hand, player_name):
        total = 0.0
        street_bet = 0.0
        current_street = 'preflop'

        for action in hand.actions:
            if action.player_name != player_name:
                if action.street != current_street:
                    total += street_bet
                    street_bet = 0.0
                    current_street = action.street
                continue

            if action.street != current_street:
                total += street_bet
                street_bet = 0.0
                current_street = action.street

            if action.is_blind:
                street_bet = action.amount_dollars
            elif action.action_type == 'call':
                street_bet += action.amount_dollars
            elif action.action_type in ('bet', 'raise'):
                street_bet = action.amount_dollars

        total += street_bet
        return total

    def _compute_divergence(self, player_action, model_action, model_strategy,
                            hero_cards, opp_hands, community_cards,
                            pot_dollars, invested_dollars, player_net,
                            street, hand_id, player_name, player, max_bet,
                            legal):
        record = {
            'player': player_name,
            'hand_id': hand_id,
            'street': street,
            'player_action': player_action,
            'model_action': model_action,
            'model_strategy': model_strategy.copy(),
            'divergence_type': 'none',
            'ev_delta': 0.0,
            'pot_dollars': pot_dollars,
            'invested_dollars': invested_dollars,
            'equity': 0.0,
            'player_net_dollars': player_net,
        }

        if model_action == player_action:
            record['divergence_type'] = 'agree'
            return record

        # Compute equity
        if opp_hands:
            equities = []
            for opp in opp_hands:
                eq = compute_equity_vs_hand(hero_cards, opp, community_cards)
                equities.append(eq)
            equity = np.mean(equities)
        else:
            equity = 0.5

        record['equity'] = equity
        call_cost_chips = max_bet - player.bet_chips
        call_cost = call_cost_chips / DOLLARS_TO_CHIPS

        # Case 1: Model folds, player continued
        if model_action == FOLD and player_action != FOLD:
            record['divergence_type'] = 'model_folds'
            record['ev_delta'] = -invested_dollars - player_net
            return record

        # Case 2: Model continues, player folded
        if player_action == FOLD and model_action != FOLD:
            record['divergence_type'] = 'model_continues'
            if model_action == CALL:
                model_ev = equity * (pot_dollars + call_cost) - call_cost
                record['ev_delta'] = model_ev - (-invested_dollars)
            elif model_action in (RAISE_HALF, RAISE_POT, ALL_IN):
                raise_size = self._estimate_raise(
                    model_action, pot_dollars, call_cost,
                    player.stake / DOLLARS_TO_CHIPS)
                fold_freq = max(0.3, 1.0 - equity)
                new_pot = pot_dollars + raise_size + call_cost
                ev_called = equity * new_pot - (raise_size + call_cost)
                ev_fold = pot_dollars
                model_ev = fold_freq * ev_fold + (1 - fold_freq) * ev_called
                record['ev_delta'] = model_ev - (-invested_dollars)
            else:
                model_ev = equity * pot_dollars - call_cost
                record['ev_delta'] = model_ev - (-invested_dollars)
            return record

        # Case 3: Different sizing
        record['divergence_type'] = 'sizing_diff'
        aggression = {FOLD: 0, CALL: 1, RAISE_HALF: 2, RAISE_POT: 3, ALL_IN: 4}
        model_agg = aggression.get(model_action, 1)
        player_agg = aggression.get(player_action, 1)

        if model_agg > player_agg:
            extra = self._extra_investment(
                model_action, player_action, pot_dollars, call_cost,
                player.stake / DOLLARS_TO_CHIPS)
            fold_bonus = 0.3 * pot_dollars if model_action >= RAISE_HALF else 0.0
            ev_extra = equity * extra - (1 - equity) * extra + fold_bonus * 0.3
            record['ev_delta'] = ev_extra
        else:
            saved = self._extra_investment(
                player_action, model_action, pot_dollars, call_cost,
                player.stake / DOLLARS_TO_CHIPS)
            record['ev_delta'] = saved * (1.0 - 2.0 * equity)

        return record

    def _estimate_raise(self, action, pot_dollars, call_cost, stack_dollars):
        if action == RAISE_HALF:
            return max(pot_dollars * 0.5, 0.05)
        elif action == RAISE_POT:
            return max(pot_dollars, 0.05)
        elif action == ALL_IN:
            return max(stack_dollars - call_cost, 0.05)
        return 0.0

    def _extra_investment(self, bigger, smaller, pot_dollars, call_cost, stack_dollars):
        def _cost(a):
            if a == FOLD:
                return 0.0
            if a == CALL:
                return call_cost
            if a == RAISE_HALF:
                return call_cost + max(pot_dollars * 0.5, 0.05)
            if a == RAISE_POT:
                return call_cost + max(pot_dollars, 0.05)
            if a == ALL_IN:
                return stack_dollars
            return 0.0
        return max(_cost(bigger) - _cost(smaller), 0.0)


# ── Divergence analysis ────────────────────────────────────────────────────

def _analyze_divergences(records):
    if not records:
        return {}

    total_hands = len(set((r['hand_id'], r['player']) for r in records))
    total_decisions = len(records)
    total_divergences = sum(1 for r in records if r['divergence_type'] != 'agree')

    street_ev = defaultdict(float)
    street_count = defaultdict(int)
    street_div_count = defaultdict(int)
    type_ev = defaultdict(float)
    type_count = defaultdict(int)
    action_pair_ev = defaultdict(lambda: {'ev': 0.0, 'count': 0})

    total_ev = 0.0

    for r in records:
        st = r['street']
        street_count[st] += 1

        if r['divergence_type'] == 'agree':
            continue

        ev = r['ev_delta']
        total_ev += ev
        street_ev[st] += ev
        street_div_count[st] += 1

        dtype = r['divergence_type']
        type_ev[dtype] += ev
        type_count[dtype] += 1

        pair_key = (ACTION_NAMES[r['player_action']], ACTION_NAMES[r['model_action']])
        action_pair_ev[pair_key]['ev'] += ev
        action_pair_ev[pair_key]['count'] += 1

    return {
        'total_hands': total_hands,
        'total_decisions': total_decisions,
        'total_divergences': total_divergences,
        'divergence_rate': total_divergences / max(total_decisions, 1),
        'total_ev': total_ev,
        'street_ev': dict(street_ev),
        'street_count': dict(street_count),
        'street_div_count': dict(street_div_count),
        'type_ev': dict(type_ev),
        'type_count': dict(type_count),
        'action_pair_ev': dict(action_pair_ev),
    }


# ── Main analysis function ──────────────────────────────────────────────────

def analyze(
    db: AnalysisDB,
    model: ModelWrapper,
    **kwargs,
) -> dict:
    """Run full hand replay analysis.

    Replays each showdown hand, comparing model decisions to actual play.

    Returns:
        Dictionary with divergence analysis results.
    """
    obs_builder = ObservationBuilder(num_players=NUM_PLAYERS)
    replayer = HandReplayer(model=model, obs_builder=obs_builder)

    all_records = []
    n_hands = 0

    for hi, hand in enumerate(db.hands):
        if hand.num_active_players != NUM_PLAYERS:
            continue
        if len(hand.shown_hands) < 2:
            continue

        records = replayer.replay_hand(hand)
        all_records.extend(records)
        n_hands += 1

        if (hi + 1) % 2000 == 0:
            n_div = sum(1 for r in all_records if r['divergence_type'] != 'agree')
            print(f'    [{hi+1}/{len(db.hands)}] {len(all_records)} decisions, '
                  f'{n_div} divergences...', flush=True)

    stats = _analyze_divergences(all_records)
    stats['records'] = all_records
    stats['n_hands_replayed'] = n_hands

    return stats


# ── Report printing ─────────────────────────────────────────────────────────

def print_report(results: dict):
    """Print the full hand replay analysis report."""
    stats = results

    if not stats or 'total_hands' not in stats:
        print('No results to display.')
        return

    print('\n' + '=' * 72)
    print('  FULL HAND REPLAY -- MODEL vs PLAYER DECISION ANALYSIS')
    print('=' * 72)

    print(f'\n  Hands analyzed (showdown):  {stats["total_hands"]:,}')
    print(f'  Total decision points:     {stats["total_decisions"]:,}')
    print(f'  Total divergences:         {stats["total_divergences"]:,}')
    print(f'  Divergence rate:           {stats["divergence_rate"]:.1%}')
    print(f'  Total EV delta:            ${stats["total_ev"]:+.2f}')

    # Per-street breakdown
    print('\n' + '-' * 72)
    print('  PER-STREET EV BREAKDOWN')
    print('-' * 72)
    print(f'  {"Street":<12} {"Decisions":>10} {"Divergences":>12} '
          f'{"EV Delta":>12} {"Avg EV/Div":>12}')
    print(f'  {"-"*12} {"-"*10} {"-"*12} {"-"*12} {"-"*12}')

    for st in STREET_ORDER:
        n_dec = stats['street_count'].get(st, 0)
        n_div = stats['street_div_count'].get(st, 0)
        ev = stats['street_ev'].get(st, 0.0)
        avg = ev / max(n_div, 1)
        print(f'  {st:<12} {n_dec:>10,} {n_div:>12,} '
              f'{"$" + f"{ev:+.2f}":>12} {"$" + f"{avg:+.4f}":>12}')

    total_dec = stats['total_decisions']
    total_div = stats['total_divergences']
    total_ev = stats['total_ev']
    avg_total = total_ev / max(total_div, 1)
    print(f'  {"TOTAL":<12} {total_dec:>10,} {total_div:>12,} '
          f'{"$" + f"{total_ev:+.2f}":>12} {"$" + f"{avg_total:+.4f}":>12}')

    # Per-type breakdown
    print('\n' + '-' * 72)
    print('  PER-ACTION-TYPE EV BREAKDOWN')
    print('-' * 72)
    type_labels = {
        'model_folds': 'Model folds (player continued)',
        'model_continues': 'Model continues (player folded)',
        'sizing_diff': 'Different sizing (both continue)',
    }
    print(f'  {"Type":<42} {"Count":>8} {"EV Delta":>12} {"Avg":>10}')
    print(f'  {"-"*42} {"-"*8} {"-"*12} {"-"*10}')
    for dtype, label in type_labels.items():
        count = stats['type_count'].get(dtype, 0)
        ev = stats['type_ev'].get(dtype, 0.0)
        avg = ev / max(count, 1)
        print(f'  {label:<42} {count:>8,} {"$" + f"{ev:+.2f}":>12} '
              f'{"$" + f"{avg:+.4f}":>10}')

    # Top action pairs
    print('\n' + '-' * 72)
    print('  TOP DIVERGENCE PAIRS (Player -> Model)')
    print('-' * 72)
    pairs = sorted(stats['action_pair_ev'].items(),
                   key=lambda x: abs(x[1]['ev']), reverse=True)
    print(f'  {"Player Action -> Model Action":<35} {"Count":>8} '
          f'{"EV Delta":>12} {"Avg":>10}')
    print(f'  {"-"*35} {"-"*8} {"-"*12} {"-"*10}')
    for (pa, ma), info in pairs[:15]:
        count = info['count']
        ev = info['ev']
        avg = ev / max(count, 1)
        label = f'{pa} -> {ma}'
        print(f'  {label:<35} {count:>8,} {"$" + f"{ev:+.2f}":>12} '
              f'{"$" + f"{avg:+.4f}":>10}')

    # BB/100 estimation
    print('\n' + '-' * 72)
    print('  ESTIMATED WIN RATE')
    print('-' * 72)

    n_hands = stats['total_hands']
    bb = 0.05

    raw_bb100 = (total_ev / max(n_hands, 1)) / bb * 100
    ev_per_hand = total_ev / max(n_hands, 1)
    print(f'  Raw showdown EV/hand:      ${ev_per_hand:+.4f}')
    print(f'  Raw showdown EV/100:       {raw_bb100:+.1f} BB/100')
    print(f'  (Showdown-only; not a realistic win rate)')
    print()

    # Adjusted estimates
    for adj_label, adj_factor, rake in [
        ('Conservative', 0.4, 0.015),
        ('Moderate', 0.6, 0.010),
        ('Optimistic', 0.8, 0.005),
    ]:
        adj_ev = ev_per_hand * adj_factor - rake
        adj_bb100 = adj_ev / bb * 100
        print(f'  {adj_label} ({adj_factor:.0%} showdown weight, '
              f'${rake:.3f} rake/hand):')
        print(f'    {adj_bb100:+.1f} BB/100')

    print('\n' + '=' * 72)
