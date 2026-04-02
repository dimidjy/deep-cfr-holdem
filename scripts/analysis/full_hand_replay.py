"""
Full decision tree replay: replays entire hands using the MODEL's decisions
instead of the real player's, computing EV at each divergence point.

Uses showdown hands where target player's cards are known. At each decision
point, compares the model's preferred action to the player's actual action.
Divergences are classified and their EV contribution is estimated.

Usage:
  python3 scripts/analysis/full_hand_replay.py \
      --model-facing-bet models/best/cql_6p_facing_bet_v7.pt \
      --model-checked-to models/best/cql_6p_checked_to_v3.pt \
      --max-hands 20000
"""

import sys
import os
import argparse
import time
import multiprocessing as mp
import warnings
import logging
from pathlib import Path
from collections import defaultdict

# Project root = parent of scripts/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import eval7
from src.hand_history.parser import parse_file
from src.hand_history import ObservationBuilder
from src.hand_history.models import ParsedHand, dollars_to_chips, DOLLARS_TO_CHIPS
from src.hand_history.action_mapper import (
    FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN,
    map_action, get_legal_actions,
)
from scripts.train_cql import QNetwork, CQLModelWrapper

# ── Constants ────────────────────────────────────────────────────────────────

NUM_PLAYERS = 6
ACTION_NAMES = ['Fold', 'Call', 'RaiseHalf', 'RaisePot', 'AllIn']
STREET_ORDER = ['preflop', 'flop', 'turn', 'river']

# Build eval7 random range for equity calculations
_RANKS = '23456789TJQKA'
_ALL_COMBOS = []
for _i, _r1 in enumerate(_RANKS):
    for _j, _r2 in enumerate(_RANKS):
        if _i < _j:
            _ALL_COMBOS.append(f'{_r2}{_r1}s')
            _ALL_COMBOS.append(f'{_r2}{_r1}o')
        elif _i == _j:
            _ALL_COMBOS.append(f'{_r1}{_r2}')
_EVAL7_RANDOM_RANGE = eval7.HandRange(','.join(_ALL_COMBOS))


# ── Model loading ────────────────────────────────────────────────────────────

def load_cql_model(model_path, exclude_fold=False, temperature=1.0):
    """Load a CQL model from checkpoint."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device)
    obs_size = checkpoint.get('obs_size', 41)
    action_size = checkpoint.get('action_size', 5)
    hidden_layers = checkpoint.get('hidden_layers', [512, 512, 256])
    dropout = checkpoint.get('dropout', 0.1)

    q_net = QNetwork(obs_size, action_size, hidden_layers, dropout).to(device)
    q_net.load_state_dict(checkpoint['model_state_dict'])
    q_net.eval()

    wrapper = CQLModelWrapper(
        q_net, obs_size, action_size, NUM_PLAYERS, device,
        iteration=checkpoint.get('epoch', 0),
        temperature=temperature,
        exclude_fold=exclude_fold,
    )
    return wrapper


# ── Equity calculation ───────────────────────────────────────────────────────

def compute_equity_vs_hand(hero_cards, opp_cards, board_cards):
    """Compute hero's equity vs a specific opponent hand on a given board.

    Uses Monte Carlo for incomplete boards, exact evaluation for complete boards.
    Returns win probability in [0, 1].
    """
    try:
        hero = [eval7.Card(c) for c in hero_cards]
        opp = [eval7.Card(c) for c in opp_cards]
        board = [eval7.Card(c) for c in board_cards] if board_cards else []
    except Exception:
        return 0.5

    remaining = 5 - len(board)
    if remaining == 0:
        # Exact evaluation on complete board
        hero_score = eval7.evaluate(hero + board)
        opp_score = eval7.evaluate(opp + board)
        if hero_score > opp_score:
            return 1.0
        elif hero_score < opp_score:
            return 0.0
        else:
            return 0.5

    # Monte Carlo for incomplete boards
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


def compute_equity_vs_range(hero_cards, board_cards, n_opponents=1):
    """Compute hero's equity vs random range (for cases where opp hand unknown)."""
    try:
        hero = [eval7.Card(c) for c in hero_cards]
        board = [eval7.Card(c) for c in board_cards] if board_cards else []
        n_samples = 1000 if not board else 500
        eq = eval7.py_hand_vs_range_monte_carlo(
            hero, _EVAL7_RANDOM_RANGE, board, n_samples)
        return eq
    except Exception:
        return 0.5


# ── Game state tracker (mirrors obs_builder._PlayerState) ────────────────────

class PlayerState:
    """Track one player's chip state through a hand."""
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


# ── Hand replay engine ───────────────────────────────────────────────────────

class HandReplayer:
    """Replays a parsed hand, building observations and querying the model
    at each decision point for a target player."""

    def __init__(self, model_facing_bet, model_checked_to, obs_builder, model_facing_bet_3bet=None):
        self.model_fb = model_facing_bet
        self.model_ct = model_checked_to
        self.model_fb_3bet = model_facing_bet_3bet
        self.obs_builder = obs_builder
        # Preflop ranges for preflop decisions
        try:
            from src.preflop_ranges import PreflopRanges
            self.preflop_ranges = PreflopRanges(num_players=NUM_PLAYERS)
        except Exception:
            self.preflop_ranges = None

    def replay_hand(self, hand):
        """Replay one hand, return list of divergence records for all showdown
        players whose cards are known.

        Each record: {
            'player': str,
            'hand_id': str,
            'street': str,
            'player_action': int,
            'model_action': int,
            'divergence_type': str,  # 'none', 'fold_save', 'call_value', 'raise_value', ...
            'ev_delta': float,       # estimated EV difference (model - player) in dollars
            'pot_dollars': float,
            'invested_dollars': float,
            'equity': float,
            'player_net_dollars': float,
        }
        """
        active_players = hand.active_players
        if len(active_players) != NUM_PLAYERS:
            return []

        # We need at least 2 shown hands for equity calculations
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

        # Button index
        button_idx = 0
        for i, pp in enumerate(active_players):
            if pp.seat == hand.button_seat:
                button_idx = i
                break

        # Community cards at each street
        cc = hand.community_cards_at_street
        current_street = 'preflop'
        community_cards = []

        # Compute each player's net result
        player_nets = self._compute_player_nets(hand, name_to_idx)

        # Opponent hands lookup: for each shown player, what are the opponents' hands?
        shown_players = set(hand.shown_hands.keys())

        # Track invested chips per player (across entire hand)
        invested = defaultdict(int)  # name -> total chips put in

        # Behavioral stats tracking
        from collections import defaultdict as dd
        avg_stack = max(1, sum(p.stake + p.bet_chips for p in players) / len(players))
        bh = {
            'pot_committed': 0,
            'street_bets': dd(int),
            'player_total': dd(int),
            'player_agg': dd(int),
            'player_fold': dd(int),
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

            # Blinds: not decisions, just apply
            if action.is_blind:
                chips = action.amount_chips
                actual = min(chips, player.stake)
                player.bet_chips += actual
                player.stake -= actual
                invested[player.name] += actual
                bh['street_bets'][action.player_name] += action.amount_chips
                continue

            # Skip folded/all-in players
            if not player.active or player.stake == 0:
                self._track_behavioral(action, bh)
                continue

            # Only process decisions for players with known hole cards
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

                # Build observation
                obs = self.obs_builder._build_observation(
                    players=[self._to_obs_player(p) for p in players],
                    player_idx=pidx,
                    button_idx=button_idx,
                    street=current_street,
                    community_cards=community_cards,
                )

                if obs is not None:
                    # Append behavioral stats
                    from src.hand_history.obs_builder import _bh_compute
                    bh_features = _bh_compute(bh, player.name)
                    obs = np.concatenate([obs, bh_features])

                    # ── Preflop: use ranges ──
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
                        # ── Postflop: use split models ──
                        facing_bet = obs[10] > 0.001
                        pot_bb = obs[8] * 600 if len(obs) > 8 else 0
                        if facing_bet and pot_bb > 12 and self.model_fb_3bet:
                            model = self.model_fb_3bet
                        elif facing_bet:
                            model = self.model_fb
                        else:
                            model = self.model_ct
                        model_legal = legal
                        if not facing_bet:
                            model_legal = [a for a in legal if a != 0] or [1]

                        strategy = model.get_strategy(obs, pidx, model_legal)
                        model_action = model_legal[0]
                        best_prob = -1
                        for a in model_legal:
                            if strategy[a] > best_prob:
                                best_prob = strategy[a]
                                model_action = a

                    # Compute divergence EV
                    pot_dollars = pot / DOLLARS_TO_CHIPS
                    invested_dollars = invested[player.name] / DOLLARS_TO_CHIPS

                    # Find opponents' hands for equity
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

            # Track behavioral stats
            self._track_behavioral(action, bh)

            # Apply action to state
            self._apply_action(player, action, max(p.bet_chips for p in players),
                               invested)

        return divergences

    def _to_obs_player(self, ps):
        """Convert our PlayerState to obs_builder's _PlayerState."""
        from src.hand_history.obs_builder import _PlayerState
        p = _PlayerState(ps.name, ps.stake, ps.seat, ps.hole_cards)
        p.pot_chips = ps.pot_chips
        p.bet_chips = ps.bet_chips
        p.active = ps.active
        return p

    def _track_behavioral(self, action, bh):
        """Update behavioral tracking state."""
        from src.hand_history.obs_builder import _bh_track_action
        _bh_track_action(action, bh)

    def _apply_action(self, player, action, max_bet, invested):
        """Apply action to player state and update invested tracker."""
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

    def _compute_player_nets(self, hand, name_to_idx):
        """Compute net profit/loss for each player (dollars)."""
        nets = {}
        # Total invested per player
        invested = defaultdict(float)
        for action in hand.actions:
            if action.is_blind:
                invested[action.player_name] += action.amount_dollars
            elif action.action_type == 'call':
                invested[action.player_name] += action.amount_dollars
            elif action.action_type in ('bet', 'raise'):
                # For raise: amount_dollars is "to" amount; invested is cumulative
                # We use a simpler approach: net = won - total_invested
                pass

        # Simpler: compute from chip stacks
        active_players = hand.active_players
        for pp in active_players:
            name = pp.name
            start_stack = pp.stack_dollars
            # What they won
            won = sum(w['amount_dollars'] for w in hand.result.winners
                      if w['name'] == name)
            # Net = won - (start_stack - end_stack_implied)
            # We don't have end stack, but net = won - invested
            # Actually we can compute invested from actions
            total_invested = 0.0
            for action in hand.actions:
                if action.player_name != name:
                    continue
                if action.is_blind:
                    total_invested = action.amount_dollars  # blinds overwrite
                elif action.action_type == 'call':
                    total_invested += action.amount_dollars
                elif action.action_type in ('bet', 'raise'):
                    # "to" amount is cumulative for the street
                    total_invested = action.amount_dollars  # simplification

            # Better approach: just use won - total pot share
            # From chip state reconstruction
            nets[name] = won - self._total_invested_from_actions(hand, name)

        return nets

    def _total_invested_from_actions(self, hand, player_name):
        """Calculate total dollars invested by player across all streets."""
        total = 0.0
        street_bet = 0.0
        current_street = 'preflop'

        for action in hand.actions:
            if action.player_name != player_name:
                # Track street transitions via any action's street
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
                # Call adds to the street bet
                # Call amount is incremental (the call cost)
                # But in PS format, call amount is the call amount, not cumulative
                street_bet += action.amount_dollars
            elif action.action_type in ('bet', 'raise'):
                # Raise "to X" means total bet this street is X
                # Bet "X" means total bet this street is X
                street_bet = action.amount_dollars

        total += street_bet
        return total

    def _compute_divergence(self, player_action, model_action, model_strategy,
                            hero_cards, opp_hands, community_cards,
                            pot_dollars, invested_dollars, player_net,
                            street, hand_id, player_name, player, max_bet,
                            legal):
        """Compute the EV impact of a divergence between model and player."""
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

        # Compute equity vs known opponents
        if opp_hands:
            # Average equity vs all known opponent hands
            equities = []
            for opp in opp_hands:
                eq = compute_equity_vs_hand(hero_cards, opp, community_cards)
                equities.append(eq)
            equity = np.mean(equities)
        else:
            equity = compute_equity_vs_range(hero_cards, community_cards)

        record['equity'] = equity

        call_cost_chips = max_bet - player.bet_chips
        call_cost = call_cost_chips / DOLLARS_TO_CHIPS

        # ── Case 1: Model folds, player continued ────────────────────────
        if model_action == FOLD and player_action != FOLD:
            record['divergence_type'] = 'model_folds'
            # If model folds, it saves or loses the rest of the hand's investment.
            # EV(fold) = -invested_so_far
            # EV(player_continue) = player_net (actual outcome)
            # ev_delta = EV(model) - EV(player) = -invested - player_net
            # But player_net already includes the invested amount (it's net profit/loss)
            # So: if player lost $2 total (net = -$2), and model folds saving future investment:
            # The fold EV is losing what we already put in.
            # We compare model fold vs player's actual trajectory.
            # ev_delta = -invested_dollars - player_net
            # If player_net = -2.0 and invested = 0.5 at this point:
            # fold saves us from the -2.0 loss, we only lose 0.5 -> delta = -0.5 - (-2.0) = +1.5
            record['ev_delta'] = -invested_dollars - player_net
            return record

        # ── Case 2: Model continues, player folded ──────────────────────
        if player_action == FOLD and model_action != FOLD:
            record['divergence_type'] = 'model_continues'
            # Player folded -> player_net = -(invested so far).
            # Model continues: estimate EV based on equity
            if model_action == CALL:
                # EV(call) = equity * (pot + call_cost) - call_cost
                # (equity of winning the pot, minus the cost to call)
                model_ev = equity * (pot_dollars + call_cost) - call_cost
                record['ev_delta'] = model_ev - (-invested_dollars)
            elif model_action in (RAISE_HALF, RAISE_POT, ALL_IN):
                # Estimate raise EV: assume opponent calls ~50% of time
                # and folds ~50%. When called, equity applies.
                raise_size = self._estimate_raise_dollars(
                    model_action, pot_dollars, call_cost,
                    player.stake / DOLLARS_TO_CHIPS)
                # Simplified: EV = fold_equity * pot + call_equity * (equity * new_pot - raise_cost)
                fold_freq = max(0.3, 1.0 - equity)  # rough: weaker hand -> opp folds less
                new_pot = pot_dollars + raise_size + call_cost
                ev_called = equity * new_pot - (raise_size + call_cost)
                ev_fold = pot_dollars  # we win pot when they fold
                model_ev = fold_freq * ev_fold + (1 - fold_freq) * ev_called
                record['ev_delta'] = model_ev - (-invested_dollars)
            else:
                # Generic: use equity * pot - call_cost
                model_ev = equity * pot_dollars - call_cost
                record['ev_delta'] = model_ev - (-invested_dollars)
            return record

        # ── Case 3: Different sizing (both continue) ────────────────────
        # e.g., model raises where player called, or model calls where player raised
        record['divergence_type'] = 'sizing_diff'

        # Compare action aggression
        aggression = {FOLD: 0, CALL: 1, RAISE_HALF: 2, RAISE_POT: 3, ALL_IN: 4}
        model_agg = aggression.get(model_action, 1)
        player_agg = aggression.get(player_action, 1)

        if model_agg > player_agg:
            # Model is more aggressive
            extra_invest = self._extra_investment(
                model_action, player_action, pot_dollars, call_cost,
                player.stake / DOLLARS_TO_CHIPS)
            # Extra EV from bigger bet: equity * extra_bet when called
            # Rough fold equity bonus
            fold_bonus = 0.3 * pot_dollars if model_action >= RAISE_HALF else 0.0
            ev_extra = equity * extra_invest - (1 - equity) * extra_invest + fold_bonus * 0.3
            record['ev_delta'] = ev_extra
        else:
            # Model is less aggressive (e.g., call where player raised)
            # The model saves the extra investment when equity doesn't justify it
            saved = self._extra_investment(
                player_action, model_action, pot_dollars, call_cost,
                player.stake / DOLLARS_TO_CHIPS)
            # EV of saving = (1 - equity) * saved - equity * saved (lost value)
            # = saved * (1 - 2 * equity)
            record['ev_delta'] = saved * (1.0 - 2.0 * equity)

        return record

    def _estimate_raise_dollars(self, action, pot_dollars, call_cost, stack_dollars):
        """Estimate the raise size in dollars for a discrete action."""
        if action == RAISE_HALF:
            return max(pot_dollars * 0.5, 0.05)
        elif action == RAISE_POT:
            return max(pot_dollars, 0.05)
        elif action == ALL_IN:
            return max(stack_dollars - call_cost, 0.05)
        return 0.0

    def _extra_investment(self, bigger_action, smaller_action, pot_dollars,
                          call_cost, stack_dollars):
        """Compute the extra investment of bigger_action over smaller_action."""
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

        return max(_cost(bigger_action) - _cost(smaller_action), 0.0)


# ── File processing ──────────────────────────────────────────────────────────

def find_hh_files(data_dir, max_files=None):
    """Find hand history text files."""
    data_path = Path(data_dir)
    files = sorted(data_path.glob('*.txt'))
    if max_files:
        files = files[:max_files]
    return files


def process_file(filepath, replayer, require_6_active=True):
    """Process one HH file and return divergence records."""
    try:
        hands = parse_file(str(filepath))
    except Exception:
        return []

    records = []
    for hand in hands:
        # Filter: NL5 only
        if abs(hand.big_blind_dollars - 0.05) > 0.001:
            continue
        # Filter: 6-max with 6 active
        if require_6_active and hand.num_active_players != 6:
            continue
        # Filter: showdown hands only
        if not hand.has_showdown:
            continue

        divs = replayer.replay_hand(hand)
        records.extend(divs)

    return records


# ── Analysis and reporting ───────────────────────────────────────────────────

def analyze_divergences(records):
    """Analyze divergence records and produce summary statistics."""
    if not records:
        return {}

    total_hands = len(set((r['hand_id'], r['player']) for r in records))
    total_decisions = len(records)
    total_divergences = sum(1 for r in records if r['divergence_type'] != 'agree')

    # Per-street EV breakdown
    street_ev = defaultdict(float)
    street_count = defaultdict(int)
    street_div_count = defaultdict(int)

    # Per-action-type EV
    type_ev = defaultdict(float)
    type_count = defaultdict(int)

    # Per-action pair: what the model chose vs what player chose
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


def compute_bb100(total_ev, total_hands, big_blind=0.05,
                  rake_rate=0.045, non_showdown_adjustment=0.6):
    """Estimate realistic BB/100 win rate.

    Args:
        total_ev: Total EV delta in dollars.
        total_hands: Number of unique (hand, player) pairs analyzed.
        big_blind: Big blind in dollars.
        rake_rate: Approximate rake as fraction of pot.
        non_showdown_adjustment: Scale factor for non-showdown hands.
            Since we only analyze showdown hands, non-showdown EV is estimated
            as a fraction of showdown EV. Typical winning players lose a bit
            on non-showdown but gain on showdown.
    """
    if total_hands == 0:
        return 0.0

    # EV per hand (from showdown analysis only)
    ev_per_hand_showdown = total_ev / total_hands

    # Showdown hands are ~30-40% of hands. Scale total EV:
    # Non-showdown hands typically contribute negative EV for winning players
    # (they fold equity but lose some pots). We use the adjustment factor.
    ev_per_hand_total = ev_per_hand_showdown * non_showdown_adjustment

    # Subtract estimated rake impact (model doesn't avoid rake)
    # Average pot in NL5 ~$0.30-0.50, rake ~$0.02-0.04
    avg_rake_per_hand = 0.015  # ~$0.015 per hand in rake
    ev_per_hand_after_rake = ev_per_hand_total - avg_rake_per_hand

    bb_per_hand = ev_per_hand_after_rake / big_blind
    bb_per_100 = bb_per_hand * 100

    return bb_per_100


def print_report(stats, total_ev, total_hands, records):
    """Print the full analysis report."""
    print('\n' + '=' * 72)
    print('  FULL HAND REPLAY — MODEL vs PLAYER DECISION ANALYSIS')
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
    total_ev_val = stats['total_ev']
    avg_total = total_ev_val / max(total_div, 1)
    print(f'  {"TOTAL":<12} {total_dec:>10,} {total_div:>12,} '
          f'{"$" + f"{total_ev_val:+.2f}":>12} {"$" + f"{avg_total:+.4f}":>12}')

    # Per-divergence-type breakdown
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

    # Raw showdown EV
    raw_bb100 = (total_ev / max(n_hands, 1)) / bb * 100
    ev_per_hand = total_ev / max(n_hands, 1)
    print(f'  Raw showdown EV/hand:      ${ev_per_hand:+.4f}')
    print(f'  Raw showdown EV/100:       {raw_bb100:+.1f} BB/100')
    print(f'  (Showdown-only; not a realistic win rate)')
    print()

    # Adjusted estimates: showdown hands are ~30-35% of total hands dealt.
    # Non-showdown EV is typically negative for the model adjustment
    # (opponent folds preflop to our raises, we fold to their raises, etc.)
    # Conservative = assuming showdown EV advantage is heavily diluted.
    for adj_name, adj_factor in [('Conservative (0.15x)', 0.15),
                                  ('Moderate (0.25x)', 0.25),
                                  ('Optimistic (0.40x)', 0.40)]:
        bb100 = compute_bb100(total_ev, n_hands, big_blind=bb,
                              non_showdown_adjustment=adj_factor)
        print(f'  {adj_name:<28} {bb100:+.1f} BB/100')

    # Notes about methodology
    print('\n' + '-' * 72)
    print('  METHODOLOGY NOTES')
    print('-' * 72)
    print('  - Only showdown hands are analyzed (player cards must be known)')
    print('  - "Model continues where player folded" is always 0 (folded players')
    print('    never show cards in hand histories)')
    print('  - Equity is computed vs actual opponent hand(s) shown at showdown')
    print('  - Sizing divergence EVs are rough estimates (assume partial fold equity)')
    print('  - Non-showdown adjustment accounts for the ~60-70% of hands where')
    print('    no showdown occurs (blind steals, folds to cbets, etc.)')
    print('  - Rake estimate: ~$0.015/hand average for NL5 6-max')

    # Comparison table
    print('\n' + '-' * 72)
    print('  COMPARISON: MODEL EV vs KNOWN NL5 WIN RATES')
    print('-' * 72)
    moderate_bb100 = compute_bb100(total_ev, n_hands, big_blind=bb,
                                   non_showdown_adjustment=0.25)
    comparisons = [
        ('Average NL5 player', 0.0),
        ('Decent regular', 3.0),
        ('Good regular', 6.0),
        ('Strong regular', 10.0),
        ('Elite player', 15.0),
        ('MODEL (moderate est.)', moderate_bb100),
    ]
    print(f'  {"Player Type":<30} {"BB/100":>10}')
    print(f'  {"-"*30} {"-"*10}')
    for label, rate in comparisons:
        marker = '  <--' if label.startswith('MODEL') else ''
        print(f'  {label:<30} {rate:>+10.1f}{marker}')

    print('\n' + '=' * 72)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Full hand replay: model vs player decision EV analysis')
    parser.add_argument('--model-facing-bet', required=True,
                        help='Path to CQL model for facing-bet spots')
    parser.add_argument('--model-checked-to', required=True,
                        help='Path to CQL model for checked-to spots')
    parser.add_argument('--model-facing-bet-3bet', type=str, default=None,
                        help='Model for facing-bet spots in 3bet pots (pot > 12BB)')
    parser.add_argument('--data-dir', default='data/poker_archive',
                        help='Directory with PokerStars HH files')
    parser.add_argument('--max-hands', type=int, default=20000,
                        help='Max showdown hands to analyze')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Max HH files to process')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Model temperature (lower = greedier)')
    parser.add_argument('--require-6-active', action='store_true', default=True,
                        help='Only analyze 6-handed hands')
    parser.add_argument('--allow-short', action='store_true',
                        help='Allow short-handed tables (< 6 players)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Parsing workers (0=auto)')
    args = parser.parse_args()

    if args.allow_short:
        args.require_6_active = False

    # Parse FIRST (before torch/CUDA — mp.Pool deadlocks after CUDA init)
    data_dir = os.path.join(_PROJECT_ROOT, args.data_dir)
    files = find_hh_files(data_dir, max_files=args.max_files)
    print(f'  Found {len(files)} HH files in {data_dir}')

    from scripts.analysis._parse_utils import parallel_parse
    t_start = time.time()
    all_hands = parallel_parse(files, max_hands=args.max_hands,
                                workers=getattr(args, 'workers', 0))

    # NOW load models (after mp.Pool is done)
    print('Loading models...')
    model_fb = load_cql_model(args.model_facing_bet, exclude_fold=False,
                              temperature=args.temperature)
    model_ct = load_cql_model(args.model_checked_to, exclude_fold=True,
                              temperature=args.temperature)
    print(f'  Facing-bet model: obs_size={model_fb.obs_size}, '
          f'action_size={model_fb.action_size}')
    model_fb_3bet = None
    if args.model_facing_bet_3bet:
        model_fb_3bet = load_cql_model(args.model_facing_bet_3bet)
        print(f'  3bet facing-bet model: obs_size={model_fb_3bet.obs_size}')
    print(f'  Checked-to model: obs_size={model_ct.obs_size}, '
          f'action_size={model_ct.action_size}')

    print(f'\nInitializing observation builder...')
    obs_builder = ObservationBuilder(num_players=NUM_PLAYERS)

    replayer = HandReplayer(model_fb, model_ct, obs_builder, model_facing_bet_3bet=model_fb_3bet)

    # Process parsed hands with model (sequential — model not picklable)
    all_records = []
    hands_seen = set()

    for hi, hand in enumerate(all_hands):
        records = replayer.replay_hand(hand)
        for r in records:
            key = (r['hand_id'], r['player'])
            hands_seen.add(key)
        all_records.extend(records)

        if (hi + 1) % 2000 == 0:
            elapsed = time.time() - t_start
            print(f'  [{hi+1}/{len(all_hands)}] {len(all_records):,} decisions, '
                  f'{elapsed:.0f}s elapsed')

    elapsed = time.time() - t_start
    print(f'\nProcessed {files_processed} files in {elapsed:.1f}s')
    print(f'  Total: {len(hands_seen):,} hands, {len(all_records):,} decisions')

    if not all_records:
        print('\nNo records found. Check data directory and filters.')
        return

    # Analyze
    stats = analyze_divergences(all_records)
    print_report(stats, stats['total_ev'], stats['total_hands'], all_records)




if __name__ == '__main__':
    main()
