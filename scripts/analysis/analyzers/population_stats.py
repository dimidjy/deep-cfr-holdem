"""
Population stats analyzer: compare model aggregate stats against player populations.

Computes HUD-style stats (VPIP, PFR, 3Bet, CBet, etc.) for good players,
bad players, and the full population, then simulates the model's stats by
replaying hands with model decisions. Scores how closely the model matches
winning play.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, _PROJECT_ROOT)

from src.hand_history import ObservationBuilder
from src.hand_history.models import ParsedHand, dollars_to_chips, DOLLARS_TO_CHIPS
from src.hand_history.action_mapper import FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN
from scripts.analysis.analysis_db import AnalysisDB
from scripts.analysis.model_wrapper import ModelWrapper


# ── Stat tracker ────────────────────────────────────────────────────────────

class PlayerStats:
    """Accumulates poker statistics for a single player or population."""

    def __init__(self):
        self.hands_dealt = 0
        self.vpip_count = 0
        self.pfr_count = 0
        self.three_bet_opps = 0
        self.three_bet_count = 0
        self.fold_to_3bet_opps = 0
        self.fold_to_3bet_count = 0

        self.cbet_opps = 0
        self.cbet_count = 0
        self.fold_to_cbet_opps = 0
        self.fold_to_cbet_count = 0
        self.turn_cbet_opps = 0
        self.turn_cbet_count = 0
        self.river_cbet_opps = 0
        self.river_cbet_count = 0

        self.agg = defaultdict(lambda: {'bets': 0, 'raises': 0, 'calls': 0})

        self.saw_flop = 0
        self.wtsd_count = 0
        self.wsd_opps = 0
        self.wsd_count = 0

        self.fold_to_bet = defaultdict(lambda: {'opps': 0, 'count': 0})

        self.check_raise_opps = 0
        self.check_raise_count = 0

        self.bet_sizes = []

    def compute(self) -> Dict[str, float]:
        h = max(self.hands_dealt, 1)
        stats = {}
        stats['VPIP'] = 100.0 * self.vpip_count / h
        stats['PFR'] = 100.0 * self.pfr_count / h
        stats['3Bet%'] = _safe_pct(self.three_bet_count, self.three_bet_opps)
        stats['Fold_to_3Bet%'] = _safe_pct(self.fold_to_3bet_count,
                                            self.fold_to_3bet_opps)
        stats['CBet%'] = _safe_pct(self.cbet_count, self.cbet_opps)
        stats['Fold_to_CBet%'] = _safe_pct(self.fold_to_cbet_count,
                                            self.fold_to_cbet_opps)
        stats['Turn_CBet%'] = _safe_pct(self.turn_cbet_count, self.turn_cbet_opps)
        stats['River_CBet%'] = _safe_pct(self.river_cbet_count, self.river_cbet_opps)

        for street in ('flop', 'turn', 'river'):
            a = self.agg[street]
            denom = max(a['calls'], 1)
            stats[f'AF_{street}'] = (a['bets'] + a['raises']) / denom

        stats['WTSD%'] = _safe_pct(self.wtsd_count, self.saw_flop)
        stats['WSD%'] = _safe_pct(self.wsd_count, self.wsd_opps)

        for street in ('flop', 'turn', 'river'):
            fb = self.fold_to_bet[street]
            stats[f'Fold_to_Bet_{street}%'] = _safe_pct(fb['count'], fb['opps'])

        stats['CheckRaise%'] = _safe_pct(self.check_raise_count,
                                         self.check_raise_opps)
        stats['Avg_Bet_Size'] = np.mean(self.bet_sizes) if self.bet_sizes else 0.0

        return stats


def _safe_pct(num: int, den: int) -> float:
    return 100.0 * num / den if den > 0 else 0.0


# ── Hand analysis for player populations ────────────────────────────────────

def _analyze_hand_population(hand: ParsedHand,
                             target_names: Optional[Set[str]],
                             stats_map: Dict[str, PlayerStats]):
    """Update stats_map for all target players in this hand."""
    active = hand.active_players
    if len(active) < 2:
        return

    active_names = set(p.name for p in active)
    big_blind_chips = dollars_to_chips(hand.big_blind_dollars)

    if target_names is None:
        track = active_names
    else:
        track = active_names & target_names
    if not track:
        return

    for name in track:
        if name not in stats_map:
            stats_map[name] = PlayerStats()
        stats_map[name].hands_dealt += 1

    player_folded = set()
    pfr_player = None
    preflop_raise_count = 0
    player_checked_this_street = defaultdict(bool)
    player_vpip = set()
    player_pfr = set()
    player_saw_flop = set()
    street_first_bettor = {}
    flop_cbettor = None
    turn_cbettor = None

    current_street = 'preflop'
    pot_chips = 0
    street_bets = defaultdict(int)
    max_bet = 0
    street_action_list = defaultdict(list)

    for action in hand.actions:
        name = action.player_name
        if name not in active_names:
            continue

        if action.street != current_street:
            pot_chips += sum(street_bets.values())
            street_bets.clear()
            max_bet = 0
            player_checked_this_street.clear()
            if current_street == 'preflop':
                player_saw_flop = active_names - player_folded
                for n in player_saw_flop & track:
                    stats_map[n].saw_flop += 1
            current_street = action.street

        if action.is_blind:
            amt = action.amount_chips
            street_bets[name] += amt
            if amt > max_bet:
                max_bet = amt
            continue

        atype = action.action_type
        street_action_list[current_street].append(
            (name, atype, action.amount_chips, action.is_all_in))
        facing_bet = max_bet > street_bets.get(name, 0)
        is_tracked = name in track

        if atype == 'fold':
            player_folded.add(name)
            if is_tracked:
                st = stats_map[name]
                if current_street != 'preflop' and facing_bet:
                    st.fold_to_bet[current_street]['opps'] += 1
                    st.fold_to_bet[current_street]['count'] += 1
                if current_street == 'preflop' and preflop_raise_count >= 2:
                    st.fold_to_3bet_opps += 1
                    st.fold_to_3bet_count += 1
                if (current_street == 'flop' and pfr_player
                        and pfr_player != name
                        and street_first_bettor.get('flop') == pfr_player):
                    st.fold_to_cbet_opps += 1
                    st.fold_to_cbet_count += 1
            continue

        if atype == 'check':
            player_checked_this_street[name] = True
            continue

        if atype == 'call':
            street_bets[name] = max_bet
            if is_tracked:
                st = stats_map[name]
                if current_street == 'preflop':
                    player_vpip.add(name)
                    if preflop_raise_count == 1:
                        st.three_bet_opps += 1
                    if preflop_raise_count >= 2:
                        st.fold_to_3bet_opps += 1
                else:
                    st.agg[current_street]['calls'] += 1
                    if facing_bet:
                        st.fold_to_bet[current_street]['opps'] += 1
                    if (current_street == 'flop' and pfr_player
                            and pfr_player != name
                            and street_first_bettor.get('flop') == pfr_player):
                        st.fold_to_cbet_opps += 1
            continue

        if atype in ('bet', 'raise'):
            new_bet = action.amount_chips
            street_bets[name] = new_bet
            if new_bet > max_bet:
                max_bet = new_bet

            if current_street == 'preflop' and atype == 'raise':
                preflop_raise_count += 1
                if preflop_raise_count == 1:
                    pfr_player = name

            if current_street not in street_first_bettor:
                street_first_bettor[current_street] = name

            if current_street == 'flop' and name == pfr_player and atype == 'bet':
                flop_cbettor = name
            elif current_street == 'turn' and name == flop_cbettor and atype == 'bet':
                turn_cbettor = name

            if is_tracked:
                st = stats_map[name]
                if current_street == 'preflop':
                    player_vpip.add(name)
                    player_pfr.add(name)
                    if preflop_raise_count == 2:
                        st.three_bet_opps += 1
                        st.three_bet_count += 1
                    if preflop_raise_count >= 3:
                        st.fold_to_3bet_opps += 1
                else:
                    st.agg[current_street][
                        'bets' if atype == 'bet' else 'raises'] += 1
                    if facing_bet:
                        st.fold_to_bet[current_street]['opps'] += 1
                    if player_checked_this_street.get(name, False):
                        st.check_raise_count += 1
                    current_pot = pot_chips + sum(street_bets.values())
                    if current_pot > 0 and action.amount_chips > 0:
                        ratio = action.amount_chips / current_pot
                        st.bet_sizes.append(ratio)
                    if current_street == 'flop' and name == pfr_player and atype == 'bet':
                        pass  # tracked above
                    elif current_street == 'turn' and name == flop_cbettor and atype == 'bet':
                        pass  # tracked above
            continue

    for name in track:
        if name in player_vpip:
            stats_map[name].vpip_count += 1
        if name in player_pfr:
            stats_map[name].pfr_count += 1

    # CBet opportunity tracking
    if pfr_player and pfr_player in track and pfr_player not in player_folded:
        if hand.flop:
            st = stats_map[pfr_player]
            if flop_cbettor == pfr_player:
                st.cbet_opps += 1
                st.cbet_count += 1
            else:
                st.cbet_opps += 1

    if flop_cbettor and flop_cbettor in track and flop_cbettor not in player_folded:
        if hand.turn:
            st = stats_map[flop_cbettor]
            if turn_cbettor == flop_cbettor:
                st.turn_cbet_opps += 1
                st.turn_cbet_count += 1
            else:
                st.turn_cbet_opps += 1

    if turn_cbettor and turn_cbettor in track and turn_cbettor not in player_folded:
        if hand.river:
            st = stats_map[turn_cbettor]
            river_bet = any(
                n == turn_cbettor and at == 'bet'
                for (n, at, _, _) in street_action_list.get('river', []))
            st.river_cbet_opps += 1
            if river_bet:
                st.river_cbet_count += 1

    # Check-raise opportunities
    for street in ('flop', 'turn', 'river'):
        actions = street_action_list.get(street, [])
        checked_players = set()
        for (n, at, amt, ai) in actions:
            if at == 'check':
                checked_players.add(n)
            elif at in ('bet', 'raise'):
                for cp in checked_players:
                    if cp in track and cp != n and cp not in player_folded:
                        stats_map[cp].check_raise_opps += 1
                break

    # Showdown stats
    winners = {w['name'] for w in hand.result.winners}
    if hand.has_showdown:
        for name in track:
            if name in player_saw_flop and name not in player_folded:
                stats_map[name].wtsd_count += 1
                stats_map[name].wsd_opps += 1
                if name in winners:
                    stats_map[name].wsd_count += 1


# ── Model stat computation ─────────────────────────────────────────────────

def _compute_model_stats(
    hands: List[ParsedHand],
    model: ModelWrapper,
    obs_builder: ObservationBuilder,
    max_hands: int = 50000,
) -> Dict[str, float]:
    """Replay hands and simulate model decisions."""
    try:
        from src.preflop_ranges import PreflopRanges
        preflop_ranges = PreflopRanges()
    except Exception:
        preflop_ranges = None

    stats = PlayerStats()
    processed = 0

    for hand in hands:
        if processed >= max_hands:
            break

        active = hand.active_players
        if len(active) < 2:
            continue

        hero = None
        for p in active:
            if p.hole_cards and len(p.hole_cards) >= 2:
                hero = p
                break
        if hero is None:
            continue

        processed += 1
        stats.hands_dealt += 1

        _simulate_hero(hand, hero.name, stats, model, preflop_ranges, obs_builder)

    return stats.compute()


def _simulate_hero(hand, hero_name, stats, model, preflop_ranges, obs_builder):
    """Replay a hand simulating model decisions for hero."""
    active = hand.active_players
    num_players = len(active)
    name_to_idx = {p.name: i for i, p in enumerate(active)}
    hero_idx = name_to_idx.get(hero_name)
    if hero_idx is None:
        return

    hero_player = active[hero_idx]
    if not hero_player.hole_cards or len(hero_player.hole_cards) < 2:
        return

    button_idx = 0
    for i, p in enumerate(active):
        if p.seat == hand.button_seat:
            button_idx = i
            break
    rel_pos = (hero_idx - button_idx) % num_players

    big_blind_chips = dollars_to_chips(hand.big_blind_dollars)

    dps = obs_builder.build_decision_points(
        hand, allow_unknown_cards=False, behavioral_stats=True)
    hero_dps = [dp for dp in dps if dp.player_name == hero_name]

    hero_folded = False
    pfr_player = None
    preflop_raise_count = 0
    hero_is_pfr = False
    hero_saw_flop = False
    hero_checked_this_street = False
    flop_cbettor = None
    turn_cbettor = None
    hero_vpip = False
    hero_pfr = False
    current_street = 'preflop'

    pot_chips = 0
    max_bet = 0
    street_bets = defaultdict(int)
    street_first_bettor = {}
    dp_index = 0

    for action in hand.actions:
        aname = action.player_name
        if aname not in name_to_idx:
            continue

        if action.street != current_street:
            pot_chips += sum(street_bets.values())
            street_bets.clear()
            max_bet = 0
            hero_checked_this_street = False
            street_first_bettor = {}
            if current_street == 'preflop' and not hero_folded:
                hero_saw_flop = True
                stats.saw_flop += 1
            current_street = action.street

        if action.is_blind:
            street_bets[aname] += action.amount_chips
            if action.amount_chips > max_bet:
                max_bet = action.amount_chips
            continue

        if aname != hero_name:
            if action.action_type in ('bet', 'raise'):
                if current_street == 'preflop':
                    preflop_raise_count += 1
                    if preflop_raise_count == 1:
                        pfr_player = aname
                if current_street not in street_first_bettor:
                    street_first_bettor[current_street] = aname
                new_bet = action.amount_chips
                street_bets[aname] = new_bet
                if new_bet > max_bet:
                    max_bet = new_bet
                if current_street == 'flop' and aname == pfr_player and action.action_type == 'bet':
                    flop_cbettor = aname
                elif current_street == 'turn' and aname == flop_cbettor and action.action_type == 'bet':
                    turn_cbettor = aname
            elif action.action_type == 'call':
                street_bets[aname] = max_bet
            continue

        if hero_folded:
            continue

        if dp_index < len(hero_dps):
            dp = hero_dps[dp_index]
            dp_index += 1
            obs = dp.observation
            legal_actions = dp.legal_actions
        else:
            continue

        # Decide action
        if current_street == 'preflop' and preflop_ranges:
            hole = hero_player.hole_cards
            pot_bb = (pot_chips + sum(street_bets.values())) / big_blind_chips
            model_action = preflop_ranges.get_action(
                hole, rel_pos, pot_bb, legal_actions)
            if model_action is None or model_action not in legal_actions:
                model_action = FOLD if FOLD in legal_actions else CALL
            if model_action == 5:  # RAISE_MIN -> RAISE_HALF
                model_action = RAISE_HALF
        else:
            model_action, _ = model.get_action(obs, legal_actions, player_idx=0)

        is_fold = (model_action == FOLD)
        is_call = (model_action == CALL)
        is_raise = model_action in (RAISE_HALF, RAISE_POT, ALL_IN)

        if current_street == 'preflop':
            if is_raise:
                hero_vpip = True
                hero_pfr = True
                preflop_raise_count += 1
                if preflop_raise_count == 1:
                    pfr_player = hero_name
                    hero_is_pfr = True
                elif preflop_raise_count == 2:
                    stats.three_bet_opps += 1
                    stats.three_bet_count += 1
            elif is_call:
                hero_vpip = True
                if preflop_raise_count >= 2:
                    stats.fold_to_3bet_opps += 1
            elif is_fold:
                hero_folded = True
                if preflop_raise_count >= 2:
                    stats.fold_to_3bet_opps += 1
                    stats.fold_to_3bet_count += 1
        else:
            if is_fold:
                hero_folded = True
                stats.fold_to_bet[current_street]['opps'] += 1
                stats.fold_to_bet[current_street]['count'] += 1
                if current_street == 'flop' and pfr_player and pfr_player != hero_name:
                    if street_first_bettor.get('flop') == pfr_player:
                        stats.fold_to_cbet_opps += 1
                        stats.fold_to_cbet_count += 1
            elif is_call:
                stats.agg[current_street]['calls'] += 1
                if max_bet > street_bets.get(hero_name, 0):
                    stats.fold_to_bet[current_street]['opps'] += 1
                if current_street == 'flop' and pfr_player and pfr_player != hero_name:
                    if street_first_bettor.get('flop') == pfr_player:
                        stats.fold_to_cbet_opps += 1
                street_bets[hero_name] = max_bet
            elif is_raise:
                if model_action in (RAISE_HALF, RAISE_POT):
                    stats.agg[current_street][
                        'bets' if max_bet <= street_bets.get(hero_name, 0) else 'raises'] += 1
                else:
                    stats.agg[current_street]['raises'] += 1
                if max_bet > street_bets.get(hero_name, 0):
                    stats.fold_to_bet[current_street]['opps'] += 1
                if hero_checked_this_street:
                    stats.check_raise_opps += 1
                    stats.check_raise_count += 1
                if current_street == 'flop' and hero_is_pfr:
                    if current_street not in street_first_bettor:
                        stats.cbet_opps += 1
                        stats.cbet_count += 1
                        flop_cbettor = hero_name
                elif current_street == 'turn' and flop_cbettor == hero_name:
                    if current_street not in street_first_bettor:
                        stats.turn_cbet_opps += 1
                        stats.turn_cbet_count += 1
                        turn_cbettor = hero_name
                elif current_street == 'river' and turn_cbettor == hero_name:
                    if current_street not in street_first_bettor:
                        stats.river_cbet_opps += 1
                        stats.river_cbet_count += 1

                cur_pot = pot_chips + sum(street_bets.values())
                if cur_pot > 0:
                    if model_action == RAISE_HALF:
                        stats.bet_sizes.append(0.5)
                    elif model_action == RAISE_POT:
                        stats.bet_sizes.append(1.0)
                    elif model_action == ALL_IN:
                        stats.bet_sizes.append(2.0)

                if current_street not in street_first_bettor:
                    street_first_bettor[current_street] = hero_name

            if is_call and not (max_bet > street_bets.get(hero_name, 0)):
                hero_checked_this_street = True

        # Update hero bet tracking
        if is_raise:
            street_bets[hero_name] = max_bet + 1
            if street_bets[hero_name] > max_bet:
                max_bet = street_bets[hero_name]
        elif is_call:
            street_bets[hero_name] = max_bet

        # Apply ACTUAL action for realistic game state continuation
        atype = action.action_type
        if atype in ('bet', 'raise'):
            street_bets[hero_name] = action.amount_chips
            if action.amount_chips > max_bet:
                max_bet = action.amount_chips
            if current_street not in street_first_bettor:
                street_first_bettor[current_street] = hero_name
            if current_street == 'flop' and hero_is_pfr and atype == 'bet':
                if flop_cbettor is None:
                    flop_cbettor = hero_name
            elif current_street == 'turn' and flop_cbettor == hero_name and atype == 'bet':
                if turn_cbettor is None:
                    turn_cbettor = hero_name
        elif atype == 'call':
            street_bets[hero_name] = max_bet
        elif atype == 'check':
            hero_checked_this_street = True

    # CBet opportunity tracking
    if hero_is_pfr and hero_saw_flop and hand.flop and flop_cbettor != hero_name:
        stats.cbet_opps += 1
    if flop_cbettor == hero_name and hand.turn and turn_cbettor != hero_name:
        stats.turn_cbet_opps += 1
    if turn_cbettor == hero_name and hand.river:
        stats.river_cbet_opps += 1

    if hero_vpip:
        stats.vpip_count += 1
    if hero_pfr:
        stats.pfr_count += 1

    winners = {w['name'] for w in hand.result.winners}
    if hand.has_showdown and hero_saw_flop and not hero_folded:
        stats.wtsd_count += 1
        stats.wsd_opps += 1
        if hero_name in winners:
            stats.wsd_count += 1


# ── Aggregation ─────────────────────────────────────────────────────────────

def _aggregate_population(stats_map: Dict[str, PlayerStats],
                          min_hands: int = 100) -> Dict[str, float]:
    pop = PlayerStats()
    count = 0
    for name, st in stats_map.items():
        if st.hands_dealt < min_hands:
            continue
        count += 1
        pop.hands_dealt += st.hands_dealt
        pop.vpip_count += st.vpip_count
        pop.pfr_count += st.pfr_count
        pop.three_bet_opps += st.three_bet_opps
        pop.three_bet_count += st.three_bet_count
        pop.fold_to_3bet_opps += st.fold_to_3bet_opps
        pop.fold_to_3bet_count += st.fold_to_3bet_count
        pop.cbet_opps += st.cbet_opps
        pop.cbet_count += st.cbet_count
        pop.fold_to_cbet_opps += st.fold_to_cbet_opps
        pop.fold_to_cbet_count += st.fold_to_cbet_count
        pop.turn_cbet_opps += st.turn_cbet_opps
        pop.turn_cbet_count += st.turn_cbet_count
        pop.river_cbet_opps += st.river_cbet_opps
        pop.river_cbet_count += st.river_cbet_count
        for street in ('flop', 'turn', 'river'):
            for key in ('bets', 'raises', 'calls'):
                pop.agg[street][key] += st.agg[street][key]
        pop.saw_flop += st.saw_flop
        pop.wtsd_count += st.wtsd_count
        pop.wsd_opps += st.wsd_opps
        pop.wsd_count += st.wsd_count
        for street in ('flop', 'turn', 'river'):
            pop.fold_to_bet[street]['opps'] += st.fold_to_bet[street]['opps']
            pop.fold_to_bet[street]['count'] += st.fold_to_bet[street]['count']
        pop.check_raise_opps += st.check_raise_opps
        pop.check_raise_count += st.check_raise_count
        pop.bet_sizes.extend(st.bet_sizes)

    print(f'    Population: {count} players, {pop.hands_dealt} total hands')
    return pop.compute()


# ── Scoring ─────────────────────────────────────────────────────────────────

_STAT_NAMES = [
    'VPIP', 'PFR', '3Bet%', 'Fold_to_3Bet%',
    'CBet%', 'Fold_to_CBet%', 'Turn_CBet%', 'River_CBet%',
    'AF_flop', 'AF_turn', 'AF_river',
    'WTSD%', 'WSD%',
    'Fold_to_Bet_flop%', 'Fold_to_Bet_turn%', 'Fold_to_Bet_river%',
    'CheckRaise%', 'Avg_Bet_Size',
]


def _score_model(model_stats, good_stats, bad_stats):
    scores = {}
    valid_count = 0
    total_score = 0.0

    for stat in _STAT_NAMES:
        m = model_stats.get(stat, 0.0)
        g = good_stats.get(stat, 0.0)
        b = bad_stats.get(stat, 0.0)

        if abs(g - b) < 0.5:
            scores[stat] = 0.0
            continue

        dg = abs(m - g)
        db = abs(m - b)
        denom = dg + db + 1e-9
        score = (db - dg) / denom
        scores[stat] = score
        total_score += score
        valid_count += 1

    if valid_count > 0:
        avg_score = total_score / valid_count
        profile_pct = (avg_score + 1.0) / 2.0 * 100.0
    else:
        profile_pct = 50.0

    return scores, profile_pct


# ── Main analysis function ──────────────────────────────────────────────────

def analyze(
    db: AnalysisDB,
    model: ModelWrapper,
    good_players: Optional[Set[str]] = None,
    bad_players: Optional[Set[str]] = None,
    min_player_hands: int = 100,
    max_model_hands: int = 50000,
    **kwargs,
) -> dict:
    """Run population stats analysis.

    Uses db.all_hands if available (includes non-showdown hands),
    otherwise falls back to db.hands (showdown only).

    Returns:
        Dictionary with population stats, model stats, and scoring.
    """
    hands = db.all_hands if db.all_hands else db.hands
    print(f'  Using {len(hands)} hands for population analysis')

    # Compute population stats
    print('  Computing good player stats...')
    good_stats_map: Dict[str, PlayerStats] = {}
    for hand in hands:
        _analyze_hand_population(hand, good_players, good_stats_map)
    good_stats = _aggregate_population(good_stats_map, min_player_hands)

    print('  Computing bad player stats...')
    bad_stats_map: Dict[str, PlayerStats] = {}
    for hand in hands:
        _analyze_hand_population(hand, bad_players, bad_stats_map)
    bad_stats = _aggregate_population(bad_stats_map, min_player_hands)

    print('  Computing all player stats...')
    all_stats_map: Dict[str, PlayerStats] = {}
    for hand in hands:
        _analyze_hand_population(hand, None, all_stats_map)
    all_stats = _aggregate_population(all_stats_map, min_player_hands)

    # Compute model stats
    print('  Computing model stats...')
    obs_builder = ObservationBuilder(num_players=6)
    hands_with_cards = [h for h in hands if h.shown_hands]
    print(f'    Hands with known cards: {len(hands_with_cards)}')

    model_stats = _compute_model_stats(
        hands_with_cards, model, obs_builder,
        max_hands=max_model_hands)

    # Score
    scores, profile_pct = _score_model(model_stats, good_stats, bad_stats)

    return {
        'model_stats': model_stats,
        'good_stats': good_stats,
        'bad_stats': bad_stats,
        'all_stats': all_stats,
        'scores': scores,
        'profile_pct': profile_pct,
        'stat_names': _STAT_NAMES,
    }


# ── Report printing ─────────────────────────────────────────────────────────

def print_report(results: dict):
    """Print the population stats comparison report."""
    model_stats = results['model_stats']
    good_stats = results['good_stats']
    bad_stats = results['bad_stats']
    all_stats = results['all_stats']
    scores = results['scores']
    profile_pct = results['profile_pct']

    hdr = f'{"Stat":<25s}  {"Model":>8s}  {"Good":>8s}  {"Bad":>8s}  {"All":>8s}  {"Score":>7s}'
    sep = '-' * len(hdr)

    print()
    print(sep)
    print('  POPULATION STATS COMPARISON')
    print(sep)
    print(hdr)
    print(sep)

    for stat in _STAT_NAMES:
        m = model_stats.get(stat, 0.0)
        g = good_stats.get(stat, 0.0)
        b = bad_stats.get(stat, 0.0)
        a = all_stats.get(stat, 0.0)
        s = scores.get(stat, 0.0)

        if stat.startswith('AF_') or stat == 'Avg_Bet_Size':
            fmt = '{:>8.2f}'
        else:
            fmt = '{:>7.1f}%'

        m_str = fmt.format(m)
        g_str = fmt.format(g)
        b_str = fmt.format(b)
        a_str = fmt.format(a)

        if abs(s) < 0.01:
            s_str = '   --  '
        else:
            s_str = f'  {s:+.2f}'

        print(f'{stat:<25s}  {m_str}  {g_str}  {b_str}  {a_str}  {s_str}')

    print(sep)
    print(f'  Winner Profile Match: {profile_pct:.1f}%')
    if profile_pct >= 70:
        label = 'STRONG match to winning players'
    elif profile_pct >= 55:
        label = 'Moderate match to winning players'
    elif profile_pct >= 45:
        label = 'Neutral - between good and bad'
    else:
        label = 'Closer to losing player profile'
    print(f'  Assessment: {label}')
    print(sep)

    # Per-stat breakdown
    print('\nPer-stat analysis (model vs good players):')
    for stat in _STAT_NAMES:
        m = model_stats.get(stat, 0.0)
        g = good_stats.get(stat, 0.0)
        diff = m - g
        if stat.startswith('AF_') or stat == 'Avg_Bet_Size':
            print(f'  {stat:<25s}  model={m:.2f}  good={g:.2f}  delta={diff:+.2f}')
        else:
            print(f'  {stat:<25s}  model={m:.1f}%  good={g:.1f}%  delta={diff:+.1f}%')
    print()
