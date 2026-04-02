"""Compare model aggregate poker statistics against player populations.

Parses hand histories, computes stats for good/bad/all players, runs the
model through the same hands, and scores how closely the model matches
winning play.

Usage:
  python3 scripts/analysis/population_stats.py \
    --model-facing-bet models/best/cql_6p_facing_bet_v7.pt \
    --model-checked-to models/best/cql_6p_checked_to_v3.pt \
    --max-hands 50000
"""

import sys
import os
import argparse
import warnings
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Project root = parent of scripts/
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from scripts.train_cql import QNetwork, CQLModelWrapper
from src.hand_history.parser import parse_file
from src.hand_history import ObservationBuilder
from src.hand_history.models import ParsedHand, ParsedAction, dollars_to_chips
from src.hand_history.action_mapper import (
    map_action, get_legal_actions, FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN,
)
from src.preflop_ranges import PreflopRanges


# ---------------------------------------------------------------------------
# Stat tracker
# ---------------------------------------------------------------------------

class PlayerStats:
    """Accumulates poker statistics for a single player or population."""

    def __init__(self):
        # Basic counters
        self.hands_dealt = 0
        self.vpip_count = 0       # voluntarily put money in pot
        self.pfr_count = 0        # preflop raise
        self.three_bet_opps = 0   # opportunities to 3-bet
        self.three_bet_count = 0
        self.fold_to_3bet_opps = 0
        self.fold_to_3bet_count = 0

        # Cbet
        self.cbet_opps = 0        # was pfr and acts on flop
        self.cbet_count = 0
        self.fold_to_cbet_opps = 0
        self.fold_to_cbet_count = 0
        self.turn_cbet_opps = 0
        self.turn_cbet_count = 0
        self.river_cbet_opps = 0
        self.river_cbet_count = 0

        # Aggression per street: {street: {'bets': n, 'raises': n, 'calls': n}}
        self.agg = defaultdict(lambda: {'bets': 0, 'raises': 0, 'calls': 0})

        # Showdown
        self.saw_flop = 0
        self.wtsd_count = 0       # went to showdown (from those who saw flop)
        self.wsd_opps = 0         # showdown opportunities
        self.wsd_count = 0        # won at showdown

        # Fold to bet per street
        self.fold_to_bet = defaultdict(lambda: {'opps': 0, 'count': 0})

        # Check-raise
        self.check_raise_opps = 0
        self.check_raise_count = 0

        # Bet sizing (relative to pot)
        self.bet_sizes = []       # list of (amount / pot) ratios

    def compute(self) -> Dict[str, float]:
        """Return computed stat dict. Values are percentages or ratios."""
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
        stats['Turn_CBet%'] = _safe_pct(self.turn_cbet_count,
                                        self.turn_cbet_opps)
        stats['River_CBet%'] = _safe_pct(self.river_cbet_count,
                                         self.river_cbet_opps)

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
        stats['Avg_Bet_Size'] = (
            np.mean(self.bet_sizes) if self.bet_sizes else 0.0)

        return stats


def _safe_pct(num: int, den: int) -> float:
    if den == 0:
        return 0.0
    return 100.0 * num / den


# ---------------------------------------------------------------------------
# Hand analysis: extract stats from a parsed hand for given players
# ---------------------------------------------------------------------------

def _analyze_hand(hand: ParsedHand,
                  target_names: Optional[Set[str]],
                  stats_map: Dict[str, PlayerStats]):
    """Update stats_map for all target players in this hand.

    target_names=None means track ALL players.
    """
    active = hand.active_players
    if len(active) < 2:
        return

    active_names = set(p.name for p in active)
    big_blind_chips = dollars_to_chips(hand.big_blind_dollars)

    # Identify which players to track in this hand
    if target_names is None:
        track = active_names
    else:
        track = active_names & target_names
    if not track:
        return

    # Ensure each tracked player has a stats object
    for name in track:
        if name not in stats_map:
            stats_map[name] = PlayerStats()
        stats_map[name].hands_dealt += 1

    # ── Replay actions to extract stats ──────────────────────────────────

    player_folded = set()
    pfr_player = None            # first preflop raiser (for cbet tracking)
    preflop_raise_count = 0      # total raises on preflop (ALL players)
    player_checked_this_street = defaultdict(bool)
    player_vpip = set()
    player_pfr = set()
    player_saw_flop = set()
    street_first_bettor = {}     # street -> name of first bettor
    flop_cbettor = None
    turn_cbettor = None

    current_street = 'preflop'
    pot_chips = 0
    street_bets = defaultdict(int)
    max_bet = 0

    player_stakes = {}
    for p in active:
        player_stakes[p.name] = p.stack_chips

    # Actions by street for check-raise opportunity pass
    street_action_list = defaultdict(list)

    for action in hand.actions:
        name = action.player_name
        if name not in active_names:
            continue

        # ── Street transition ────────────────────────────────────
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

        # ── Blinds ───────────────────────────────────────────────
        if action.is_blind:
            amt = action.amount_chips
            street_bets[name] += amt
            if amt > max_bet:
                max_bet = amt
            player_stakes[name] -= min(amt, player_stakes.get(name, 0))
            continue

        atype = action.action_type
        street_action_list[current_street].append(
            (name, atype, action.amount_chips, action.is_all_in))

        facing_bet = max_bet > street_bets.get(name, 0)
        is_tracked = name in track

        # ──────────────────────────────────────────────────────────
        # FOLD
        # ──────────────────────────────────────────────────────────
        if atype == 'fold':
            player_folded.add(name)
            if is_tracked:
                st = stats_map[name]
                # Fold to bet (postflop)
                if current_street != 'preflop' and facing_bet:
                    st.fold_to_bet[current_street]['opps'] += 1
                    st.fold_to_bet[current_street]['count'] += 1
                # Fold to 3bet (preflop, facing 2+ raises)
                if current_street == 'preflop' and preflop_raise_count >= 2:
                    st.fold_to_3bet_opps += 1
                    st.fold_to_3bet_count += 1
                # Fold to cbet
                if (current_street == 'flop' and pfr_player
                        and pfr_player != name
                        and street_first_bettor.get('flop') == pfr_player):
                    st.fold_to_cbet_opps += 1
                    st.fold_to_cbet_count += 1
            continue

        # ──────────────────────────────────────────────────────────
        # CHECK
        # ──────────────────────────────────────────────────────────
        if atype == 'check':
            player_checked_this_street[name] = True
            continue

        # ──────────────────────────────────────────────────────────
        # CALL
        # ──────────────────────────────────────────────────────────
        if atype == 'call':
            street_bets[name] = max_bet
            player_stakes[name] -= min(
                max(0, max_bet - street_bets.get(name, 0)),
                player_stakes.get(name, 0))

            if is_tracked:
                st = stats_map[name]
                if current_street == 'preflop':
                    player_vpip.add(name)
                    # 3bet opportunity: facing exactly 1 raise, could 3bet but called
                    if preflop_raise_count == 1:
                        st.three_bet_opps += 1
                    # Fold to 3bet: facing 2+ raises, called (did not fold)
                    if preflop_raise_count >= 2:
                        st.fold_to_3bet_opps += 1
                else:
                    st.agg[current_street]['calls'] += 1
                    # Fold to bet: had opportunity, didn't fold
                    if facing_bet:
                        st.fold_to_bet[current_street]['opps'] += 1
                    # Fold to cbet: opp, didn't fold
                    if (current_street == 'flop' and pfr_player
                            and pfr_player != name
                            and street_first_bettor.get('flop') == pfr_player):
                        st.fold_to_cbet_opps += 1
            continue

        # ──────────────────────────────────────────────────────────
        # BET / RAISE
        # ──────────────────────────────────────────────────────────
        if atype in ('bet', 'raise'):
            old_max = max_bet
            new_bet = action.amount_chips
            street_bets[name] = new_bet
            if new_bet > max_bet:
                max_bet = new_bet

            # Track game state (all players)
            if current_street == 'preflop' and atype == 'raise':
                preflop_raise_count += 1
                if preflop_raise_count == 1:
                    pfr_player = name

            if current_street not in street_first_bettor:
                street_first_bettor[current_street] = name

            # Cbet tracking for non-tracked pfr
            if current_street == 'flop' and name == pfr_player and atype == 'bet':
                flop_cbettor = name
            elif current_street == 'turn' and name == flop_cbettor and atype == 'bet':
                turn_cbettor = name

            if is_tracked:
                st = stats_map[name]

                if current_street == 'preflop':
                    player_vpip.add(name)
                    player_pfr.add(name)

                    if preflop_raise_count == 1:
                        pass  # open raise, no 3bet opp for raiser
                    elif preflop_raise_count == 2:
                        # This is a 3bet
                        st.three_bet_opps += 1
                        st.three_bet_count += 1
                    # Fold to 3bet: facing 2+ raises, raised (did not fold)
                    if preflop_raise_count >= 3:
                        st.fold_to_3bet_opps += 1
                else:
                    st.agg[current_street][
                        'bets' if atype == 'bet' else 'raises'] += 1

                    # Fold to bet: had opportunity but raised instead
                    if facing_bet:
                        st.fold_to_bet[current_street]['opps'] += 1

                    # Check-raise
                    if player_checked_this_street.get(name, False):
                        st.check_raise_count += 1

                    # Bet sizing
                    current_pot = pot_chips + sum(street_bets.values())
                    if current_pot > 0 and action.amount_chips > 0:
                        ratio = action.amount_chips / current_pot
                        st.bet_sizes.append(ratio)

                    # CBet detection (tracked pfr)
                    if current_street == 'flop' and name == pfr_player:
                        if atype == 'bet':
                            flop_cbettor = name
                    elif current_street == 'turn' and name == flop_cbettor:
                        if atype == 'bet':
                            turn_cbettor = name

            continue

    # ── Post-replay: VPIP / PFR ──────────────────────────────────────
    for name in track:
        if name in player_vpip:
            stats_map[name].vpip_count += 1
        if name in player_pfr:
            stats_map[name].pfr_count += 1

    # ── 3bet opportunity: tracked players who faced exactly 1 raise
    #    and folded (not already counted above) ───────────────────────
    # Already counted inline for call/raise/fold cases.

    # ── CBet opportunity tracking ────────────────────────────────────
    # PFR saw flop and was first to act (or checked): opportunity exists
    if pfr_player and pfr_player in track and pfr_player not in player_folded:
        if hand.flop:
            st = stats_map[pfr_player]
            if flop_cbettor == pfr_player:
                # Already counted the cbet in the bet detection
                st.cbet_opps += 1
                st.cbet_count += 1
            else:
                # Had opportunity, didn't cbet
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
            # Check if this player bet the river (need to look at street actions)
            river_bet = any(
                n == turn_cbettor and at == 'bet'
                for (n, at, _, _) in street_action_list.get('river', []))
            st.river_cbet_opps += 1
            if river_bet:
                st.river_cbet_count += 1

    # ── Check-raise opportunities (second pass) ─────────────────────
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
                break  # only first bet matters for check-raise opportunity

    # ── Showdown stats ───────────────────────────────────────────────
    winners = {w['name'] for w in hand.result.winners}
    if hand.has_showdown:
        for name in track:
            if name in player_saw_flop and name not in player_folded:
                stats_map[name].wtsd_count += 1
                stats_map[name].wsd_opps += 1
                if name in winners:
                    stats_map[name].wsd_count += 1


# ---------------------------------------------------------------------------
# Model stat computation: simulate model decisions on parsed hands
# ---------------------------------------------------------------------------

def _load_model(model_path: str, device: str) -> Tuple:
    """Load a CQL model checkpoint. Returns (q_net, obs_size, action_size, meta)."""
    ckpt = torch.load(model_path, map_location=device)
    obs_size = ckpt['obs_size']
    action_size = ckpt['action_size']
    hidden_layers = ckpt['hidden_layers']
    dropout = ckpt.get('dropout', 0.1)

    q_net = QNetwork(obs_size, action_size, hidden_layers, dropout)
    q_net.load_state_dict(ckpt['model_state_dict'])
    q_net.to(device)
    q_net.eval()

    temperature = ckpt.get('temperature', 1.0)
    num_players = ckpt.get('num_players', 6)

    wrapper = CQLModelWrapper(
        q_net, obs_size, action_size, num_players, device,
        temperature=temperature)

    return wrapper, obs_size, action_size, ckpt


def _compute_model_stats(
    hands: List[ParsedHand],
    wrapper_fb,           # CQLModelWrapper for facing_bet
    wrapper_ct,           # CQLModelWrapper for checked_to
    preflop_ranges: PreflopRanges,
    obs_builder: ObservationBuilder,
    max_hands: int = 50000,
) -> Dict[str, float]:
    """Replay hands and simulate model decisions. Return aggregated stats."""

    stats = PlayerStats()
    processed = 0

    for hand in hands:
        if processed >= max_hands:
            break

        active = hand.active_players
        if len(active) < 2:
            continue

        # We simulate the model as each player in the hand
        # But to be consistent, we pick one "hero" per hand (first active player
        # with known hole cards, or if none, skip)
        hero = None
        for p in active:
            if p.hole_cards and len(p.hole_cards) >= 2:
                hero = p
                break
        if hero is None:
            continue

        processed += 1
        stats.hands_dealt += 1

        # Replay the hand to build observations at each hero decision point
        _simulate_hero_in_hand(hand, hero.name, stats, wrapper_fb, wrapper_ct,
                               preflop_ranges, obs_builder)

    return stats.compute()


def _simulate_hero_in_hand(
    hand: ParsedHand,
    hero_name: str,
    stats: PlayerStats,
    wrapper_fb,
    wrapper_ct,
    preflop_ranges: PreflopRanges,
    obs_builder: ObservationBuilder,
):
    """Replay a hand, and at each hero decision point, use the model to choose
    an action. Update stats accordingly."""

    active = hand.active_players
    num_players = len(active)
    name_to_idx = {p.name: i for i, p in enumerate(active)}
    hero_idx = name_to_idx.get(hero_name)
    if hero_idx is None:
        return

    hero_player = active[hero_idx]
    if not hero_player.hole_cards or len(hero_player.hole_cards) < 2:
        return

    # Find button
    button_idx = 0
    for i, p in enumerate(active):
        if p.seat == hand.button_seat:
            button_idx = i
            break
    rel_pos = (hero_idx - button_idx) % num_players

    big_blind_chips = dollars_to_chips(hand.big_blind_dollars)
    big_blind_dollars = hand.big_blind_dollars

    # Build decision points using the obs builder (which handles full replay)
    dps = obs_builder.build_decision_points(
        hand, allow_unknown_cards=False, behavioral_stats=True)

    # Filter to hero's decision points
    hero_dps = [dp for dp in dps if dp.player_name == hero_name]

    # Also need to replay actions to know the game state at each decision
    # We use the DP info directly

    # Track hero state
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

    # We need to replay ALL actions (not just hero's) to track game state
    pot_chips = 0
    max_bet = 0
    street_bets = defaultdict(int)
    street_first_bettor = {}

    dp_index = 0

    for action in hand.actions:
        aname = action.player_name
        if aname not in name_to_idx:
            continue

        # Street transition
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

        # Blinds: update pot tracking
        if action.is_blind:
            street_bets[aname] += action.amount_chips
            if action.amount_chips > max_bet:
                max_bet = action.amount_chips
            continue

        # Non-hero actions: just track state
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
            elif action.action_type == 'fold':
                pass
            continue

        # ── Hero's turn: use model to decide ─────────────────────
        if hero_folded:
            continue

        # Get the observation from the DP if available
        if dp_index < len(hero_dps):
            dp = hero_dps[dp_index]
            dp_index += 1
            obs = dp.observation
            legal_actions = dp.legal_actions
        else:
            # No more DPs (shouldn't happen normally)
            continue

        # Decide action
        if current_street == 'preflop':
            # Use preflop ranges
            hole = hero_player.hole_cards
            pot_bb = (pot_chips + sum(street_bets.values())) / big_blind_chips
            num_limpers = sum(
                1 for n, b in street_bets.items()
                if n != hero_name and b == big_blind_chips
                and n in name_to_idx)

            model_action = preflop_ranges.get_action(
                hole, rel_pos, pot_bb, legal_actions,
                num_limpers=num_limpers)

            if model_action is None or model_action not in legal_actions:
                model_action = FOLD if FOLD in legal_actions else CALL

            # Map RAISE_MIN to RAISE_HALF for stat purposes
            if model_action == 5:  # RAISE_MIN
                model_action = RAISE_HALF
        else:
            # Postflop: use split model
            # obs[10] is pot_odds — nonzero means facing a bet
            facing_bet = obs[10] > 0.001 if len(obs) > 10 else False
            pot_bb = obs[8] * 600 if len(obs) > 8 else 0

            if facing_bet:
                strategy = wrapper_fb.get_strategy(obs, 0, legal_actions)
            else:
                strategy = wrapper_ct.get_strategy(obs, 0, legal_actions)

            # Sample from strategy
            model_action = np.random.choice(len(strategy), p=strategy)

        # ── Record stats for model_action ────────────────────────

        # Map model action to stat categories
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
                    # Did not fold
            elif is_fold:
                hero_folded = True
                if preflop_raise_count >= 2:
                    stats.fold_to_3bet_opps += 1
                    stats.fold_to_3bet_count += 1
        else:
            # Postflop stats
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
                    stats.agg[current_street]['bets' if max_bet <= street_bets.get(hero_name, 0) else 'raises'] += 1
                else:
                    stats.agg[current_street]['raises'] += 1

                if max_bet > street_bets.get(hero_name, 0):
                    stats.fold_to_bet[current_street]['opps'] += 1

                # Check-raise
                if hero_checked_this_street:
                    stats.check_raise_opps += 1
                    stats.check_raise_count += 1

                # CBet
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

                # Bet sizing estimate
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

            # Check tracking (model checked = called when not facing bet)
            if is_call and not (max_bet > street_bets.get(hero_name, 0)):
                hero_checked_this_street = True

        # Update hero bet tracking
        if is_raise:
            street_bets[hero_name] = max_bet + 1  # approximate
            if street_bets[hero_name] > max_bet:
                max_bet = street_bets[hero_name]
        elif is_call:
            street_bets[hero_name] = max_bet

        # Apply actual hand action for other players (we continue replay)
        # The actual action in the hand happened regardless of model choice
        # but we track MODEL decisions, not actual. So we use model_action
        # for hero stats, but the hand continues as-is for game state.
        # (This means model stats reflect the model's TENDENCY, not a
        #  full counterfactual simulation.)

        # For realistic game state, apply the ACTUAL action
        atype = action.action_type
        if atype in ('bet', 'raise'):
            street_bets[hero_name] = action.amount_chips
            if action.amount_chips > max_bet:
                max_bet = action.amount_chips
            if current_street not in street_first_bettor:
                street_first_bettor[current_street] = hero_name
            if current_street == 'preflop':
                # Already counted
                pass
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

    # ── CBet opportunity tracking (if hero was pfr but didn't bet flop) ──
    if hero_is_pfr and hero_saw_flop and hand.flop and flop_cbettor != hero_name:
        stats.cbet_opps += 1
    if flop_cbettor == hero_name and hand.turn and turn_cbettor != hero_name:
        stats.turn_cbet_opps += 1
    if turn_cbettor == hero_name and hand.river:
        stats.river_cbet_opps += 1

    # ── VPIP / PFR ───────────────────────────────────────────────────
    if hero_vpip:
        stats.vpip_count += 1
    if hero_pfr:
        stats.pfr_count += 1

    # ── Showdown stats ───────────────────────────────────────────────
    # Can only track WTSD for hands where hero wasn't modeled to fold
    # but the actual hand went to showdown
    winners = {w['name'] for w in hand.result.winners}
    if hand.has_showdown and hero_saw_flop and not hero_folded:
        stats.wtsd_count += 1
        stats.wsd_opps += 1
        if hero_name in winners:
            stats.wsd_count += 1


# ---------------------------------------------------------------------------
# Population aggregation
# ---------------------------------------------------------------------------

def _aggregate_population(stats_map: Dict[str, PlayerStats],
                          min_hands: int = 100) -> Dict[str, float]:
    """Aggregate per-player stats into a population average."""
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

    print(f'  Population: {count} players, {pop.hands_dealt} total hands')
    return pop.compute()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Stats where higher is unambiguously better/worse don't need special handling.
# The scoring just measures proximity to good vs bad players.

_STAT_NAMES = [
    'VPIP', 'PFR', '3Bet%', 'Fold_to_3Bet%',
    'CBet%', 'Fold_to_CBet%', 'Turn_CBet%', 'River_CBet%',
    'AF_flop', 'AF_turn', 'AF_river',
    'WTSD%', 'WSD%',
    'Fold_to_Bet_flop%', 'Fold_to_Bet_turn%', 'Fold_to_Bet_river%',
    'CheckRaise%', 'Avg_Bet_Size',
]


def _score_model(model_stats: Dict[str, float],
                 good_stats: Dict[str, float],
                 bad_stats: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """Score model's stats vs good/bad players.

    For each stat:
      distance_good = |model - good|
      distance_bad  = |model - bad|
      score = (distance_bad - distance_good) / (distance_bad + distance_good + eps)

    Score in [-1, 1]:  +1 = perfectly matches good, -1 = perfectly matches bad.

    Returns: (per_stat_scores, overall_profile_score_pct)
    """
    scores = {}
    valid_count = 0
    total_score = 0.0

    for stat in _STAT_NAMES:
        m = model_stats.get(stat, 0.0)
        g = good_stats.get(stat, 0.0)
        b = bad_stats.get(stat, 0.0)

        # Skip stats where good and bad are nearly identical
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

    # Overall: average score mapped to 0-100%
    if valid_count > 0:
        avg_score = total_score / valid_count   # in [-1, 1]
        profile_pct = (avg_score + 1.0) / 2.0 * 100.0  # map to [0, 100]
    else:
        profile_pct = 50.0

    return scores, profile_pct


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_table(model_stats: Dict[str, float],
                 good_stats: Dict[str, float],
                 bad_stats: Dict[str, float],
                 all_stats: Dict[str, float],
                 scores: Dict[str, float],
                 profile_pct: float):
    """Print a clean comparison table."""

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

        # Format: % stats with one decimal, ratios with two
        if stat.startswith('AF_') or stat == 'Avg_Bet_Size':
            fmt = '{:>8.2f}'
        else:
            fmt = '{:>7.1f}%'

        m_str = fmt.format(m)
        g_str = fmt.format(g)
        b_str = fmt.format(b)
        a_str = fmt.format(a)

        # Score indicator
        if abs(s) < 0.01:
            s_str = '   --  '
        elif s > 0.3:
            s_str = f'  +{s:+.2f}'[1:]
        elif s < -0.3:
            s_str = f'  {s:+.2f}'
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
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare model stats against player populations.')
    parser.add_argument('--model-facing-bet', required=True,
                        help='Path to facing_bet CQL model checkpoint')
    parser.add_argument('--model-checked-to', required=True,
                        help='Path to checked_to CQL model checkpoint')
    parser.add_argument('--model-facing-bet-3bet', type=str, default=None,
                        help='Model for facing-bet spots in 3bet pots (pot > 12BB)')
    parser.add_argument('--data-dir', default=str(_PROJECT_ROOT / 'data' / 'poker_archive'),
                        help='Directory with HH files')
    parser.add_argument('--good-players', default=str(_PROJECT_ROOT / 'data' / 'good_players.txt'),
                        help='File with good player names (one per line)')
    parser.add_argument('--bad-players', default=str(_PROJECT_ROOT / 'data' / 'bad_players.txt'),
                        help='File with bad player names (one per line)')
    parser.add_argument('--max-hands', type=int, default=50000,
                        help='Max hands to process')
    parser.add_argument('--max-files', type=int, default=0,
                        help='Max HH files to parse (0 = auto based on max-hands)')
    parser.add_argument('--min-player-hands', type=int, default=100,
                        help='Min hands per player for population stats')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=0,
                        help='Parsing workers (0=auto)')
    args = parser.parse_args()

    # ── Load player lists ────────────────────────────────────────────────
    def _load_names(path: str) -> Set[str]:
        names = set()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    names.add(line)
        return names

    good_names = _load_names(args.good_players)
    bad_names = _load_names(args.bad_players)
    print(f'Loaded {len(good_names)} good players, {len(bad_names)} bad players')

    # ── Parse hand histories ─────────────────────────────────────────────
    print(f'Parsing hand histories from {args.data_dir} ...')
    data_dir = Path(args.data_dir)
    hh_files = sorted(data_dir.rglob('*.txt'))
    print(f'  Found {len(hh_files)} files')

    # Estimate files needed: ~15 hands per file on average
    if args.max_files > 0:
        max_files = args.max_files
    else:
        # Parse enough files to likely hit max_hands
        # Be generous: 3x what we might need
        max_files = min(len(hh_files), max(args.max_hands // 5, 2000))

    from scripts.analysis._parse_utils import parallel_parse
    all_hands = parallel_parse(hh_files[:max_files], max_hands=args.max_hands * 3,
                                workers=getattr(args, 'workers', 0),
                                showdown_only=False)
    print(f'  Total: {len(all_hands)} hands')

    if not all_hands:
        print('ERROR: No hands found. Check data directory.')
        return

    # Limit hands
    if len(all_hands) > args.max_hands:
        np.random.seed(42)
        indices = np.random.choice(len(all_hands), args.max_hands, replace=False)
        all_hands = [all_hands[i] for i in sorted(indices)]
        print(f'  Sampled {len(all_hands)} hands')

    # ── Compute population stats ─────────────────────────────────────────
    print('\nComputing good player stats ...')
    good_stats_map: Dict[str, PlayerStats] = {}
    for hand in all_hands:
        _analyze_hand(hand, good_names, good_stats_map)
    good_stats = _aggregate_population(good_stats_map, args.min_player_hands)

    print('Computing bad player stats ...')
    bad_stats_map: Dict[str, PlayerStats] = {}
    for hand in all_hands:
        _analyze_hand(hand, bad_names, bad_stats_map)
    bad_stats = _aggregate_population(bad_stats_map, args.min_player_hands)

    print('Computing all player stats ...')
    all_stats_map: Dict[str, PlayerStats] = {}
    for hand in all_hands:
        _analyze_hand(hand, None, all_stats_map)
    all_stats = _aggregate_population(all_stats_map, args.min_player_hands)

    # ── Load models ──────────────────────────────────────────────────────
    print(f'\nLoading models ...')
    wrapper_fb, obs_fb, act_fb, _ = _load_model(args.model_facing_bet, args.device)
    wrapper_ct, obs_ct, act_ct, _ = _load_model(args.model_checked_to, args.device)
    print(f'  Facing-bet model: obs={obs_fb}, actions={act_fb}')
    print(f'  Checked-to model: obs={obs_ct}, actions={act_ct}')

    # ── Compute model stats ──────────────────────────────────────────────
    print('Computing model stats ...')
    preflop_ranges = PreflopRanges()
    obs_builder = ObservationBuilder(num_players=6)

    # Only use hands that have at least one showdown player with known cards
    hands_with_cards = [h for h in all_hands if h.shown_hands]
    print(f'  Hands with known cards: {len(hands_with_cards)}')

    model_stats = _compute_model_stats(
        hands_with_cards, wrapper_fb, wrapper_ct,
        preflop_ranges, obs_builder,
        max_hands=args.max_hands)

    # ── Score and display ────────────────────────────────────────────────
    scores, profile_pct = _score_model(model_stats, good_stats, bad_stats)
    _print_table(model_stats, good_stats, bad_stats, all_stats,
                 scores, profile_pct)

    # ── Per-stat breakdown ───────────────────────────────────────────────
    print('Per-stat analysis (model vs good players):')
    for stat in _STAT_NAMES:
        m = model_stats.get(stat, 0.0)
        g = good_stats.get(stat, 0.0)
        diff = m - g
        if stat.startswith('AF_') or stat == 'Avg_Bet_Size':
            print(f'  {stat:<25s}  model={m:.2f}  good={g:.2f}  delta={diff:+.2f}')
        else:
            print(f'  {stat:<25s}  model={m:.1f}%  good={g:.1f}%  delta={diff:+.1f}%')
    print()


    # Save captured output to file
    sys.stdout = sys.__stdout__
    with open(str(_outfile), 'w') as _f:
    print(f'Results saved to: {_outfile}')


if __name__ == '__main__':
    main()
