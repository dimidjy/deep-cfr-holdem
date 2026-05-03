#!/usr/bin/env python3
"""
Self-play analysis: run the model 6-max vs itself and compute core poker metrics.

Plays N hands using DecisionEngine (same pipeline as runtime agent), records every
action, then computes standard poker statistics:

  Preflop: VPIP, PFR, 3Bet%, Fold-to-3Bet%, RFI by position, open-raise sizes
  Postflop: CBet%, Fold-to-CBet%, AF (aggression factor), WTSD%, W$SD%
  Overall: Avg pot size, showdown rate, positional win rates

Usage:
  python3 scripts/analysis/self_play.py --hands 10000
  python3 scripts/analysis/self_play.py --hands 50000 --no-raise-randomize
  python3 scripts/analysis/self_play.py --hands 10000 --model-facing-bet models/fb.pt \
      --model-checked-to models/ct.pt
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _PROJECT_ROOT)

from src.decision_engine import (
    DecisionEngine, DecisionConfig, SpotContext,
    FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN,
    IDX_POT_ODDS, IDX_MADE_HAND, NUM_ACTIONS, ACTION_NAMES,
)
from src.engine import FastEnhancedPokerEngine
from src.archetypes import get_archetype_skew, ARCHETYPE_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# Hand notation
# ─────────────────────────────────────────────────────────────────────────────

_RANK_ORDER = '23456789TJQKA'
_RANK_IDX = {r: i for i, r in enumerate(_RANK_ORDER)}

_POS_LABELS_6 = {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'UTG', 4: 'MP', 5: 'CO'}


def _cards_to_combo(cards: List[str]) -> str:
    """Convert ['As', 'Kd'] to 'AKo' notation."""
    if not cards or len(cards) < 2:
        return '??'
    r1, s1 = cards[0][0].upper(), cards[0][1].lower()
    r2, s2 = cards[1][0].upper(), cards[1][1].lower()
    # Sort high card first
    if _RANK_IDX.get(r1, 0) < _RANK_IDX.get(r2, 0):
        r1, s1, r2, s2 = r2, s2, r1, s1
    if r1 == r2:
        return f'{r1}{r2}'
    elif s1 == s2:
        return f'{r1}{r2}s'
    else:
        return f'{r1}{r2}o'


def _hand_category(combo: str) -> str:
    """Classify hand into a category for range display."""
    if len(combo) == 2 and combo[0] == combo[1]:
        r = _RANK_IDX.get(combo[0], 0)
        if r >= _RANK_IDX['T']:
            return 'premium_pair'  # TT+
        elif r >= _RANK_IDX['7']:
            return 'medium_pair'  # 77-99
        else:
            return 'small_pair'  # 22-66
    if len(combo) == 3:
        r1, r2 = _RANK_IDX.get(combo[0], 0), _RANK_IDX.get(combo[1], 0)
        is_suited = combo[2] == 's'
        if r1 >= _RANK_IDX['T'] and r2 >= _RANK_IDX['T']:
            return 'broadway_s' if is_suited else 'broadway_o'
        if combo[0] == 'A':
            return 'ax_suited' if is_suited else 'ax_offsuit'
        if r1 - r2 <= 2 and is_suited:
            return 'suited_connector'
        return 'other_suited' if is_suited else 'other_offsuit'
    return 'unknown'


# ─────────────────────────────────────────────────────────────────────────────
# Per-player stats tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlayerStats:
    """Accumulates HUD-style stats for one seat across many hands."""
    hands: int = 0
    # Preflop
    vpip: int = 0               # voluntarily put $ in pot (call or raise, not BB check)
    pfr: int = 0                # preflop raise
    three_bet_opps: int = 0     # opportunities to 3bet
    three_bet: int = 0          # actually 3bet
    fold_to_3bet_opps: int = 0
    fold_to_3bet: int = 0
    open_raise_opps: int = 0    # first to act (no prior raise)
    open_raise: int = 0         # RFI (raise first in)
    limp: int = 0               # call preflop without prior raise
    # Postflop
    cbet_opps: int = 0          # was preflop raiser and first to act on flop
    cbet: int = 0               # actually cbet
    fold_to_cbet_opps: int = 0
    fold_to_cbet: int = 0
    # Aggression (postflop)
    postflop_bets: int = 0      # bets + raises postflop
    postflop_calls: int = 0
    postflop_folds: int = 0
    postflop_checks: int = 0
    # Showdown
    saw_flop: int = 0
    went_to_showdown: int = 0
    won_at_showdown: int = 0
    # Winnings
    total_won_bb: float = 0.0
    # By position
    vpip_by_pos: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    hands_by_pos: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    pfr_by_pos: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    won_by_pos: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    # By street
    actions_by_street: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    # Per-(position, action_history) range tracking — only populated when
    # --range-analysis is set. Key: (pos_label, hist) → combo → action_label → count
    # action_history ∈ {'open', 'limped', 'vs_raise', 'vs_3bet', 'vs_4bet+'}
    # action_label ∈ {'fold', 'call', 'raise'}
    range_grid: Dict[Tuple[str, str], Dict[str, Dict[str, int]]] = field(
        default_factory=lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))))


# ─────────────────────────────────────────────────────────────────────────────
# Archetype engine wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ArchetypeEngine:
    """Wraps a DecisionEngine, applying an archetype skew to its strategy output.

    Used for calibrating archetype multipliers: run self-play with this as
    engine_b, then check if the resulting stats match the target profile.
    """

    def __init__(self, base_engine: DecisionEngine, skew_fn):
        self._base = base_engine
        self._skew = skew_fn

    def decide(self, ctx):
        strategy = self._base.get_strategy(ctx)
        strategy = self._skew(strategy, ctx.legal_actions)
        strategy_list = [float(strategy[i]) for i in range(NUM_ACTIONS)]
        # Sample action from skewed strategy
        legal = ctx.legal_actions
        probs = np.array([strategy_list[a] for a in legal])
        s = probs.sum()
        if s > 0:
            probs /= s
        else:
            probs = np.ones(len(legal)) / len(legal)
        action = int(np.random.choice(legal, p=probs))
        return action, strategy_list

    def info(self):
        return f'{self._base.info()} + archetype_skew'


# ─────────────────────────────────────────────────────────────────────────────
# Self-play simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_self_play(engine: DecisionEngine, num_hands: int,
                  num_players: int = 6,
                  engine_b: DecisionEngine = None,
                  track_range_analysis: bool = False
                  ) -> Dict[int, PlayerStats]:
    """Play num_hands of poker with effective `num_players` seats.

    Engine is always 6-max so observation shape matches training (50 dims).
    Shorter tables are simulated by force-folding (6 - num_players) random
    non-blind seats preflop each hand — same mechanism as random_preflop_folds
    in training.

    If engine_b is provided: seats 0,2,4 use engine (A), seats 1,3,5 use engine_b (B).
    Returns dict of seat_index → PlayerStats.
    """
    from src.engine.factory import make_engine
    ENGINE_PLAYERS = 6  # always 6-max engine to keep obs shape consistent
    eng = make_engine(num_players=ENGINE_PLAYERS)
    num_force_fold = max(0, ENGINE_PLAYERS - num_players)

    stats = {i: PlayerStats() for i in range(ENGINE_PLAYERS)}
    t0 = time.time()

    for hand_idx in range(num_hands):
        state = eng.reset()
        if hand_idx % 2000 == 0 and hand_idx > 0:
            elapsed = time.time() - t0
            rate = hand_idx / elapsed
            print(f'  [{hand_idx}/{num_hands}] {rate:.0f} hands/s')

        # Track per-hand state
        button = eng._button
        hand_actions = []         # [(seat, stage, action_id), ...]
        preflop_raiser = None     # seat who was last preflop raiser
        preflop_raise_count = 0   # how many raises preflop
        saw_flop_seats = set()
        prev_stage = 'Preflop'
        first_to_act_flop = {}    # stage → first actor seat

        # Pick random seats to force-fold preflop (simulates shorter tables).
        # Excludes SB/BB so the blinds structure stays intact; only non-blind
        # positions become "inactive" — matches training's random_preflop_folds.
        forced_fold = set()
        if num_force_fold > 0:
            sb_seat = (button + 1) % ENGINE_PLAYERS
            bb_seat = (button + 2) % ENGINE_PLAYERS
            candidates = [s for s in range(ENGINE_PLAYERS)
                          if s != sb_seat and s != bb_seat]
            k = min(num_force_fold, len(candidates))
            forced_fold = set(int(s) for s in np.random.choice(
                candidates, size=k, replace=False))

        # Mark hands dealt — skip forced-fold seats (they don't effectively play)
        for i in range(ENGINE_PLAYERS):
            if i in forced_fold:
                continue
            pos_label = _POS_LABELS_6.get(
                (i - button) % ENGINE_PLAYERS, f'P{i}')
            stats[i].hands += 1
            stats[i].hands_by_pos[pos_label] += 1

        while not eng.is_done():
            agent = eng.current_agent
            if agent is None:
                break
            legal = eng.get_legal_actions()
            if not legal:
                break

            seat = eng.agents.index(agent)
            stage = eng.get_stage()

            # Forced fold preflop for inactive-table seats (before any obs/stats work)
            if stage == 'Preflop' and seat in forced_fold and FOLD in legal:
                hand_actions.append((seat, 'Preflop', FOLD))
                eng.step(FOLD)
                continue

            obs = eng.get_observation()
            cards = eng.get_player_cards(agent)
            rel_pos = (seat - button) % ENGINE_PLAYERS
            pos_label = _POS_LABELS_6.get(rel_pos, f'P{rel_pos}')

            # Detect street change
            if stage != prev_stage:
                # Players still in at flop
                if stage == 'Flop':
                    for i in range(ENGINE_PLAYERS):
                        a = eng.agents[i]
                        try:
                            if eng.get_player_chips(a) > 0 or any(
                                    s == i and act != FOLD
                                    for s, st, act in hand_actions):
                                # Check if player hasn't folded
                                folded = any(s == i and act == FOLD
                                             for s, st, act in hand_actions)
                                if not folded:
                                    saw_flop_seats.add(i)
                        except Exception:
                            pass
                prev_stage = stage

            # Track first actor on each postflop street
            if stage != 'Preflop' and stage not in first_to_act_flop:
                first_to_act_flop[stage] = seat

            # Build context for DecisionEngine
            facing_bet = float(obs[IDX_POT_ODDS]) > 0.001 if len(obs) > IDX_POT_ODDS else False
            pot_bb = obs[8] * 600 if len(obs) > 8 else 0

            # Preflop: use full context for ranges
            hole_cards = None
            if stage == 'Preflop' and cards:
                hole_cards = cards

            ctx = SpotContext(
                obs=obs,
                legal_actions=legal,
                facing_bet=facing_bet,
                street=stage.lower() if stage != 'Preflop' else 'preflop',
                hole_cards=hole_cards,
                rel_pos=rel_pos,
                pot_bb=pot_bb,
                # Use engine's 6 so preflop range lookup matches training
                # (training always uses 6-max ranges even with forced folds).
                num_players=ENGINE_PLAYERS,
                effective_stack=eng.get_player_chips(agent) / 2.0,
            )

            # Get action from DecisionEngine (same pipeline as runtime agent)
            # In vs mode: even seats (0,2,4) = engine A, odd seats (1,3,5) = engine B
            active_engine = engine if (engine_b is None or seat % 2 == 0) else engine_b
            action, strategy_list = active_engine.decide(ctx)

            # ── Postflop sizing override (mirrors default_core.decide()) ──
            # The DecisionEngine itself doesn't invoke _postflop_sizing — that
            # call lives in the runtime agent's wrapper. To make self-play stats
            # reflect the same action distribution the runtime agent produces,
            # invoke the override here too. This may promote a raise → ALL_IN.
            if (track_range_analysis
                    and stage in ('Flop', 'Turn', 'River')
                    and action in (RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN)
                    and getattr(active_engine, '_postflop_sizing', None) is not None):
                try:
                    community = []
                    try:
                        community = [str(c) for c in eng.get_community_cards()]
                    except Exception:
                        pass
                    spot_inputs = {
                        'street': stage.lower(),
                        'obs': obs,
                        'hole_cards': cards,
                        'community_cards': community,
                        'rel_pos': rel_pos,
                        'num_active': ENGINE_PLAYERS,
                        'effective_stack_bb': eng.get_player_chips(agent) / 2.0,
                        'pot_bb': max(pot_bb, 0.01),
                        'was_preflop_raiser': (seat == preflop_raiser),
                        # facing_bet_size_pct: rough proxy from pot_odds (only
                        # nonzero when facing a bet; ratio is approximate).
                        'facing_bet_size_pct': (1.0 if facing_bet else None),
                        'facing_check_raise': False,
                        'call_amount_bb': 0.0,
                    }
                    new_action, _ = active_engine._postflop_sizing.adjust(
                        action, spot_inputs)
                    if new_action != action:
                        action = new_action
                        strategy_list = [0.0] * NUM_ACTIONS
                        if 0 <= action < NUM_ACTIONS:
                            strategy_list[action] = 1.0
                except Exception:
                    # Counter increments to error_fallback on its own
                    pass

            # ── Record preflop stats ──
            if stage == 'Preflop':
                is_bb = (rel_pos == 2)
                combo = _cards_to_combo(cards) if cards else '??'

                # Check if there was a prior raise
                pf_raises = [a for s, st, a in hand_actions
                             if st == 'Preflop' and a >= RAISE_HALF]
                has_prior_raise = len(pf_raises) > 0

                # ── Range tracking (only when --range-analysis enabled) ──
                # Bucket the action_history for the spot being decided NOW.
                if combo != '??':
                    if preflop_raise_count == 0:
                        pf_calls = [a for s, st, a in hand_actions
                                    if st == 'Preflop' and a == CALL]
                        # Calls without prior raise = limps. SB posting and
                        # BB posting are blinds, not voluntary actions, so
                        # any CALL recorded here is a voluntary limp.
                        hist = 'limped' if pf_calls else 'open'
                    elif preflop_raise_count == 1:
                        hist = 'vs_raise'
                    elif preflop_raise_count == 2:
                        hist = 'vs_3bet'
                    else:
                        hist = 'vs_4bet+'
                    if action >= RAISE_HALF:
                        action_label = 'raise'
                    elif action == CALL:
                        action_label = 'call'
                    else:
                        action_label = 'fold'
                    stats[seat].range_grid[(pos_label, hist)][combo][action_label] += 1

                if action >= RAISE_HALF:
                    stats[seat].vpip += 1
                    stats[seat].pfr += 1
                    stats[seat].vpip_by_pos[pos_label] += 1
                    stats[seat].pfr_by_pos[pos_label] += 1

                    if not has_prior_raise:
                        stats[seat].open_raise += 1
                    else:
                        # This is a 3bet (or 4bet etc)
                        if preflop_raise_count == 1:
                            stats[seat].three_bet += 1

                    preflop_raiser = seat
                    preflop_raise_count += 1

                elif action == CALL:
                    if not is_bb or has_prior_raise:
                        stats[seat].vpip += 1
                        stats[seat].vpip_by_pos[pos_label] += 1
                    if not has_prior_raise:
                        stats[seat].limp += 1

                elif action == FOLD:
                    if has_prior_raise and preflop_raise_count == 1:
                        # Fold to a raise — check if we had 3bet opportunity
                        pass  # tracked below

                # Open raise opportunity (no prior raise, not BB)
                if not has_prior_raise and not is_bb:
                    stats[seat].open_raise_opps += 1

                # 3bet opportunity (exactly one prior raise)
                if preflop_raise_count == 1 and has_prior_raise:
                    stats[seat].three_bet_opps += 1
                    # fold-to-3bet: did the original raiser fold?
                    # (tracked after action)

                # Fold to 3bet
                if preflop_raise_count >= 2:
                    # The previous raiser faces a 3bet
                    if seat == preflop_raiser and action == FOLD:
                        stats[seat].fold_to_3bet += 1

            # ── Record postflop stats ──
            elif stage in ('Flop', 'Turn', 'River'):
                street_key = stage.lower()

                if action == FOLD:
                    stats[seat].postflop_folds += 1
                    stats[seat].actions_by_street[street_key]['fold'] += 1
                elif action >= RAISE_HALF:
                    stats[seat].postflop_bets += 1
                    stats[seat].actions_by_street[street_key]['raise'] += 1
                    # Granular raise tracking
                    if action == RAISE_HALF:
                        stats[seat].actions_by_street[street_key]['raise_half'] += 1
                    elif action == RAISE_POT:
                        stats[seat].actions_by_street[street_key]['raise_pot'] += 1
                    elif action == ALL_IN:
                        stats[seat].actions_by_street[street_key]['all_in'] += 1
                elif action == CALL:
                    # Action 1 = check when not facing bet, call when facing bet
                    if not facing_bet:
                        stats[seat].postflop_checks += 1
                        stats[seat].actions_by_street[street_key]['check'] += 1
                    else:
                        stats[seat].postflop_calls += 1
                        stats[seat].actions_by_street[street_key]['call'] += 1

                # CBet tracking (preflop raiser betting on flop)
                if stage == 'Flop' and seat == preflop_raiser:
                    if first_to_act_flop.get('Flop') == seat or not facing_bet:
                        stats[seat].cbet_opps += 1
                        if action >= RAISE_HALF:
                            stats[seat].cbet += 1

                # Fold to CBet (preflop raiser bet, we fold)
                if stage == 'Flop' and preflop_raiser is not None:
                    if seat != preflop_raiser and facing_bet:
                        # Check if this bet came from preflop raiser
                        flop_bets = [(s, a) for s, st, a in hand_actions
                                     if st == 'Flop' and a >= RAISE_HALF]
                        if any(s == preflop_raiser for s, _ in flop_bets):
                            stats[seat].fold_to_cbet_opps += 1
                            if action == FOLD:
                                stats[seat].fold_to_cbet += 1

            hand_actions.append((seat, stage, action))
            eng.step(action)

        # ── Hand finished — record results ──
        rewards = eng.get_rewards()
        for i in range(ENGINE_PLAYERS):
            if i in forced_fold:
                continue  # seat didn't effectively play this hand
            agent_name = eng.agents[i]
            reward = rewards.get(agent_name, 0)
            reward_bb = reward / 2.0  # chips → BB (blinds are 1/2)
            rp = (i - button) % ENGINE_PLAYERS
            pl = _POS_LABELS_6.get(rp, f'P{rp}')

            stats[i].total_won_bb += reward_bb
            stats[i].won_by_pos[pl] += reward_bb

            if i in saw_flop_seats:
                stats[i].saw_flop += 1

            # Went to showdown? (didn't fold)
            folded = any(s == i and a == FOLD for s, st, a in hand_actions)
            # Check if hand reached showdown (last street is River and game finished)
            reached_river = any(st == 'River' for _, st, _ in hand_actions)
            if not folded and reached_river:
                stats[i].went_to_showdown += 1
                if reward > 0:
                    stats[i].won_at_showdown += 1

    elapsed = time.time() - t0
    print(f'  Completed {num_hands} hands in {elapsed:.1f}s '
          f'({num_hands / elapsed:.0f} hands/s)')
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def _pct(num, den):
    return f'{100 * num / den:.1f}%' if den > 0 else '-'


def _pct_f(num, den):
    return 100 * num / den if den > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 13×13 hand-chart renderer for --range-analysis
# ─────────────────────────────────────────────────────────────────────────────

# Standard chart layout: rows/cols are ranks A-down-to-2, diagonal=pairs,
# upper triangle (row<col)=suited, lower triangle (row>col)=offsuit.
_CHART_RANKS = 'AKQJT98765432'


def _combos_for_class(combo: str) -> int:
    """pair=6, suited=4, offsuit=12. Mirrors PreflopLookup._combos_for_class
    but accepts the lowercase-suffix notation used in self_play (AKs, AKo)."""
    if len(combo) == 2:
        return 6
    if combo.endswith('s'):
        return 4
    return 12


def _combo_at(row_idx: int, col_idx: int) -> str:
    """Return the canonical combo string for a 13×13 chart cell.
    Diagonal = pair (e.g., 'AA'), row<col = suited ('AKs'), row>col = offsuit ('AKo').
    """
    if row_idx == col_idx:
        return _CHART_RANKS[row_idx] * 2
    if row_idx < col_idx:
        return _CHART_RANKS[row_idx] + _CHART_RANKS[col_idx] + 's'
    return _CHART_RANKS[col_idx] + _CHART_RANKS[row_idx] + 'o'


def _cell_symbol(actions: Dict[str, int]) -> str:
    """Pick a single-char symbol summarizing the action distribution.
      *  always raise (≥80%)
      ○  mostly raise (50-79%)
      ~  mixed raise/call (raise 30-50%)
      c  mostly call (call ≥70%, raise <30%)
      .  mostly fold (fold ≥70%)
      ·  small sample / mixed/other (fallback)
      ' ' never observed
    """
    total = actions.get('fold', 0) + actions.get('call', 0) + actions.get('raise', 0)
    if total == 0:
        return ' '
    fp = actions.get('fold', 0) / total
    cp = actions.get('call', 0) / total
    rp = actions.get('raise', 0) / total
    if rp >= 0.80:
        return '*'
    if rp >= 0.50:
        return 'o'   # ○ (use ASCII 'o' to keep alignment in any terminal)
    if rp >= 0.30:
        return '~'
    if cp >= 0.70:
        return 'c'
    if fp >= 0.70:
        return '.'
    return ':'  # genuinely mixed


def _render_hand_chart(grid_data: Dict[str, Dict[str, int]],
                       title: str) -> List[str]:
    """Render a 13×13 hand chart for one (position, action_history) spot.

    grid_data: combo (e.g., 'AKs', 'AA', '72o') → {fold/call/raise: count}
    Returns a list of lines (no trailing newlines).
    """
    out: List[str] = []
    out.append('')
    out.append(f'  RANGE: {title}')
    # Header row
    header = '       ' + ' '.join(f' {r} ' for r in _CHART_RANKS)
    out.append(header)
    out.append('     ' + '┌' + '─' * (4 * 13 - 1) + '┐')
    total_combos_played = 0
    total_combos_raised = 0
    total_decisions = 0
    for ri in range(13):
        cells = []
        for ci in range(13):
            combo = _combo_at(ri, ci)
            actions = grid_data.get(combo, {})
            sym = _cell_symbol(actions)
            cells.append(f' {sym} ')
            n = sum(actions.values())
            if n > 0:
                total_decisions += n
                # weight by combos for "% of deck" stats
                if (actions.get('raise', 0) + actions.get('call', 0)) / n >= 0.50:
                    total_combos_played += _combos_for_class(combo)
                if actions.get('raise', 0) / n >= 0.50:
                    total_combos_raised += _combos_for_class(combo)
        row_label = _CHART_RANKS[ri]
        out.append(f'  {row_label}  │' + '│'.join(cells) + '│')
    out.append('     ' + '└' + '─' * (4 * 13 - 1) + '┘')
    out.append(f'  played: {total_combos_played}/1326 combos '
               f'({100*total_combos_played/1326:.1f}%)   '
               f'raised: {total_combos_raised}/1326 combos '
               f'({100*total_combos_raised/1326:.1f}%)   '
               f'n={total_decisions}')
    return out


def _render_chart_legend() -> List[str]:
    return [
        '  Symbol legend: * always-raise (≥80%)  o mostly-raise (50-79%)  '
        '~ mixed (raise 30-49%)',
        '                 c mostly-call (≥70%)   . mostly-fold (≥70%)    '
        ': genuinely mixed   (blank) never observed',
    ]


def _aggregate_range_grids(
        all_stats: Dict[int, 'PlayerStats']
) -> Dict[Tuple[str, str], Dict[str, Dict[str, int]]]:
    """Sum range_grid across all seats (model plays every seat equally)."""
    agg: Dict[Tuple[str, str], Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)))
    for ps in all_stats.values():
        for spot_key, combos in ps.range_grid.items():
            for combo, actions in combos.items():
                for label, n in actions.items():
                    agg[spot_key][combo][label] += n
    return agg


def generate_report(all_stats: Dict[int, PlayerStats], num_players: int,
                    num_hands: int,
                    engine: Optional['DecisionEngine'] = None,
                    range_analysis: bool = False,
                    history_filter: Optional[List[str]] = None) -> str:
    """Generate text report from aggregated stats."""
    lines = []
    w = lines.append

    # Aggregate across all seats (model plays every seat equally)
    agg = PlayerStats()
    for s in all_stats.values():
        agg.hands += s.hands
        agg.vpip += s.vpip
        agg.pfr += s.pfr
        agg.three_bet_opps += s.three_bet_opps
        agg.three_bet += s.three_bet
        agg.fold_to_3bet_opps += s.fold_to_3bet_opps
        agg.fold_to_3bet += s.fold_to_3bet
        agg.open_raise_opps += s.open_raise_opps
        agg.open_raise += s.open_raise
        agg.limp += s.limp
        agg.cbet_opps += s.cbet_opps
        agg.cbet += s.cbet
        agg.fold_to_cbet_opps += s.fold_to_cbet_opps
        agg.fold_to_cbet += s.fold_to_cbet
        agg.postflop_bets += s.postflop_bets
        agg.postflop_calls += s.postflop_calls
        agg.postflop_folds += s.postflop_folds
        agg.postflop_checks += s.postflop_checks
        agg.saw_flop += s.saw_flop
        agg.went_to_showdown += s.went_to_showdown
        agg.won_at_showdown += s.won_at_showdown
        agg.total_won_bb += s.total_won_bb
        for pos, cnt in s.vpip_by_pos.items():
            agg.vpip_by_pos[pos] += cnt
        for pos, cnt in s.hands_by_pos.items():
            agg.hands_by_pos[pos] += cnt
        for pos, cnt in s.pfr_by_pos.items():
            agg.pfr_by_pos[pos] += cnt
        for pos, val in s.won_by_pos.items():
            agg.won_by_pos[pos] += val
        for street, acts in s.actions_by_street.items():
            for act_type, cnt in acts.items():
                agg.actions_by_street[street][act_type] += cnt

    w('SELF-PLAY ANALYSIS REPORT')
    w('=' * 70)
    w(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    w(f'Hands: {num_hands:,}  |  Players: {num_players}  |  '
      f'Total decisions: {agg.hands:,}')
    w('')

    # ── Overall stats ──
    w('─' * 70)
    w('OVERALL STATISTICS')
    w('─' * 70)
    w(f'  VPIP:              {_pct(agg.vpip, agg.hands):>8}   '
      f'({agg.vpip}/{agg.hands})')
    w(f'  PFR:               {_pct(agg.pfr, agg.hands):>8}   '
      f'({agg.pfr}/{agg.hands})')
    w(f'  3Bet:              {_pct(agg.three_bet, agg.three_bet_opps):>8}   '
      f'({agg.three_bet}/{agg.three_bet_opps})')
    w(f'  Fold to 3Bet:      {_pct(agg.fold_to_3bet, agg.fold_to_3bet_opps):>8}   '
      f'({agg.fold_to_3bet}/{agg.fold_to_3bet_opps})')
    w(f'  RFI (open raise):  {_pct(agg.open_raise, agg.open_raise_opps):>8}   '
      f'({agg.open_raise}/{agg.open_raise_opps})')
    w(f'  Limp:              {_pct(agg.limp, agg.hands):>8}   '
      f'({agg.limp}/{agg.hands})')
    w(f'  CBet:              {_pct(agg.cbet, agg.cbet_opps):>8}   '
      f'({agg.cbet}/{agg.cbet_opps})')
    w(f'  Fold to CBet:      {_pct(agg.fold_to_cbet, agg.fold_to_cbet_opps):>8}   '
      f'({agg.fold_to_cbet}/{agg.fold_to_cbet_opps})')

    # Aggression factor = (bets+raises) / calls
    af = (agg.postflop_bets / agg.postflop_calls
          if agg.postflop_calls > 0 else 0)
    total_postflop = (agg.postflop_bets + agg.postflop_calls +
                      agg.postflop_folds + agg.postflop_checks)
    agg_pct = _pct(agg.postflop_bets, total_postflop) if total_postflop > 0 else '-'
    w(f'  AF (agg factor):   {af:>8.2f}   '
      f'(bets={agg.postflop_bets}, calls={agg.postflop_calls})')
    w(f'  Aggression%:       {agg_pct:>8}   '
      f'(of {total_postflop} postflop actions)')
    w(f'  Saw flop:          {_pct(agg.saw_flop, agg.hands):>8}')
    w(f'  WTSD (went to SD): {_pct(agg.went_to_showdown, agg.saw_flop):>8}')
    w(f'  W$SD (won at SD):  {_pct(agg.won_at_showdown, agg.went_to_showdown):>8}')
    w('')

    # ── GTO reference ranges ──
    gto_ref = {
        'VPIP': (20, 25, '20-25%'),
        'PFR': (16, 21, '16-21%'),
        'RFI': (40, 55, '40-55%'),
        '3Bet': (7, 12, '7-12%'),
        'CBet': (55, 70, '55-70%'),
        'AF': (2.0, 3.5, '2.0-3.5'),
        'WTSD': (25, 32, '25-32%'),
    }
    w('─' * 70)
    w('GTO REFERENCE COMPARISON')
    w('─' * 70)
    w(f'  {"Stat":<18} {"Model":>8}  {"GTO Range":>12}  {"Status"}')
    w(f'  {"-"*18} {"-"*8}  {"-"*12}  {"-"*10}')

    model_vals = {
        'VPIP': _pct_f(agg.vpip, agg.hands),
        'PFR': _pct_f(agg.pfr, agg.hands),
        'RFI': _pct_f(agg.open_raise, agg.open_raise_opps),
        '3Bet': _pct_f(agg.three_bet, agg.three_bet_opps),
        'CBet': _pct_f(agg.cbet, agg.cbet_opps),
        'AF': af,
        'WTSD': _pct_f(agg.went_to_showdown, agg.saw_flop),
    }
    for stat, (lo, hi, label) in gto_ref.items():
        val = model_vals[stat]
        if stat == 'AF':
            val_str = f'{val:.2f}'
        else:
            val_str = f'{val:.1f}%'
        if val < lo:
            status = 'TOO LOW'
        elif val > hi:
            status = 'TOO HIGH'
        else:
            status = 'OK'
        w(f'  {stat:<18} {val_str:>8}  {label:>12}  {status}')
    w('')

    # ── By position ──
    w('─' * 70)
    w('STATS BY POSITION')
    w('─' * 70)
    pos_order = ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']
    w(f'  {"Pos":<5} {"Hands":>7} {"VPIP":>8} {"PFR":>8} {"BB/100":>8}')
    w(f'  {"-"*5} {"-"*7} {"-"*8} {"-"*8} {"-"*8}')
    for pos in pos_order:
        h = agg.hands_by_pos.get(pos, 0)
        v = agg.vpip_by_pos.get(pos, 0)
        p = agg.pfr_by_pos.get(pos, 0)
        won = agg.won_by_pos.get(pos, 0.0)
        bb100 = (won / h * 100) if h > 0 else 0
        w(f'  {pos:<5} {h:>7} {_pct(v, h):>8} {_pct(p, h):>8} '
          f'{bb100:>+8.1f}')
    w('')

    # ── Postflop action distribution by street ──
    w('─' * 70)
    w('POSTFLOP ACTION DISTRIBUTION BY STREET')
    w('─' * 70)
    w(f'  {"Street":<8} {"Check":>8} {"Call":>8} {"Raise":>8} {"Fold":>8} '
      f'{"Total":>8}  {"AF":>5}')
    w(f'  {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*8}  {"-"*5}')
    for st in ['flop', 'turn', 'river']:
        acts = agg.actions_by_street.get(st, {})
        ck = acts.get('check', 0)
        ca = acts.get('call', 0)
        ra = acts.get('raise', 0)
        fo = acts.get('fold', 0)
        tot = ck + ca + ra + fo
        st_af = ra / ca if ca > 0 else 0
        w(f'  {st:<8} {_pct(ck, tot):>8} {_pct(ca, tot):>8} '
          f'{_pct(ra, tot):>8} {_pct(fo, tot):>8} '
          f'{tot:>8}  {st_af:>5.2f}')
    w('')

    # ── Granular action breakdown (Raise½ / RaisePot / AllIn) ──
    w('─' * 70)
    w('POSTFLOP RAISE BREAKDOWN BY STREET')
    w('─' * 70)
    w(f'  {"Street":<8} {"Fold":>8} {"Check":>8} {"Call":>8} '
      f'{"Raise½":>8} {"RaisePot":>9} {"AllIn":>8}')
    w(f'  {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*9} {"-"*8}')
    for st in ['flop', 'turn', 'river']:
        acts = agg.actions_by_street.get(st, {})
        fo = acts.get('fold', 0)
        ck = acts.get('check', 0)
        ca = acts.get('call', 0)
        rh = acts.get('raise_half', 0)
        rp = acts.get('raise_pot', 0)
        ai = acts.get('all_in', 0)
        tot = fo + ck + ca + rh + rp + ai
        w(f'  {st:<8} {_pct(fo, tot):>8} {_pct(ck, tot):>8} {_pct(ca, tot):>8} '
          f'{_pct(rh, tot):>8} {_pct(rp, tot):>9} {_pct(ai, tot):>8}')

    # Overall action totals
    all_acts = defaultdict(int)
    for st in ['flop', 'turn', 'river']:
        for k, v in agg.actions_by_street.get(st, {}).items():
            all_acts[k] += v
    tot_all = sum(all_acts.values())
    if tot_all > 0:
        w(f'  {"TOTAL":<8} {_pct(all_acts["fold"], tot_all):>8} '
          f'{_pct(all_acts["check"], tot_all):>8} '
          f'{_pct(all_acts["call"], tot_all):>8} '
          f'{_pct(all_acts["raise_half"], tot_all):>8} '
          f'{_pct(all_acts["raise_pot"], tot_all):>9} '
          f'{_pct(all_acts["all_in"], tot_all):>8}')
    w('')

    # ── Preflop ranges summary ──
    w('─' * 70)
    w('PREFLOP RFI BY POSITION (raise first in)')
    w('─' * 70)
    # Computed per-seat, averaged across seats in same position
    pos_rfi = defaultdict(lambda: [0, 0])
    for s in all_stats.values():
        for pos in pos_order:
            h = s.hands_by_pos.get(pos, 0)
            p = s.pfr_by_pos.get(pos, 0)
            pos_rfi[pos][0] += p
            pos_rfi[pos][1] += h
    w(f'  {"Pos":<5} {"RFI%":>8}  {"GTO~":>8}')
    w(f'  {"-"*5} {"-"*8}  {"-"*8}')
    gto_rfi = {'UTG': '15-20%', 'MP': '18-25%', 'CO': '27-35%',
               'BTN': '40-55%', 'SB': '35-45%', 'BB': '-'}
    for pos in pos_order:
        r, h = pos_rfi[pos]
        w(f'  {pos:<5} {_pct(r, h):>8}  {gto_rfi.get(pos, "-"):>8}')
    w('')

    # ─────────────────────────────────────────────────────────────────────
    # Range analysis sections (only when --range-analysis was set)
    # ─────────────────────────────────────────────────────────────────────
    if range_analysis:
        w('')
        w('═' * 70)
        w('RANGE ANALYSIS — preflop hand-chart grids')
        w('═' * 70)
        for line in _render_chart_legend():
            w(line)

        agg_grid = _aggregate_range_grids(all_stats)
        # Print charts in a sensible order: per-position, per-history
        position_order = ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']
        history_order = ['open', 'limped', 'vs_raise', 'vs_3bet', 'vs_4bet+']
        if history_filter is not None:
            history_order = [h for h in history_order if h in history_filter]
        MIN_DECISIONS = 50  # skip charts with too few samples to be meaningful
        printed = 0
        for hist in history_order:
            for pos in position_order:
                grid_data = agg_grid.get((pos, hist), {})
                total_n = sum(sum(a.values()) for a in grid_data.values())
                if total_n < MIN_DECISIONS:
                    continue
                title = f'{pos} {hist}'
                for line in _render_hand_chart(grid_data, title):
                    w(line)
                printed += 1
        if printed == 0:
            w('  (no spots reached the 50-decision minimum — try more hands)')
        w('')

        # Preflop lookup level distribution
        if engine is not None and engine._preflop_ranges:
            w('─' * 70)
            w('Preflop lookup level distribution')
            w('─' * 70)
            # Sum counters across all per-num_players lookup instances
            level_totals: Dict[str, int] = defaultdict(int)
            seen_ids: set = set()
            for pf in engine._preflop_ranges.values():
                if id(pf) in seen_ids:
                    continue
                seen_ids.add(id(pf))
                if hasattr(pf, 'counters'):
                    for k, v in pf.counters.items():
                        level_totals[k] += v
            total = sum(level_totals.values())
            if total > 0:
                for label in ['L0_exact', 'L1_no_nta', 'L2_no_stack',
                              'L3_fold_default', 'error_fallback']:
                    cnt = level_totals.get(label, 0)
                    w(f'  {label:<22s} {cnt:>10,d} ({_pct(cnt, total):>6})')
            else:
                w('  (no preflop lookup decisions recorded)')
            w('')

        # Postflop sizing override stats
        if engine is not None and getattr(engine, '_postflop_sizing', None) is not None:
            w('─' * 70)
            w('Postflop sizing override usage')
            w('─' * 70)
            c = engine._postflop_sizing.counters
            queries = c.get('queries', 0)
            sized = c.get('sizing_override', 0)
            promoted = c.get('all_in_promotions', 0)
            no_data = c.get('no_data', 0)
            non_pf = c.get('non_postflop', 0)
            non_raise = c.get('non_raise', 0)
            errors = c.get('error_fallback', 0)
            # The lookup is ONLY consulted when the sampled action is a raise
            # on a postflop street, so 'queries' is the universe for the
            # override-rate columns.
            w(f'  Postflop raise queries:        {queries:>10,d}')
            w(f'    All-in promotions:           {promoted:>10,d} '
              f'({_pct(promoted, queries):>6})')
            w(f'    Sizing overrides (typed):    {sized:>10,d} '
              f'({_pct(sized, queries):>6})')
            w(f'    No-data fallthroughs:        {no_data:>10,d} '
              f'({_pct(no_data, queries):>6})')
            w(f'  Skipped (non-postflop):        {non_pf:>10,d}')
            w(f'  Skipped (non-raise):           {non_raise:>10,d}')
            w(f'  Errors:                        {errors:>10,d}')
            override_rate = (promoted + sized) / queries if queries > 0 else 0.0
            w(f'  → Effective override rate:     {override_rate*100:.1f}% '
              f'of postflop raise decisions used cohort sizing')
            w('')

    w('=' * 70)
    w('END OF REPORT')
    return '\n'.join(lines)


def generate_json(all_stats: Dict[int, PlayerStats], num_players: int,
                  num_hands: int) -> dict:
    """Generate JSON-serializable results dict."""
    agg = PlayerStats()
    for s in all_stats.values():
        agg.hands += s.hands
        agg.vpip += s.vpip
        agg.pfr += s.pfr
        agg.three_bet_opps += s.three_bet_opps
        agg.three_bet += s.three_bet
        agg.open_raise_opps += s.open_raise_opps
        agg.open_raise += s.open_raise
        agg.limp += s.limp
        agg.cbet_opps += s.cbet_opps
        agg.cbet += s.cbet
        agg.fold_to_cbet_opps += s.fold_to_cbet_opps
        agg.fold_to_cbet += s.fold_to_cbet
        agg.postflop_bets += s.postflop_bets
        agg.postflop_calls += s.postflop_calls
        agg.postflop_folds += s.postflop_folds
        agg.postflop_checks += s.postflop_checks
        agg.saw_flop += s.saw_flop
        agg.went_to_showdown += s.went_to_showdown
        agg.won_at_showdown += s.won_at_showdown
        for pos, cnt in s.vpip_by_pos.items():
            agg.vpip_by_pos[pos] += cnt
        for pos, cnt in s.hands_by_pos.items():
            agg.hands_by_pos[pos] += cnt
        for pos, cnt in s.pfr_by_pos.items():
            agg.pfr_by_pos[pos] += cnt
        for pos, val in s.won_by_pos.items():
            agg.won_by_pos[pos] += val

    af = agg.postflop_bets / agg.postflop_calls if agg.postflop_calls > 0 else 0
    total_postflop_actions = (agg.postflop_folds + agg.postflop_checks +
                              agg.postflop_calls + agg.postflop_bets)
    # AFq = (bets+raises) / (bets+raises+calls+folds) — excludes checks, more stable than AF
    afq_denom = agg.postflop_bets + agg.postflop_calls + agg.postflop_folds
    afq = (agg.postflop_bets / afq_denom * 100) if afq_denom > 0 else 0

    # Per-street stats from aggregated action counts
    street_stats = {}
    for st in ['flop', 'turn', 'river']:
        sa = defaultdict(int)
        for s in all_stats.values():
            for k, v in s.actions_by_street[st].items():
                sa[k] += v
        st_total = sum(sa.values())
        st_bets = sa['raise'] + sa['raise_half'] + sa['raise_pot'] + sa['all_in']
        st_calls = sa['call']
        st_af = st_bets / st_calls if st_calls > 0 else 0
        street_stats[st] = {
            'fold_pct': _pct_f(sa['fold'], st_total),
            'check_pct': _pct_f(sa['check'], st_total),
            'call_pct': _pct_f(sa['call'], st_total),
            'bet_raise_pct': _pct_f(st_bets, st_total),
            'af': round(st_af, 2),
            'total_actions': st_total,
        }

    # GTO range check
    vpip_val = _pct_f(agg.vpip, agg.hands)
    pfr_val = _pct_f(agg.pfr, agg.hands)
    cbet_val = _pct_f(agg.cbet, agg.cbet_opps)
    wtsd_val = _pct_f(agg.went_to_showdown, agg.saw_flop)

    gto_flags = []
    if vpip_val < 20 or vpip_val > 25: gto_flags.append('VPIP')
    if pfr_val < 16 or pfr_val > 21: gto_flags.append('PFR')
    if cbet_val < 55 or cbet_val > 70: gto_flags.append('CBET')
    if af < 2.0 or af > 3.5: gto_flags.append('AF')
    if wtsd_val < 25 or wtsd_val > 35: gto_flags.append('WTSD')

    return {
        'num_hands': num_hands,
        'num_players': num_players,
        'vpip': vpip_val,
        'pfr': pfr_val,
        'three_bet': _pct_f(agg.three_bet, agg.three_bet_opps),
        'rfi': _pct_f(agg.open_raise, agg.open_raise_opps),
        'limp_pct': _pct_f(agg.limp, agg.hands),
        'cbet': cbet_val,
        'fold_to_cbet': _pct_f(agg.fold_to_cbet, agg.fold_to_cbet_opps),
        'af': round(af, 2),
        'afq': round(afq, 1),
        'saw_flop': _pct_f(agg.saw_flop, agg.hands),
        'wtsd': wtsd_val,
        'wsd': _pct_f(agg.won_at_showdown, agg.went_to_showdown),
        'postflop_actions': {
            'fold_pct': _pct_f(agg.postflop_folds, total_postflop_actions),
            'check_pct': _pct_f(agg.postflop_checks, total_postflop_actions),
            'call_pct': _pct_f(agg.postflop_calls, total_postflop_actions),
            'bet_raise_pct': _pct_f(agg.postflop_bets, total_postflop_actions),
            'total': total_postflop_actions,
        },
        'by_street': street_stats,
        'by_position': {
            pos: {
                'hands': agg.hands_by_pos.get(pos, 0),
                'vpip': _pct_f(agg.vpip_by_pos.get(pos, 0),
                               agg.hands_by_pos.get(pos, 1)),
                'pfr': _pct_f(agg.pfr_by_pos.get(pos, 0),
                              agg.hands_by_pos.get(pos, 1)),
                'bb_per_100': round(
                    (agg.won_by_pos.get(pos, 0) /
                     max(agg.hands_by_pos.get(pos, 1), 1)) * 100, 1),
            }
            for pos in ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']
        },
        'gto_flags': gto_flags if gto_flags else ['OK'],
        'raw_counts': {
            'postflop_bets': agg.postflop_bets,
            'postflop_calls': agg.postflop_calls,
            'postflop_checks': agg.postflop_checks,
            'postflop_folds': agg.postflop_folds,
            'saw_flop': agg.saw_flop,
            'went_to_showdown': agg.went_to_showdown,
            'won_at_showdown': agg.won_at_showdown,
            'cbet_opps': agg.cbet_opps,
            'cbet_made': agg.cbet,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _generate_vs_report(stats: Dict[int, PlayerStats], num_players: int,
                        num_hands: int, label_a: str, label_b: str) -> str:
    """Generate head-to-head report: A (seats 0,2,4) vs B (seats 1,3,5)."""
    lines = []
    w = lines.append

    # Engine is always 6-max; `num_players` here is effective table size.
    # Aggregate across all engine seats (forced-fold seats contribute zero hands).
    engine_seats = sorted(stats.keys())
    seats_a = [i for i in engine_seats if i % 2 == 0]
    seats_b = [i for i in engine_seats if i % 2 == 1]

    def _agg(seat_list):
        a = PlayerStats()
        for i in seat_list:
            s = stats[i]
            a.hands += s.hands
            a.total_won_bb += s.total_won_bb
            a.vpip += s.vpip
            a.pfr += s.pfr
            a.postflop_bets += s.postflop_bets
            a.postflop_calls += s.postflop_calls
            a.postflop_folds += s.postflop_folds
            a.postflop_checks += s.postflop_checks
            a.saw_flop += s.saw_flop
            a.went_to_showdown += s.went_to_showdown
            a.won_at_showdown += s.won_at_showdown
            a.cbet_opps += s.cbet_opps
            a.cbet += s.cbet
        return a

    a = _agg(seats_a)
    b = _agg(seats_b)

    bb100_a = a.total_won_bb / a.hands * 100 if a.hands > 0 else 0
    bb100_b = b.total_won_bb / b.hands * 100 if b.hands > 0 else 0
    af_a = a.postflop_bets / a.postflop_calls if a.postflop_calls > 0 else 0
    af_b = b.postflop_bets / b.postflop_calls if b.postflop_calls > 0 else 0

    w('HEAD-TO-HEAD RESULTS')
    w('=' * 70)
    w(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    w(f'Hands: {num_hands:,}  |  Players: {num_players}')
    w(f'  A (seats 0,2,4): {label_a}')
    w(f'  B (seats 1,3,5): {label_b}')
    w('')
    w('─' * 70)
    w(f'  {"Metric":<20} {"Model A":>12} {"Model B":>12} {"Diff":>10}')
    w(f'  {"-"*20} {"-"*12} {"-"*12} {"-"*10}')
    w(f'  {"BB/100":<20} {bb100_a:>+12.1f} {bb100_b:>+12.1f} '
      f'{bb100_a - bb100_b:>+10.1f}')
    w(f'  {"Total BB won":<20} {a.total_won_bb:>+12.0f} {b.total_won_bb:>+12.0f} '
      f'{a.total_won_bb - b.total_won_bb:>+10.0f}')
    w(f'  {"VPIP":<20} {_pct(a.vpip, a.hands):>12} {_pct(b.vpip, b.hands):>12}')
    w(f'  {"PFR":<20} {_pct(a.pfr, a.hands):>12} {_pct(b.pfr, b.hands):>12}')
    w(f'  {"AF":<20} {af_a:>12.2f} {af_b:>12.2f}')
    w(f'  {"CBet":<20} {_pct(a.cbet, a.cbet_opps):>12} {_pct(b.cbet, b.cbet_opps):>12}')
    w(f'  {"WTSD":<20} {_pct(a.went_to_showdown, a.saw_flop):>12} '
      f'{_pct(b.went_to_showdown, b.saw_flop):>12}')
    w(f'  {"W$SD":<20} {_pct(a.won_at_showdown, a.went_to_showdown):>12} '
      f'{_pct(b.won_at_showdown, b.went_to_showdown):>12}')
    w('')

    if bb100_a > bb100_b + 5:
        w(f'  >>> Model A wins by {bb100_a - bb100_b:+.1f} BB/100')
    elif bb100_b > bb100_a + 5:
        w(f'  >>> Model B wins by {bb100_b - bb100_a:+.1f} BB/100')
    else:
        w(f'  >>> Too close to call (diff = {bb100_a - bb100_b:+.1f} BB/100)')
    w('')

    # Per-seat breakdown
    w('─' * 70)
    w('PER-SEAT BREAKDOWN')
    w('─' * 70)
    w(f'  {"Seat":>4} {"Model":>7} {"Hands":>7} {"BB/100":>8} {"Won BB":>10}')
    w(f'  {"-"*4} {"-"*7} {"-"*7} {"-"*8} {"-"*10}')
    for i in engine_seats:
        s = stats[i]
        if s.hands == 0:
            continue  # seat was forced-fold every hand (shouldn't happen)
        label = 'A' if i % 2 == 0 else 'B'
        sbb = s.total_won_bb / s.hands * 100 if s.hands > 0 else 0
        w(f'  {i:>4} {label:>7} {s.hands:>7} {sbb:>+8.1f} {s.total_won_bb:>+10.0f}')

    w('')
    w('=' * 70)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Self-play analysis: model vs itself or model A vs model B')
    parser.add_argument('--hands', type=int, default=10000,
                        help='Number of hands to play (default: 10000)')
    parser.add_argument('--players', type=int, default=6,
                        help='Number of players (default: 6)')
    parser.add_argument('--model', type=str, default='',
                        help='Single Deep CFR checkpoint (used for all spots)')
    parser.add_argument('--model-facing-bet', type=str,
                        default='models/cql_6p_facing_bet_v9.pt')
    parser.add_argument('--model-checked-to', type=str,
                        default='models/cql_6p_checked_to_v5.pt')
    parser.add_argument('--model-facing-bet-3bet', type=str,
                        default='models/cql_6p_facing_bet_3bet_v2.pt')
    # VS mode: second model (only facing_bet differs)
    parser.add_argument('--vs-facing-bet', type=str, default='',
                        help='Model B facing_bet (vs mode: A vs B)')
    parser.add_argument('--vs-checked-to', type=str, default='',
                        help='Model B checked_to (default: same as A)')
    parser.add_argument('--vs-facing-bet-3bet', type=str, default='',
                        help='Model B 3bet (default: same as A)')
    parser.add_argument('--no-raise-randomize', action='store_true',
                        help='Disable 80/20 raise-size randomization (always pick argmax raise).')
    parser.add_argument('--no-preflop-ranges', action='store_true',
                        help='Disable GTO preflop ranges (model plays preflop too)')
    parser.add_argument('--archetype', type=str, default='',
                        choices=['', 'calling_station', 'nit', 'maniac', 'weak_tight', 'nl5_population'],
                        help='Use archetype as opponent (engine B). '
                             'Wraps opponent model with a strategy skew.')
    parser.add_argument('--opponent-model', type=str, default='',
                        help='Separate model for opponent (engine B). '
                             'Use with --archetype to skew a different model.')
    parser.add_argument('--range-analysis', action='store_true',
                        help='Track per-(position, action_history) ranges and '
                             'append 13×13 hand-chart grids + preflop/postflop '
                             'lookup counter sections to the report.')
    parser.add_argument('--history-filter', nargs='+', default=None,
                        choices=['open', 'limped', 'vs_raise',
                                  'vs_3bet', 'vs_4bet+'],
                        help='Restrict --range-analysis grid output to these '
                             'action_history buckets (default: all). Useful '
                             'for inspecting just RFI vs 3bet defense vs etc.')
    # ── Preflop lookup knobs (override DecisionConfig defaults) ──
    parser.add_argument('--preflop-min-play-rate', type=float, default=None,
                        help='Override DecisionConfig.preflop_min_play_rate '
                             '(default 0.25). Higher values fold marginal hands '
                             'where the cohort plays them less often than AA.')
    parser.add_argument('--preflop-min-samples-exact', type=int, default=None,
                        help='Override min_samples for L0 exact-key lookup '
                             '(default 5). Higher = more spots fall to L1/L2.')
    parser.add_argument('--preflop-min-samples-no-nta', type=int, default=None,
                        help='Override min_samples for L1 (drop num_to_act) '
                             '(default 3).')
    parser.add_argument('--preflop-min-samples-no-stack', type=int, default=None,
                        help='Override min_samples for L2 (drop stack_bucket) '
                             '(default 5). Higher = sparser L2 → more static.')
    args = parser.parse_args()

    vs_mode = bool(args.vs_facing_bet) or bool(args.archetype) or bool(args.opponent_model)

    print('=' * 70)
    print('SELF-PLAY ANALYSIS' + (' — VS MODE' if vs_mode else ''))
    print('=' * 70)

    # Setup DecisionEngine A
    _dc_kwargs = dict(
        use_preflop_ranges=not args.no_preflop_ranges,
        raise_randomize=not args.no_raise_randomize,
    )
    if args.preflop_min_play_rate is not None:
        _dc_kwargs['preflop_min_play_rate'] = args.preflop_min_play_rate
    if args.preflop_min_samples_exact is not None:
        _dc_kwargs['preflop_min_samples_exact'] = args.preflop_min_samples_exact
    if args.preflop_min_samples_no_nta is not None:
        _dc_kwargs['preflop_min_samples_no_nta'] = args.preflop_min_samples_no_nta
    if args.preflop_min_samples_no_stack is not None:
        _dc_kwargs['preflop_min_samples_no_stack'] = args.preflop_min_samples_no_stack
    dc = DecisionConfig(**_dc_kwargs)
    engine = DecisionEngine(dc)
    if args.model:
        # Single Deep CFR checkpoint — use for all spots
        engine.load_models(
            facing_bet=args.model,
            checked_to=args.model,
        )
    else:
        engine.load_models(
            facing_bet=args.model_facing_bet,
            checked_to=args.model_checked_to,
            facing_bet_3bet=args.model_facing_bet_3bet
                if Path(args.model_facing_bet_3bet).exists() else '',
        )
    engine.load_preflop_ranges()
    print(f'  Engine A: {engine.info()}')

    # Setup DecisionEngine B (vs mode or archetype mode)
    engine_b = None
    if args.archetype or args.opponent_model:
        # Load separate opponent model if specified, otherwise use engine A
        if args.opponent_model:
            opp_engine = DecisionEngine(dc)
            opp_engine.load_models(
                facing_bet=args.opponent_model,
                checked_to=args.opponent_model,
            )
            opp_engine.load_preflop_ranges()
            opp_label = Path(args.opponent_model).stem
        else:
            opp_engine = engine
            opp_label = 'engine A'

        if args.archetype:
            skew_fn = get_archetype_skew(args.archetype)
            engine_b = ArchetypeEngine(opp_engine, skew_fn)
            print(f'  Engine B: archetype={args.archetype} (skew on {opp_label})')
        else:
            engine_b = opp_engine
            print(f'  Engine B: {opp_label}')
    elif vs_mode:
        engine_b = DecisionEngine(dc)
        engine_b.load_models(
            facing_bet=args.vs_facing_bet,
            checked_to=args.vs_checked_to or args.model_checked_to,
            facing_bet_3bet=(args.vs_facing_bet_3bet or args.model_facing_bet_3bet)
                if Path(args.vs_facing_bet_3bet or args.model_facing_bet_3bet).exists() else '',
        )
        engine_b.load_preflop_ranges()
        print(f'  Engine B: {engine_b.info()}')

    print(f'  Raise randomization: '
          f'{"ON" if not args.no_raise_randomize else "OFF"}')
    print(f'  Hands to play: {args.hands:,}')
    print('=' * 70)

    # Reset lookup counters so the post-run report reflects only this run.
    if args.range_analysis:
        seen_ids = set()
        for pf in (engine._preflop_ranges or {}).values():
            if id(pf) in seen_ids:
                continue
            seen_ids.add(id(pf))
            if hasattr(pf, 'reset_counters'):
                pf.reset_counters()
        if getattr(engine, '_postflop_sizing', None) is not None:
            engine._postflop_sizing.reset_counters()

    # Run self-play
    print(f'\nRunning {args.hands:,} hands of {args.players}-max '
          f'{"A vs B" if vs_mode else "self-play"}...')
    stats = run_self_play(engine, args.hands, args.players,
                          engine_b=engine_b,
                          track_range_analysis=args.range_analysis)

    # Generate report
    if vs_mode:
        label_a = Path(args.model or args.model_facing_bet).stem
        label_b = args.archetype if args.archetype else Path(args.vs_facing_bet).stem
        report = _generate_vs_report(stats, args.players, args.hands, label_a, label_b)
    else:
        report = generate_report(stats, args.players, args.hands,
                                 engine=engine,
                                 range_analysis=args.range_analysis,
                                 history_filter=args.history_filter)
    print(f'\n{report}')

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    txt_path = results_dir / f'self_play_{timestamp}.txt'
    txt_path.write_text(report)

    json_data = generate_json(stats, args.players, args.hands)
    json_path = results_dir / f'self_play_{timestamp}.json'
    json_path.write_text(json.dumps(json_data, indent=2))

    print(f'\nResults saved to:')
    print(f'  {txt_path}')
    print(f'  {json_path}')


if __name__ == '__main__':
    main()
