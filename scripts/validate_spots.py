"""
Poker spot validator — comprehensive model quality assessment.

Tests:
  1. Canonical spots (AA/KK/72o/set/air etc.)
     - Legal-action-aware averaging: fold% correct even when check is available
     - Context split: first-to-act vs facing a raise/bet
  2. Both average-strategy network AND current advantage-based strategy
  3. Standard poker stats: VPIP, PFR, 3-bet%
  4. Position-stratified open-raise frequencies vs GTO targets
  5. EV benchmark: model player vs random opponents (chips won/hand)

Usage:
  python3 scripts/validate_spots.py --model models/final_model_enhanced.pt
  python3 scripts/validate_spots.py --model models/checkpoint_iter_500.pt --samples 2000
  python3 scripts/validate_spots.py --model models/final_model_enhanced.pt --strategy both
  python3 scripts/validate_spots.py --model models/final_model_enhanced.pt --ev-games 500
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import warnings
import logging
import numpy as np
from collections import defaultdict
from pathlib import Path

import torch

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

from src.engine import EnhancedPokerEngine
from src.training import DeepCFR
from src.utils import load_config

# ---------------------------------------------------------------------------
# Action indices (texas_holdem_no_limit_v6)
# ---------------------------------------------------------------------------
FOLD       = 0
CALL       = 1   # also covers check
RAISE_HALF = 2
RAISE_FULL = 3
ALL_IN     = 4
AGGRESSIVE = [RAISE_HALF, RAISE_FULL, ALL_IN]

RANK_MAP = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
            'T':10,'J':11,'Q':12,'K':13,'A':14}


def _rank(ch: str) -> int:
    return RANK_MAP.get(ch, 0)


# ---------------------------------------------------------------------------
# Strategy functions
# ---------------------------------------------------------------------------

def _avg_strategy(deep_cfr, obs, agent_idx, legal):
    """Historical average strategy from the strategy network."""
    return deep_cfr.get_strategy(obs, agent_idx, legal)


def _current_strategy(deep_cfr, obs, agent_idx, legal):
    """Current strategy via regret-matching on the advantage network."""
    advantages = deep_cfr.compute_counterfactual_values(obs, agent_idx, legal)
    pos_adv = np.maximum(advantages, 0.0)
    s = pos_adv.sum()
    if s > 0:
        return pos_adv / s
    mask = np.zeros(deep_cfr.action_size, dtype=np.float32)
    mask[np.array(legal)] = 1.0
    return mask / mask.sum()


# ---------------------------------------------------------------------------
# Game context helpers
# ---------------------------------------------------------------------------

def _game_context(engine, agent):
    """Return position label and betting context for the current decision."""
    try:
        game = engine._get_game()
        if not game:
            return {}
        agent_idx  = engine.agents.index(agent)
        player     = game.players[agent_idx]
        n          = engine.num_players
        rel_pos    = (agent_idx - game.dealer_id) % n

        if n == 2:
            pos_label = 'BTN' if rel_pos == 0 else 'BB'
        elif rel_pos == 0:
            pos_label = 'BTN'
        elif rel_pos == 1:
            pos_label = 'SB'
        elif rel_pos == 2:
            pos_label = 'BB'
        elif rel_pos == n - 1:
            pos_label = 'CO'
        elif rel_pos == 3:
            pos_label = 'UTG'
        else:
            pos_label = 'MP'

        max_raised   = max(game.round.raised) if game.round.raised else 0
        call_amount  = max(0, max_raised - player.in_chips)

        return {
            'rel_pos':               rel_pos,
            'pos_label':             pos_label,
            'call_amount':           call_amount,
            # Preflop: someone raised above BB (> 2 chips)
            'facing_preflop_raise':  call_amount > 2,
            # Postflop: someone bet (any call_amount > 0)
            'facing_postflop_bet':   call_amount > 0,
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Preflop hand classifier
# ---------------------------------------------------------------------------

def classify_preflop(cards):
    if len(cards) < 2:
        return None
    r1, r2    = _rank(cards[0][0]), _rank(cards[1][0])
    s1, s2    = cards[0][1], cards[1][1]
    hi, lo    = max(r1, r2), min(r1, r2)
    is_pair   = r1 == r2
    is_suited = s1 == s2

    if is_pair:
        if hi == 14: return 'AA'
        if hi == 13: return 'KK'
        if hi == 12: return 'QQ'
        if hi >= 10: return 'JJ-TT'
        return None

    if hi == 14 and lo == 13:
        return 'AKs' if is_suited else 'AKo'
    if hi == 14 and lo >= 11:
        return 'AQs-AJs' if is_suited else 'AQo-AJo'
    if hi == 7 and lo == 2 and not is_suited:
        return '72o'
    if hi <= 8 and (hi - lo) >= 3 and not is_suited:
        return 'trash_offsuit'
    return None


# ---------------------------------------------------------------------------
# Post-flop hand classifier helpers
# ---------------------------------------------------------------------------

def _has_straight(hole_ranks, board_ranks):
    all_r = sorted(set(hole_ranks + board_ranks))
    if 14 in all_r:
        all_r = sorted(set([1] + all_r))
    for start in range(1, 11):
        if all(r in all_r for r in range(start, start + 5)):
            return True
    return False


def _has_flush_draw(hole_suits, board_suits):
    """Exactly 4 cards to a flush (not a made 5-card flush)."""
    counts = {}
    for s in hole_suits + board_suits:
        counts[s] = counts.get(s, 0) + 1
    return any(v == 4 for v in counts.values())


def _has_oesd(hole_ranks, board_ranks):
    """4+ connected ranks forming an open-ended or gutshot straight draw."""
    all_r = sorted(set(hole_ranks + board_ranks))
    if 14 in all_r:
        all_r = sorted(set([1] + all_r))
    for start in range(1, 12):
        if len(set(range(start, start + 4)) & set(all_r)) >= 4:
            return True
    return False


def classify_postflop(hole_cards, community_cards):
    if len(hole_cards) < 2 or len(community_cards) < 3:
        return None

    r1, r2       = _rank(hole_cards[0][0]), _rank(hole_cards[1][0])
    s1, s2       = hole_cards[0][1], hole_cards[1][1]
    board_ranks  = [_rank(c[0]) for c in community_cards]
    board_suits  = [c[1]        for c in community_cards]
    hole_ranks   = [r1, r2]
    hole_suits   = [s1, s2]
    is_pair      = r1 == r2
    is_suited    = s1 == s2

    # 1. Monster: quads or full house
    if is_pair:
        cnt = board_ranks.count(r1)
        if cnt >= 2:
            return 'monster'   # quads (3+ of same rank on board)
        if cnt == 1:
            other = [r for r in board_ranks if r != r1]
            # set + board pair = full house; otherwise trips (set)
            return 'monster' if len(other) > len(set(other)) else 'set'
        # cnt == 0: pocket pair doesn't appear on board
        # Check if board carries a pair → pocket pair + board pair = two pair
        board_pair = len(board_ranks) > len(set(board_ranks))
        if board_pair:
            return 'two_pair'  # e.g. AA hole + KK2 board

    # 2. Made flush (5 cards same suit)
    if is_suited and board_suits.count(s1) >= 3:
        return 'flush'

    # 3. Made straight
    if _has_straight(hole_ranks, board_ranks):
        return 'straight'

    # 4. Two pair (both unpaired hole cards pair different board cards)
    if r1 != r2 and r1 in board_ranks and r2 in board_ranks:
        return 'two_pair'

    # 5. Top pair (hole card equals the highest board card, not a pocket pair)
    max_board = max(board_ranks)
    if max_board in hole_ranks and not is_pair:
        return 'top_pair'

    # 6. Flush draw (4 to flush, not already a made flush)
    if _has_flush_draw(hole_suits, board_suits):
        return 'flush_draw'

    # 7. OESD / gutshot (pure draw, no pair)
    if (not is_pair and r1 not in board_ranks and r2 not in board_ranks
            and _has_oesd(hole_ranks, board_ranks)):
        return 'oesd'

    # 8. Complete air: no pair, no draw, weak high card
    if (r1 not in board_ranks and r2 not in board_ranks
            and not is_pair and max(r1, r2) <= 9
            and board_suits.count(s1) < 2
            and board_suits.count(s2) < 2):
        return 'air'

    return None


# ---------------------------------------------------------------------------
# Spot definitions
# ---------------------------------------------------------------------------
# Each spot: category, street, label, hint, and threshold keys:
#   max_fold, min_fold, min_aggr, max_aggr  (all optional)
# ---------------------------------------------------------------------------
SPOTS = [
    # ── Preflop: premium hands ──────────────────────────────────────────────
    {'category': 'AA',        'street': 'preflop', 'label': 'AA preflop',
     'hint': 'fold<10%  aggr>60%', 'max_fold': 0.10, 'min_aggr': 0.60},
    {'category': 'KK',        'street': 'preflop', 'label': 'KK preflop',
     'hint': 'fold<10%  aggr>50%', 'max_fold': 0.10, 'min_aggr': 0.50},
    {'category': 'QQ',        'street': 'preflop', 'label': 'QQ preflop',
     'hint': 'fold<15%  aggr>35%', 'max_fold': 0.15, 'min_aggr': 0.35},
    {'category': 'JJ-TT',    'street': 'preflop', 'label': 'JJ-TT preflop',
     'hint': 'fold<20%',           'max_fold': 0.20},
    {'category': 'AKo',       'street': 'preflop', 'label': 'AKo preflop',
     'hint': 'fold<15%  aggr>40%', 'max_fold': 0.15, 'min_aggr': 0.40},
    {'category': 'AKs',       'street': 'preflop', 'label': 'AKs preflop',
     'hint': 'fold<10%  aggr>50%', 'max_fold': 0.10, 'min_aggr': 0.50},
    {'category': 'AQo-AJo',  'street': 'preflop', 'label': 'AQo-AJo preflop',
     'hint': 'fold<20%',           'max_fold': 0.20},
    # ── Preflop: trash ──────────────────────────────────────────────────────
    {'category': '72o',         'street': 'preflop', 'label': '7-2 offsuit (worst hand)',
     'hint': 'fold>30%',           'min_fold': 0.30},
    {'category': 'trash_offsuit','street': 'preflop', 'label': 'Trash offsuit (hi≤8, gap≥3)',
     'hint': 'fold>25%',           'min_fold': 0.25},
    # ── Post-flop: made hands ───────────────────────────────────────────────
    {'category': 'monster',   'street': 'postflop', 'label': 'Full house / quads',
     'hint': 'fold<3%  aggr>35%',  'max_fold': 0.03, 'min_aggr': 0.35},
    {'category': 'flush',     'street': 'postflop', 'label': 'Made flush',
     'hint': 'fold<5%  aggr>25%',  'max_fold': 0.05, 'min_aggr': 0.25},
    {'category': 'straight',  'street': 'postflop', 'label': 'Made straight',
     'hint': 'fold<5%  aggr>20%',  'max_fold': 0.05, 'min_aggr': 0.20},
    {'category': 'set',       'street': 'postflop', 'label': 'Set (trips with pocket pair)',
     'hint': 'fold<5%  aggr>30%',  'max_fold': 0.05, 'min_aggr': 0.30},
    {'category': 'two_pair',  'street': 'postflop', 'label': 'Two pair (both hole cards hit)',
     'hint': 'fold<10%  aggr>20%', 'max_fold': 0.10, 'min_aggr': 0.20},
    {'category': 'top_pair',  'street': 'postflop', 'label': 'Top pair',
     'hint': 'fold<15%  aggr>15%', 'max_fold': 0.15, 'min_aggr': 0.15},
    # ── Post-flop: draws (should semi-bluff occasionally) ───────────────────
    {'category': 'flush_draw','street': 'postflop', 'label': 'Flush draw (4 to flush)',
     'hint': 'aggr>10%',           'min_aggr': 0.10},
    {'category': 'oesd',      'street': 'postflop', 'label': 'Open-ended straight draw',
     'hint': 'aggr>10%',           'min_aggr': 0.10},
    # ── Post-flop: air ──────────────────────────────────────────────────────
    {'category': 'air',       'street': 'postflop', 'label': 'Complete air (no pair/draw, hi≤9)',
     'hint': 'aggr<35%',           'max_aggr': 0.35},
]


def _spot_passes(spot, fold_pct, aggr_pct):
    if 'max_fold' in spot and fold_pct > spot['max_fold']:
        return False
    if 'min_fold' in spot and fold_pct < spot['min_fold']:
        return False
    if 'min_aggr' in spot and aggr_pct < spot['min_aggr']:
        return False
    if 'max_aggr' in spot and aggr_pct > spot['max_aggr']:
        return False
    return True


# ---------------------------------------------------------------------------
# GTO reference targets (6-max, approximate)
# ---------------------------------------------------------------------------
GTO_OPEN = {   # open-raise % when first to act
    'UTG': 0.15, 'MP': 0.22, 'CO': 0.35, 'BTN': 0.45, 'SB': 0.35,
}
GTO_VPIP = {   # voluntary put-in %
    'UTG': 0.22, 'MP': 0.30, 'CO': 0.45, 'BTN': 0.55, 'SB': 0.50, 'BB': 0.60,
}
GTO_3BET = {   # 3-bet %
    'UTG': 0.04, 'MP': 0.05, 'CO': 0.07, 'BTN': 0.09, 'SB': 0.08, 'BB': 0.07,
}


# ---------------------------------------------------------------------------
# Legal-action-aware stat computation
# ---------------------------------------------------------------------------

def _compute_spot_stats(cat_samples):
    """
    Compute fold/call/aggr percentages correctly by only averaging each
    action over samples where that action was actually legal.

    cat_samples: list of (strategy_array, legal_set, context_dict)
    """
    if not cat_samples:
        return None
    n = len(cat_samples)

    fold_vals = [float(s[FOLD]) for s, L, _ in cat_samples if FOLD in L]
    call_vals = [float(s[CALL]) for s, L, _ in cat_samples if CALL in L]
    aggr_vals = [sum(float(s[a]) for a in AGGRESSIVE if a in L)
                 for s, L, _ in cat_samples]

    return {
        'n':        n,
        'fold_pct': float(np.mean(fold_vals)) if fold_vals else 0.0,
        'call_pct': float(np.mean(call_vals)) if call_vals else 0.0,
        'aggr_pct': float(np.mean(aggr_vals)),
    }


def _split_context(cat_samples, street):
    """Split samples into (not-facing-raise, facing-raise)."""
    key = 'facing_preflop_raise' if street == 'preflop' else 'facing_postflop_bet'
    facing     = [(s, L, ctx) for s, L, ctx in cat_samples if ctx.get(key)]
    not_facing = [(s, L, ctx) for s, L, ctx in cat_samples if not ctx.get(key)]
    return not_facing, facing


# ---------------------------------------------------------------------------
# Core collection loop
# ---------------------------------------------------------------------------

def collect_samples(deep_cfr, engine, num_games, strategy_fn):
    """
    Run num_games hands and record model strategy at every decision point.

    Returns:
      samples   – dict[category → list of (strategy, legal_set, ctx)]
      pos_stats – dict[pos_label → {hands, vpip, pfr, three_bet, three_bet_opp,
                                     open_raise, open_opp}]
    """
    samples   = defaultdict(list)
    pos_stats = defaultdict(lambda: {
        'hands': 0, 'vpip': 0.0, 'pfr': 0.0,
        'three_bet': 0.0, 'three_bet_opp': 0,
        'open_raise': 0.0, 'open_opp': 0,
    })

    for _ in range(num_games):
        engine.reset()
        acted_preflop = set()   # players who already had their first preflop action

        while not engine.is_done():
            agent = engine.current_agent
            if agent is None:
                break
            legal = engine.get_legal_actions()
            if not legal:
                break

            agent_idx = engine.agents.index(agent)
            stage     = engine.get_stage()
            obs       = engine.get_observation()
            ctx       = _game_context(engine, agent)

            cards     = engine.get_player_cards(agent)
            community = engine.get_community_cards()

            # Classify current spot
            cat = (classify_preflop(cards)
                   if stage == 'Preflop'
                   else classify_postflop(cards, community))

            strategy = strategy_fn(obs, agent_idx, legal)

            if cat is not None:
                samples[cat].append((strategy.copy(), set(legal), ctx))

            # ── Track VPIP / PFR / 3-bet per position (first preflop action) ──
            if stage == 'Preflop' and agent not in acted_preflop and ctx:
                pos = ctx.get('pos_label', 'UNK')
                pos_stats[pos]['hands'] += 1
                acted_preflop.add(agent)

                fold_p  = float(strategy[FOLD]) if FOLD in legal else 0.0
                raise_p = sum(float(strategy[a]) for a in AGGRESSIVE if a in legal)
                ca      = ctx.get('call_amount', 0)
                facing  = ctx.get('facing_preflop_raise', False)

                # VPIP: probability of voluntarily putting chips in
                # BB with call_amount==0 (free check) is excluded
                is_bb_free = (pos == 'BB' and ca == 0)
                if not is_bb_free:
                    pos_stats[pos]['vpip'] += (1.0 - fold_p)

                # PFR: probability of raising
                pos_stats[pos]['pfr'] += raise_p

                # 3-bet: raised facing a prior raise
                if facing:
                    pos_stats[pos]['three_bet_opp'] += 1
                    pos_stats[pos]['three_bet']     += raise_p

                # Open-raise: raised when no prior raise (not from BB)
                if not facing and pos != 'BB':
                    pos_stats[pos]['open_opp']   += 1
                    pos_stats[pos]['open_raise'] += raise_p

            # Sample action and advance
            probs  = strategy[legal].copy()
            probs /= probs.sum()
            action = int(np.random.choice(legal, p=probs))
            engine.step(action)

    return samples, pos_stats


# ---------------------------------------------------------------------------
# EV test: model player vs random opponents
# ---------------------------------------------------------------------------

def run_ev_test(deep_cfr, engine, num_games, strategy_fn):
    """Run model (player_0) vs random agents for num_games hands.
    Returns average chips won per hand (positive = profitable)."""
    model_agent  = 'player_0'
    total_reward = 0.0

    for _ in range(num_games):
        engine.reset()
        while not engine.is_done():
            agent = engine.current_agent
            if agent is None:
                break
            legal = engine.get_legal_actions()
            if not legal:
                break

            if agent == model_agent:
                obs      = engine.get_observation()
                idx      = engine.agents.index(agent)
                strategy = strategy_fn(obs, idx, legal)
                probs    = strategy[legal].copy()
                probs   /= probs.sum()
                action   = int(np.random.choice(legal, p=probs))
            else:
                action = int(np.random.choice(legal))

            engine.step(action)

        total_reward += engine.get_rewards().get(model_agent, 0.0)

    return total_reward / max(num_games, 1)


# ---------------------------------------------------------------------------
# Report printer helpers
# ---------------------------------------------------------------------------

def _print_spot_row(label, stats, spot, low_n_thresh=20, indent='  '):
    if stats is None:
        print(f"{indent}{label:<38} {'—':>5}  (no samples)")
        return None

    n        = stats['n']
    fold_pct = stats['fold_pct']
    call_pct = stats['call_pct']
    aggr_pct = stats['aggr_pct']
    passed   = _spot_passes(spot, fold_pct, aggr_pct)
    low_n    = n < low_n_thresh

    status = 'PASS ✓' if passed else f"FAIL  (want: {spot['hint']})"
    if low_n:
        status += ' [low N]'

    print(f"{indent}{label:<38} {n:>5}  {fold_pct:>6.1%}  {call_pct:>6.1%}"
          f"  {aggr_pct:>6.1%}  {status}")
    return passed, low_n


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------

def run_validation(deep_cfr, engine, num_games: int = 1000,
                   verbose: bool = True, strategy_mode: str = 'avg',
                   ev_games: int = 0) -> dict:
    """
    Run validation and return a results dict.
    Backward-compatible with train_gpu.py usage:
        run_validation(deep_cfr, engine, num_games=500, verbose=False)

    strategy_mode: 'avg' (default) | 'current' | 'both'
    ev_games: if > 0, also run EV test vs random agents
    """
    for net in deep_cfr.advantage_networks + deep_cfr.strategy_networks:
        net.eval()

    def _make_strategy_fn(mode):
        if mode == 'current':
            return lambda obs, idx, lg: _current_strategy(deep_cfr, obs, idx, lg)
        return lambda obs, idx, lg: _avg_strategy(deep_cfr, obs, idx, lg)

    modes = ['avg', 'current'] if strategy_mode == 'both' else [strategy_mode]

    all_mode_results = {}
    for mode in modes:
        sfn = _make_strategy_fn(mode)
        samples, pos_stats = collect_samples(deep_cfr, engine, num_games, sfn)
        all_mode_results[mode] = (samples, pos_stats)

    # Use the first mode as the canonical result for backward compat
    primary_mode             = modes[0]
    samples, pos_stats       = all_mode_results[primary_mode]

    results = {}
    passed  = 0
    total   = 0

    if verbose:
        print()
        print('=' * 80)
        print(f'  SPOT VALIDATION  ({num_games} games, iter {deep_cfr.iteration}, '
              f'strategy={strategy_mode})')
        print('=' * 80)

    current_street = None
    hdr = (f"  {'Spot':<38} {'N':>5}  {'fold%':>6}  {'call%':>6}"
           f"  {'aggr%':>6}  Status")
    sep = '  ' + '-' * 74

    for spot in SPOTS:
        cat    = spot['category']
        street = spot['street']
        label  = spot['label']

        if verbose and street != current_street:
            current_street = street
            section = {'preflop': '── PREFLOP ──', 'postflop': '── POST-FLOP ──'}.get(street, '')
            print(f'\n  {section}')
            print(hdr)
            print(sep)

        # ── Primary mode spot ──
        cat_samples = samples.get(cat, [])
        stats       = _compute_spot_stats(cat_samples)

        if verbose:
            result = _print_spot_row(label, stats, spot)
        else:
            result = None

        # ── Context split (preflop: first-to-act vs facing raise) ──
        if verbose and cat_samples:
            not_facing, facing = _split_context(cat_samples, street)
            for sub_label, sub_samples in [
                    ('  ├─ first to act', not_facing),
                    ('  └─ facing raise/bet', facing)]:
                sub_stats = _compute_spot_stats(sub_samples)
                if sub_stats and sub_stats['n'] >= 5:
                    _print_spot_row(sub_label, sub_stats, spot,
                                    low_n_thresh=10, indent='    ')

        # ── Secondary mode comparison (if 'both') ──
        if verbose and strategy_mode == 'both' and len(modes) > 1:
            other_mode    = modes[1]
            other_samples = all_mode_results[other_mode][0].get(cat, [])
            other_stats   = _compute_spot_stats(other_samples)
            if other_stats and other_stats['n'] >= 5:
                _print_spot_row(f'  [current] {label}', other_stats, spot,
                                indent='    ')

        # Accumulate pass/fail (primary mode, exclude low-N postflop)
        if stats:
            n        = stats['n']
            low_n    = n < 20
            fold_pct = stats['fold_pct']
            aggr_pct = stats['aggr_pct']
            passed_spot = _spot_passes(spot, fold_pct, aggr_pct)

            results[cat] = {
                'n':        n,
                'fold_pct': fold_pct,
                'call_pct': stats['call_pct'],
                'aggr_pct': aggr_pct,
                'passed':   passed_spot,
                'low_n':    low_n,
            }
            if not low_n:
                if passed_spot:
                    passed += 1
                total += 1
        else:
            results[cat] = {'n': 0, 'passed': None}

    # ── Position-stratified stats ────────────────────────────────────────────
    if verbose and pos_stats:
        print(f'\n  ── POSITION STATS (preflop, all players using model strategy) ──')
        pos_hdr  = (f"  {'Pos':<6} {'Hands':>6}  {'VPIP':>6}  {'PFR':>6}"
                    f"  {'Open%':>6}  {'3-bet%':>7}  Open-GTO  VPIP-GTO")
        pos_sep  = '  ' + '-' * 70
        print(pos_hdr)
        print(pos_sep)

        pos_order = ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']
        for pos in pos_order:
            if pos not in pos_stats:
                continue
            ps    = pos_stats[pos]
            hands = ps['hands']
            if hands < 5:
                continue

            vpip_pct  = ps['vpip']  / hands
            pfr_pct   = ps['pfr']   / hands
            open_pct  = (ps['open_raise'] / ps['open_opp']
                         if ps['open_opp'] > 0 else float('nan'))
            tbet_pct  = (ps['three_bet'] / ps['three_bet_opp']
                         if ps['three_bet_opp'] > 0 else float('nan'))

            gto_open = GTO_OPEN.get(pos, float('nan'))
            gto_vpip = GTO_VPIP.get(pos, float('nan'))

            open_str = f'{open_pct:6.1%}' if not np.isnan(open_pct) else '  n/a '
            tbet_str = f'{tbet_pct:7.1%}' if not np.isnan(tbet_pct) else '   n/a '
            gopen    = f'{gto_open:>8.0%}' if not np.isnan(gto_open) else '     n/a'
            gvpip    = f'{gto_vpip:>9.0%}' if not np.isnan(gto_vpip) else '      n/a'

            print(f"  {pos:<6} {hands:>6}  {vpip_pct:>6.1%}  {pfr_pct:>6.1%}"
                  f"  {open_str}  {tbet_str}  {gopen}  {gvpip}")

        # ── Aggregate VPIP/PFR/3-bet across all positions ──
        total_hands = sum(ps['hands'] for ps in pos_stats.values())
        if total_hands > 0:
            agg_vpip = sum(ps['vpip'] for ps in pos_stats.values()) / total_hands
            agg_pfr  = sum(ps['pfr']  for ps in pos_stats.values()) / total_hands
            tot_3opp = sum(ps['three_bet_opp'] for ps in pos_stats.values())
            agg_3bet = (sum(ps['three_bet'] for ps in pos_stats.values()) / tot_3opp
                        if tot_3opp > 0 else 0.0)
            print(pos_sep)
            print(f"  {'ALL':<6} {total_hands:>6}  {agg_vpip:>6.1%}  {agg_pfr:>6.1%}"
                  f"              (GTO 6-max: VPIP≈23%, PFR≈17%, 3-bet≈8%)")
            print(f"  3-bet (all positions): {agg_3bet:.1%}")

    # ── EV test ─────────────────────────────────────────────────────────────
    if ev_games > 0:
        if verbose:
            print(f'\n  ── EV TEST (model player_0 vs random, {ev_games} hands) ──')
        sfn    = _make_strategy_fn(primary_mode)
        ev_per_hand = run_ev_test(deep_cfr, engine, ev_games, sfn)
        results['_ev'] = ev_per_hand
        if verbose:
            sign = '+' if ev_per_hand >= 0 else ''
            print(f"  Avg chips won per hand: {sign}{ev_per_hand:.2f}  "
                  f"(random baseline: 0.00)")

    # ── Summary ──────────────────────────────────────────────────────────────
    if verbose:
        print()
        print('  ' + '-' * 74)
        verdict = ('Model shows basic poker understanding'
                   if total > 0 and passed >= total * 0.6
                   else 'Model needs more training')
        print(f"  Result: {passed}/{total} spots passed (low-N excluded)  —  {verdict}")
        print('=' * 80)
        print()

    results['_summary'] = {'passed': passed, 'total': total}
    return results


# ---------------------------------------------------------------------------
# Compact one-liner for training log  (backward-compatible)
# ---------------------------------------------------------------------------

def validation_summary_line(results: dict) -> str:
    parts = []
    for spot in SPOTS:
        cat = spot['category']
        if cat in results and results[cat].get('n', 0) >= 5:
            r   = results[cat]
            tag = 'ok' if r.get('passed') else 'BAD'
            if r.get('low_n'):
                tag += '?'
            parts.append(
                f"{cat}:fold={r['fold_pct']:.0%}"
                f",call={r['call_pct']:.0%}"
                f",raise={r['aggr_pct']:.0%}({tag})"
            )
    s = results.get('_summary', {})
    ev = results.get('_ev')
    ev_str = f" EV={ev:+.1f}" if ev is not None else ''
    return f"[spots {s.get('passed',0)}/{s.get('total',0)}{ev_str}] " + '  '.join(parts)


# ---------------------------------------------------------------------------
# Engine factory (DRY helper used in both auto-detect and main engine)
# ---------------------------------------------------------------------------

def _make_engine(num_players: int) -> EnhancedPokerEngine:
    return EnhancedPokerEngine(
        num_players=num_players,
        add_position_features=True,
        add_pot_odds_features=True,
        add_hand_strength_features=True,
        add_stack_features=True,
        add_equity_features=True,
        add_street_features=True,
        add_game_state_features=True,
        add_betting_history_features=True,
    )


# ---------------------------------------------------------------------------
# Model loader helper
# ---------------------------------------------------------------------------

def _load_model(model_path: Path, config: dict, engine) -> 'DeepCFR':
    deep_cfr = DeepCFR(
        observation_size=engine.get_observation_space_size(),
        action_size=engine.get_action_space_size(),
        num_players=engine.num_players,
        hidden_layers=config['network']['hidden_layers'],
        device='cpu',
    )
    deep_cfr.load_models(str(model_path))
    return deep_cfr


def _autodetect_players(model_path: Path, default: int = 6) -> int:
    try:
        ckpt       = torch.load(str(model_path), map_location='cpu')
        input_size = ckpt['advantage_networks'][0]['network.0.weight'].shape[1]
        for p in range(2, 10):
            if _make_engine(p).get_observation_space_size() == input_size:
                return p
    except Exception:
        pass
    return default


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def run_comparison(model_a: 'DeepCFR', model_b: 'DeepCFR', engine,
                   num_games: int = 1000, strategy_mode: str = 'avg') -> None:
    """
    Side-by-side comparison of two checkpoints.
    Prints a diff table for every spot and the position stats.
    """
    for m in (model_a, model_b):
        for net in m.advantage_networks + m.strategy_networks:
            net.eval()

    def _sfn(model, mode):
        if mode == 'current':
            return lambda obs, idx, lg: _current_strategy(model, obs, idx, lg)
        return lambda obs, idx, lg: _avg_strategy(model, obs, idx, lg)

    print(f'\nCollecting samples for model A (iter {model_a.iteration})...')
    samp_a, pos_a = collect_samples(model_a, engine, num_games, _sfn(model_a, strategy_mode))
    print(f'Collecting samples for model B (iter {model_b.iteration})...')
    samp_b, pos_b = collect_samples(model_b, engine, num_games, _sfn(model_b, strategy_mode))

    tag_a = f'iter_{model_a.iteration}'
    tag_b = f'iter_{model_b.iteration}'

    W = 80
    print()
    print('=' * W)
    print(f'  COMPARISON:  {tag_a}  vs  {tag_b}  '
          f'({num_games} games each, strategy={strategy_mode})')
    print('=' * W)

    # Column layout:
    # Spot(30) | N_a(5) fold_a(6) aggr_a(6) pass_a(6) | N_b(5) fold_b(6) aggr_b(6) pass_b(6) | Δfold Δaggr
    hdr1 = (f"  {'Spot':<30}  "
            f"{'── ' + tag_a + ' ──':^24}  "
            f"{'── ' + tag_b + ' ──':^24}  "
            f"{'Δfold':>6} {'Δaggr':>6}")
    hdr2 = (f"  {'':<30}  "
            f"{'N':>5} {'fold%':>6} {'aggr%':>6} {'ok?':>5}  "
            f"{'N':>5} {'fold%':>6} {'aggr%':>6} {'ok?':>5}  "
            f"{'':>6} {'':>6}")
    sep  = '  ' + '-' * (W - 2)

    current_street = None
    pass_a_total = pass_b_total = 0
    spots_counted = 0

    for spot in SPOTS:
        cat    = spot['category']
        street = spot['street']
        label  = spot['label'][:30]

        if street != current_street:
            current_street = street
            section = {'preflop': '── PREFLOP ──', 'postflop': '── POST-FLOP ──'}.get(street, '')
            print(f'\n  {section}')
            print(hdr1)
            print(hdr2)
            print(sep)

        sa = _compute_spot_stats(samp_a.get(cat, []))
        sb = _compute_spot_stats(samp_b.get(cat, []))

        def _fmt(stats, s):
            if stats is None or stats['n'] == 0:
                return f"{'—':>5} {'—':>6} {'—':>6} {'—':>5}"
            p = _spot_passes(s, stats['fold_pct'], stats['aggr_pct'])
            ok = 'PASS' if p else 'FAIL'
            if stats['n'] < 20:
                ok = ok[:2] + '?'
            return (f"{stats['n']:>5} {stats['fold_pct']:>6.1%}"
                    f" {stats['aggr_pct']:>6.1%} {ok:>5}")

        row_a = _fmt(sa, spot)
        row_b = _fmt(sb, spot)

        # Delta columns
        if sa and sb and sa['n'] > 0 and sb['n'] > 0:
            df = sb['fold_pct'] - sa['fold_pct']
            da = sb['aggr_pct'] - sa['aggr_pct']
            arrow_f = ('↑' if df > 0.02 else '↓' if df < -0.02 else ' ')
            arrow_a = ('↑' if da > 0.02 else '↓' if da < -0.02 else ' ')
            delta = (f"{arrow_f}{df:>+5.1%} {arrow_a}{da:>+5.1%}")

            # Count passes (exclude low-N)
            low_n = (sa['n'] < 20 or sb['n'] < 20)
            if not low_n:
                spots_counted += 1
                if _spot_passes(spot, sa['fold_pct'], sa['aggr_pct']):
                    pass_a_total += 1
                if _spot_passes(spot, sb['fold_pct'], sb['aggr_pct']):
                    pass_b_total += 1
        else:
            delta = f"{'—':>7} {'—':>7}"

        print(f"  {label:<30}  {row_a}  {row_b}  {delta}")

    # ── Position stats comparison ────────────────────────────────────────────
    print(f'\n  ── POSITION STATS ──')
    pos_hdr = (f"  {'Pos':<5}  "
               f"{'── ' + tag_a + ' ──':^28}  "
               f"{'── ' + tag_b + ' ──':^28}  "
               f"{'ΔVPIP':>6} {'Δ3bet':>6}")
    pos_hdr2 = (f"  {'':<5}  "
                f"{'VPIP':>6} {'PFR':>6} {'Open%':>6} {'3bet%':>6}  "
                f"{'VPIP':>6} {'PFR':>6} {'Open%':>6} {'3bet%':>6}  "
                f"{'':>6} {'':>6}")
    print(pos_hdr)
    print(pos_hdr2)
    print('  ' + '-' * (W - 2))

    def _pos_row(ps, pos):
        if pos not in ps or ps[pos]['hands'] < 5:
            return f"{'—':>6} {'—':>6} {'—':>6} {'—':>6}"
        p     = ps[pos]
        h     = p['hands']
        vpip  = p['vpip'] / h
        pfr   = p['pfr']  / h
        opn   = p['open_raise'] / p['open_opp'] if p['open_opp'] > 0 else float('nan')
        tbet  = p['three_bet']  / p['three_bet_opp'] if p['three_bet_opp'] > 0 else float('nan')
        opn_s = f'{opn:6.1%}' if not np.isnan(opn) else '  n/a '
        tbt_s = f'{tbet:6.1%}' if not np.isnan(tbet) else '  n/a '
        return f"{vpip:>6.1%} {pfr:>6.1%} {opn_s} {tbt_s}"

    for pos in ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']:
        ra = _pos_row(pos_a, pos)
        rb = _pos_row(pos_b, pos)
        # Compute deltas
        if pos in pos_a and pos in pos_b:
            ha, hb = pos_a[pos]['hands'], pos_b[pos]['hands']
            if ha >= 5 and hb >= 5:
                dv = pos_b[pos]['vpip']/hb - pos_a[pos]['vpip']/ha
                oa = pos_a[pos]['three_bet_opp']
                ob = pos_b[pos]['three_bet_opp']
                dt = ((pos_b[pos]['three_bet']/ob if ob > 0 else 0)
                      - (pos_a[pos]['three_bet']/oa if oa > 0 else 0))
                av = '↑' if dv > 0.01 else '↓' if dv < -0.01 else ' '
                at = '↑' if dt > 0.005 else '↓' if dt < -0.005 else ' '
                delta = f"{av}{dv:>+5.1%} {at}{dt:>+5.1%}"
            else:
                delta = f"{'—':>7} {'—':>7}"
        else:
            delta = f"{'—':>7} {'—':>7}"
        print(f"  {pos:<5}  {ra}  {rb}  {delta}")

    # Aggregate 3-bet
    def _agg_3bet(ps):
        opp = sum(p['three_bet_opp'] for p in ps.values())
        return (sum(p['three_bet'] for p in ps.values()) / opp if opp > 0 else 0.0)

    d3 = _agg_3bet(pos_b) - _agg_3bet(pos_a)
    print('  ' + '-' * (W - 2))
    print(f"  Aggregate 3-bet:  {tag_a}={_agg_3bet(pos_a):.1%}  "
          f"{tag_b}={_agg_3bet(pos_b):.1%}  Δ={d3:+.1%}  (GTO≈8%)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print('  ' + '-' * (W - 2))
    diff = pass_b_total - pass_a_total
    sign = f'+{diff}' if diff > 0 else str(diff)
    print(f"  Spots passed (low-N excluded):  "
          f"{tag_a}: {pass_a_total}/{spots_counted}   "
          f"{tag_b}: {pass_b_total}/{spots_counted}  ({sign})")
    print('=' * W)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Validate model strategy on canonical spots across all streets'
    )
    parser.add_argument('--model',    type=str, required=True,
                        help='Path to .pt checkpoint file')
    parser.add_argument('--compare',  type=str, default=None,
                        help='Second checkpoint to compare against --model side-by-side')
    parser.add_argument('--samples',  type=int, default=1000,
                        help='Number of games to sample (default: 1000)')
    parser.add_argument('--strategy', type=str, default='avg',
                        choices=['avg', 'current', 'both'],
                        help='Strategy to evaluate: avg (default), current, or both')
    parser.add_argument('--ev-games', type=int, default=0,
                        help='If > 0, run EV test vs random agents for this many hands')
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f'Model not found: {model_path}')
        sys.exit(1)

    config_path = Path(__file__).parent.parent / 'configs/training_config.json'
    config      = load_config(str(config_path))

    num_players = _autodetect_players(model_path)
    print(f'Auto-detected: {num_players} players')
    engine = _make_engine(num_players)

    deep_cfr = _load_model(model_path, config, engine)
    print(f'Loaded A: {model_path.name}  (iteration {deep_cfr.iteration})')

    # ── Comparison mode ──────────────────────────────────────────────────────
    if args.compare:
        compare_path = Path(args.compare)
        if not compare_path.exists():
            print(f'Compare model not found: {compare_path}')
            sys.exit(1)
        deep_cfr_b = _load_model(compare_path, config, engine)
        print(f'Loaded B: {compare_path.name}  (iteration {deep_cfr_b.iteration})')
        run_comparison(
            deep_cfr, deep_cfr_b, engine,
            num_games=args.samples,
            strategy_mode=args.strategy if args.strategy != 'both' else 'avg',
        )
        return

    # ── Single model validation ──────────────────────────────────────────────
    run_validation(
        deep_cfr, engine,
        num_games=args.samples,
        strategy_mode=args.strategy,
        ev_games=args.ev_games,
    )


if __name__ == '__main__':
    main()
