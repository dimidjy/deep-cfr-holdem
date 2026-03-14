"""
Poker spot validator — postflop model quality assessment.

Tests postflop decision quality across streets and bet contexts:
  1. Made hands that should never/rarely fold (set, flush, straight, two pair, etc.)
  2. Drawing hands (flush draw, OESD) — semi-bluff frequency
  3. Air / weak hands — bluff frequency control
  4. Split by street (flop / turn / river) and context (first-to-act vs facing bet)
  5. Bet-size awareness: facing small bet vs facing big bet

Usage:
  python3 scripts/validate_spots.py --model models/checkpoint_iter_500.pt
  python3 scripts/validate_spots.py --model models/checkpoint_iter_500.pt --samples 2000
  python3 scripts/validate_spots.py --model models/checkpoint_iter_500.pt --strategy both
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
from src.preflop_ranges import PreflopRanges

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
        pot_size     = engine.get_pot_size()
        bet_to_pot   = call_amount / max(pot_size, 1)

        return {
            'rel_pos':               rel_pos,
            'pos_label':             pos_label,
            'call_amount':           call_amount,
            'pot_size':              pot_size,
            'bet_to_pot':            bet_to_pot,
            # Preflop: someone raised above BB (> 2 chips)
            'facing_preflop_raise':  call_amount > 2,
            # Postflop: someone bet (any call_amount > 0)
            'facing_bet':            call_amount > 0,
            # Bet size buckets (relative to pot)
            'facing_small_bet':      0 < bet_to_pot <= 0.5,
            'facing_big_bet':        bet_to_pot > 0.5,
        }
    except Exception:
        return {}




# ---------------------------------------------------------------------------
# Preflop hand classifier (exported for validate_vs_hh.py)
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
# GTO reference targets (exported for validate_vs_hh.py)
# ---------------------------------------------------------------------------
GTO_OPEN = {'UTG': 0.15, 'MP': 0.22, 'CO': 0.35, 'BTN': 0.45, 'SB': 0.35}
GTO_VPIP = {'UTG': 0.22, 'MP': 0.30, 'CO': 0.45, 'BTN': 0.55, 'SB': 0.50, 'BB': 0.60}
GTO_3BET = {'UTG': 0.04, 'MP': 0.05, 'CO': 0.07, 'BTN': 0.09, 'SB': 0.08, 'BB': 0.07}


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
    max_board    = max(board_ranks)

    # 1. Monster: quads or full house
    if is_pair:
        cnt = board_ranks.count(r1)
        if cnt >= 2:
            return 'monster'   # quads
        if cnt == 1:
            other = [r for r in board_ranks if r != r1]
            if len(other) > len(set(other)):
                return 'monster'  # full house
            return 'set'

    # 2. Made flush (5 cards same suit)
    if is_suited and board_suits.count(s1) >= 3:
        return 'flush'

    # 3. Made straight
    if _has_straight(hole_ranks, board_ranks):
        return 'straight'

    # 4. Set from non-pair (trips: one hole card matches board pair)
    for r in hole_ranks:
        if board_ranks.count(r) >= 2:
            return 'trips'

    # 5. Two pair (both unpaired hole cards pair different board cards)
    if r1 != r2 and r1 in board_ranks and r2 in board_ranks:
        return 'two_pair'

    # 6. Overpair (pocket pair above all board cards)
    if is_pair and r1 > max_board:
        return 'overpair'

    # 7. Top pair (hole card equals the highest board card)
    if max_board in hole_ranks and not is_pair:
        return 'top_pair'

    # 8. Middle/bottom pair (hole card matches non-top board card)
    sorted_board = sorted(set(board_ranks), reverse=True)
    if not is_pair:
        for r in hole_ranks:
            if r in board_ranks and r != max_board:
                return 'mid_low_pair'

    # 9. Flush draw (4 to flush, not already a made flush)
    if _has_flush_draw(hole_suits, board_suits):
        return 'flush_draw'

    # 10. OESD / straight draw (pure draw, no pair)
    if (not is_pair and r1 not in board_ranks and r2 not in board_ranks
            and _has_oesd(hole_ranks, board_ranks)):
        return 'oesd'

    # 11. Complete air: no pair, no draw, weak high card
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
    # ── NEVER FOLD: strongest made hands ────────────────────────────────────
    {'category': 'monster',   'label': 'Full house / quads',
     'hint': 'fold<2%  aggr>40%',  'max_fold': 0.02, 'min_aggr': 0.40},
    {'category': 'flush',     'label': 'Made flush',
     'hint': 'fold<3%  aggr>25%',  'max_fold': 0.03, 'min_aggr': 0.25},
    {'category': 'straight',  'label': 'Made straight',
     'hint': 'fold<5%  aggr>20%',  'max_fold': 0.05, 'min_aggr': 0.20},
    {'category': 'set',       'label': 'Set (pocket pair hit board)',
     'hint': 'fold<3%  aggr>30%',  'max_fold': 0.03, 'min_aggr': 0.30},
    {'category': 'trips',     'label': 'Trips (hole card + board pair)',
     'hint': 'fold<5%  aggr>20%',  'max_fold': 0.05, 'min_aggr': 0.20},
    # ── RARELY FOLD: strong hands ───────────────────────────────────────────
    {'category': 'two_pair',  'label': 'Two pair',
     'hint': 'fold<10%  aggr>20%', 'max_fold': 0.10, 'min_aggr': 0.20},
    {'category': 'overpair',  'label': 'Overpair (pair > board)',
     'hint': 'fold<10%  aggr>25%', 'max_fold': 0.10, 'min_aggr': 0.25},
    {'category': 'top_pair',  'label': 'Top pair',
     'hint': 'fold<15%  aggr>15%', 'max_fold': 0.15, 'min_aggr': 0.15},
    # ── MARGINAL: can fold sometimes ────────────────────────────────────────
    {'category': 'mid_low_pair', 'label': 'Middle / bottom pair',
     'hint': 'fold<50%',          'max_fold': 0.50},
    # ── DRAWS: semi-bluff ───────────────────────────────────────────────────
    {'category': 'flush_draw','label': 'Flush draw (4 to flush)',
     'hint': 'fold<40%  aggr>10%', 'max_fold': 0.40, 'min_aggr': 0.10},
    {'category': 'oesd',      'label': 'Open-ended straight draw',
     'hint': 'fold<45%  aggr>10%', 'max_fold': 0.45, 'min_aggr': 0.10},
    # ── AIR: should fold facing bets, limited bluffs ────────────────────────
    {'category': 'air',       'label': 'Complete air (no pair/draw, hi<=9)',
     'hint': 'aggr<45%',          'max_aggr': 0.45},
]

# Context-specific thresholds (applied to sub-splits)
CONTEXT_THRESHOLDS = {
    # Strong hands should NEVER fold facing a small bet
    'monster':    {'facing_small': {'max_fold': 0.01}, 'facing_big': {'max_fold': 0.03}},
    'flush':      {'facing_small': {'max_fold': 0.02}, 'facing_big': {'max_fold': 0.05}},
    'straight':   {'facing_small': {'max_fold': 0.03}, 'facing_big': {'max_fold': 0.08}},
    'set':        {'facing_small': {'max_fold': 0.02}, 'facing_big': {'max_fold': 0.05}},
    'trips':      {'facing_small': {'max_fold': 0.03}, 'facing_big': {'max_fold': 0.08}},
    'two_pair':   {'facing_small': {'max_fold': 0.05}, 'facing_big': {'max_fold': 0.15}},
    'overpair':   {'facing_small': {'max_fold': 0.05}, 'facing_big': {'max_fold': 0.15}},
    'top_pair':   {'facing_small': {'max_fold': 0.08}, 'facing_big': {'max_fold': 0.25}},
    # Air facing bet should fold often
    'air':        {'facing_small': {'min_fold': 0.40}, 'facing_big': {'min_fold': 0.60}},
}


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


def _split_context(cat_samples):
    """Split samples into (first-to-act, facing-small-bet, facing-big-bet)."""
    first_act  = [(s, L, ctx) for s, L, ctx in cat_samples if not ctx.get('facing_bet')]
    small_bet  = [(s, L, ctx) for s, L, ctx in cat_samples if ctx.get('facing_small_bet')]
    big_bet    = [(s, L, ctx) for s, L, ctx in cat_samples if ctx.get('facing_big_bet')]
    return first_act, small_bet, big_bet


def _split_by_street(cat_samples):
    """Split samples by street."""
    flop  = [(s, L, ctx) for s, L, ctx in cat_samples if ctx.get('street') == 'Flop']
    turn  = [(s, L, ctx) for s, L, ctx in cat_samples if ctx.get('street') == 'Turn']
    river = [(s, L, ctx) for s, L, ctx in cat_samples if ctx.get('street') == 'River']
    return flop, turn, river


# ---------------------------------------------------------------------------
# Core collection loop
# ---------------------------------------------------------------------------

def collect_samples(deep_cfr, engine, num_games, strategy_fn,
                    preflop_ranges=None):
    """
    Run num_games hands and record model strategy at every postflop decision.

    Returns:
      samples – dict[category → list of (strategy, legal_set, ctx)]
    """
    samples = defaultdict(list)
    num_players = engine.num_players

    for _ in range(num_games):
        engine.reset()

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
            ctx['street'] = stage  # 'Preflop', 'Flop', 'Turn', 'River'

            # Use preflop ranges if available
            if stage == 'Preflop' and preflop_ranges is not None:
                cards = engine.get_player_cards(agent)
                button_idx = engine._button
                rel_pos = (agent_idx - button_idx) % num_players
                pot_bb = engine.get_pot_size() / 2.0
                strategy = preflop_ranges.get_strategy(
                    cards, rel_pos, pot_bb, legal, deep_cfr.action_size)
            else:
                strategy = strategy_fn(obs, agent_idx, legal)

            # Only classify and record postflop decisions
            if stage != 'Preflop':
                cards     = engine.get_player_cards(agent)
                community = engine.get_community_cards()
                cat = classify_postflop(cards, community)
                if cat is not None:
                    samples[cat].append((strategy.copy(), set(legal), ctx))

            # Sample action and advance
            probs  = strategy[legal].copy()
            probs /= probs.sum()
            action = int(np.random.choice(legal, p=probs))
            engine.step(action)

    return samples


# ---------------------------------------------------------------------------
# EV test: model player vs random opponents
# ---------------------------------------------------------------------------

def run_ev_test(deep_cfr, engine, num_games, strategy_fn,
                preflop_ranges=None):
    """Run model (player_0) vs random agents for num_games hands.
    Returns average chips won per hand (positive = profitable)."""
    model_agent  = 'player_0'
    total_reward = 0.0
    num_players  = engine.num_players

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
                stage    = engine.get_stage()

                if stage == 'Preflop' and preflop_ranges is not None:
                    cards = engine.get_player_cards(agent)
                    button_idx = engine._button
                    rel_pos = (idx - button_idx) % num_players
                    pot_bb = engine.get_pot_size() / 2.0
                    strategy = preflop_ranges.get_strategy(
                        cards, rel_pos, pot_bb, legal, deep_cfr.action_size)
                else:
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
                   ev_games: int = 0, preflop_ranges=None) -> dict:
    """
    Run postflop validation and return a results dict.
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
        samples = collect_samples(deep_cfr, engine, num_games, sfn,
                                  preflop_ranges=preflop_ranges)
        all_mode_results[mode] = samples

    primary_mode = modes[0]
    samples      = all_mode_results[primary_mode]

    results = {}
    passed  = 0
    total   = 0

    if verbose:
        print()
        print('=' * 90)
        print(f'  POSTFLOP SPOT VALIDATION  ({num_games} games, iter {deep_cfr.iteration}, '
              f'strategy={strategy_mode})')
        print('=' * 90)
        hdr = (f"  {'Spot':<35} {'N':>5}  {'fold%':>6}  {'call%':>6}"
               f"  {'aggr%':>6}  Status")
        sep = '  ' + '-' * 84
        print(hdr)
        print(sep)

    for spot in SPOTS:
        cat   = spot['category']
        label = spot['label']

        cat_samples = samples.get(cat, [])
        stats       = _compute_spot_stats(cat_samples)

        if verbose:
            _print_spot_row(label, stats, spot)

        # ── Context splits: first-to-act / facing small / facing big ──
        if verbose and cat_samples:
            first_act, small_bet, big_bet = _split_context(cat_samples)
            ctx_thresholds = CONTEXT_THRESHOLDS.get(cat, {})

            sub_items = [
                ('first to act',     first_act,  {}),
                ('facing small bet', small_bet,  ctx_thresholds.get('facing_small', {})),
                ('facing big bet',   big_bet,    ctx_thresholds.get('facing_big', {})),
            ]
            for i, (sub_label, sub_samples, sub_thresh) in enumerate(sub_items):
                sub_stats = _compute_spot_stats(sub_samples)
                if sub_stats and sub_stats['n'] >= 5:
                    is_last = (i == len(sub_items) - 1) or all(
                        _compute_spot_stats(sub_items[j][1]) is None
                        or _compute_spot_stats(sub_items[j][1])['n'] < 5
                        for j in range(i + 1, len(sub_items))
                    )
                    prefix = '  └─' if is_last else '  ├─'
                    # Use context-specific thresholds if available
                    ctx_spot = dict(spot)
                    ctx_spot.update(sub_thresh)
                    if sub_thresh:
                        ctx_spot['hint'] = '  '.join(
                            f"{k.replace('_', '')}{v:.0%}"
                            for k, v in sub_thresh.items()
                        )
                    _print_spot_row(f'{prefix} {sub_label}', sub_stats, ctx_spot,
                                    low_n_thresh=10, indent='    ')

            # ── Street breakdown ──
            flop, turn, river = _split_by_street(cat_samples)
            street_items = [('flop', flop), ('turn', turn), ('river', river)]
            street_parts = []
            for sname, ssamples in street_items:
                ss = _compute_spot_stats(ssamples)
                if ss and ss['n'] >= 3:
                    street_parts.append(f"{sname}:{ss['n']}x f={ss['fold_pct']:.0%}")
            if street_parts:
                print(f"      [{' | '.join(street_parts)}]")

        # Accumulate pass/fail
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

    # ── EV test ─────────────────────────────────────────────────────────────
    if ev_games > 0:
        if verbose:
            print(f'\n  ── EV TEST (model player_0 vs random, {ev_games} hands) ──')
        sfn    = _make_strategy_fn(primary_mode)
        ev_per_hand = run_ev_test(deep_cfr, engine, ev_games, sfn,
                                  preflop_ranges=preflop_ranges)
        results['_ev'] = ev_per_hand
        if verbose:
            sign = '+' if ev_per_hand >= 0 else ''
            print(f"  Avg chips won per hand: {sign}{ev_per_hand:.2f}  "
                  f"(random baseline: 0.00)")

    # ── Summary ──────────────────────────────────────────────────────────────
    if verbose:
        print()
        print('  ' + '-' * 84)
        verdict = ('Postflop play looks solid'
                   if total > 0 and passed >= total * 0.7
                   else 'Postflop needs improvement')
        print(f"  Result: {passed}/{total} spots passed (low-N excluded)  —  {verdict}")
        print('=' * 90)
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
                f"{cat}:f={r['fold_pct']:.0%}"
                f",c={r['call_pct']:.0%}"
                f",r={r['aggr_pct']:.0%}({tag})"
            )
    s = results.get('_summary', {})
    ev = results.get('_ev')
    ev_str = f" EV={ev:+.1f}" if ev is not None else ''
    return f"[postflop {s.get('passed',0)}/{s.get('total',0)}{ev_str}] " + '  '.join(parts)


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
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Validate postflop model strategy on key spots'
    )
    parser.add_argument('--model',    type=str, required=True,
                        help='Path to .pt checkpoint file')
    parser.add_argument('--samples',  type=int, default=1000,
                        help='Number of games to sample (default: 1000)')
    parser.add_argument('--strategy', type=str, default='avg',
                        choices=['avg', 'current', 'both'],
                        help='Strategy to evaluate: avg (default), current, or both')
    parser.add_argument('--ev-games', type=int, default=0,
                        help='If > 0, run EV test vs random agents for this many hands')
    parser.add_argument('--no-preflop-ranges', action='store_true',
                        help='Disable preflop range override (default: ranges enabled)')
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
    print(f'Loaded: {model_path.name}  (iteration {deep_cfr.iteration})')

    pf_ranges = None if args.no_preflop_ranges else PreflopRanges()
    if pf_ranges:
        print('Preflop: GTO ranges (locked)')

    run_validation(
        deep_cfr, engine,
        num_games=args.samples,
        strategy_mode=args.strategy,
        ev_games=args.ev_games,
        preflop_ranges=pf_ranges,
    )


if __name__ == '__main__':
    main()
