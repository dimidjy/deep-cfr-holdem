#!/usr/bin/env python3
"""Research model weaknesses via strategy-perturbation sweeps over HH validation.

Mechanism: at every postflop decision point in the HH dataset, the model's
strategy can be perturbed by a hand-crafted rule (e.g. "shift +20% mass from
CALL to RAISE_HALF when hero has top pair facing a 3bet"). For each
perturbation we re-run the showdown-EV computation and measure the delta vs
baseline. Perturbations with positive delta on a non-trivial bucket size
indicate leaks the model could fix by retraining with an appropriate shaper.

Usage:
  python3 scripts/research_weaknesses.py --model models/best/checkpoint_iter_3600_v2.pt --max-hands 50000

Output:
  logs/research_weaknesses/<timestamp>/results.csv — ranked perturbation deltas
  Console summary highlighting the top positive-delta buckets.
"""

import argparse
import csv
import sys
import time
import warnings
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# Project root = parent of scripts/. validate_vs_hh uses scripts-relative
# imports (e.g. `from validate_spots import ...`), so add scripts/ to path too.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.WARNING)

# Reuse machinery from validate_vs_hh — bootstrap, EV computation, model loading.
import validate_vs_hh as vhh
from validate_spots import classify_postflop
from src.hand_history import ObservationBuilder
from src.preflop_ranges import PreflopRanges, FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN
from src.utils import load_config

# Action helpers
AGGRESSIVE = {RAISE_HALF, RAISE_POT, ALL_IN}


def _bb_per_100(ev_data: dict) -> float:
    """Compute showdown-only BB/100 from compute_showdown_ev's output."""
    total_sp = ev_data['total_showdown_players']
    if total_sp == 0:
        return 0.0
    ev_diff_total = sum(ev_data['ev_diffs'])
    ev_per_hand = ev_diff_total / total_sp
    return (ev_per_hand / 0.05) * 100.0


def _hand_cat(dp) -> str:
    """Return postflop hand category, or 'preflop' / 'unknown'."""
    if dp.street == 'preflop':
        return 'preflop'
    cat = classify_postflop(dp.hole_cards, dp.community_cards)
    return cat or 'unknown'


def _facing_bet(legal) -> bool:
    return FOLD in legal


# ──────────────────────────────────────────────────────────────────────────────
# Perturbation registry
# Each entry: (name, match_fn(dp, legal) -> bool, apply_fn(strat, dp, legal) -> ndarray)
# The match_fn filters which decisions the perturbation fires on; apply_fn
# returns a re-normalized strategy.
# ──────────────────────────────────────────────────────────────────────────────

def _shift(strat, from_actions, to_action, frac, legal):
    """Move `frac` of total mass from from_actions → to_action.

    If to_action isn't legal, returns original strategy unchanged.
    Re-normalizes to sum 1.0.
    """
    if to_action not in legal:
        return strat
    new = strat.copy()
    moved = 0.0
    for a in from_actions:
        if a < len(new) and a in legal:
            moved += new[a] * frac
            new[a] *= (1.0 - frac)
    if to_action < len(new):
        new[to_action] += moved
    s = new.sum()
    if s > 0:
        new = new / s
    return new


def build_perturbations():
    """Build the perturbation registry — each targets a specific suspected leak.

    Initial seed: target the 'exploitable' signals already showing up in HH
    validator output (made_flush over-aggression, trips over-aggression,
    mid_pair fold-too-low, etc.) and the known caveman-strategy gaps.
    """
    P = []

    # 1) Made flush over-aggression: validator flags +18% aggr vs NL5.
    #    NL5 fish slow-play flushes; betting big folds out worse value. Cool aggr.
    P.append((
        'less_aggr_made_flush',
        lambda dp, legal: _hand_cat(dp) == 'flush',
        lambda s, dp, legal: _shift(s, AGGRESSIVE, CALL, 0.25, legal),
    ))

    # 2) Trips over-aggression: validator flags +15% aggr.
    P.append((
        'less_aggr_trips',
        lambda dp, legal: _hand_cat(dp) == 'trips',
        lambda s, dp, legal: _shift(s, AGGRESSIVE, CALL, 0.25, legal),
    ))

    # 3) Air on river facing a bet: classic "give up the bluff catch".
    #    Should fold more — NL5 doesn't bluff rivers.
    P.append((
        'more_fold_air_river_facing_bet',
        lambda dp, legal: (dp.street == 'river' and _facing_bet(legal)
                            and _hand_cat(dp) in ('high_card', 'air', 'unknown')),
        lambda s, dp, legal: _shift(s, {CALL}, FOLD, 0.40, legal),
    ))

    # 4) Missed flush/straight draw on river facing bet: should fold ~always.
    P.append((
        'more_fold_missed_draw_river',
        lambda dp, legal: (dp.street == 'river' and _facing_bet(legal)
                            and _hand_cat(dp) in ('flush_draw', 'oesd',
                                                   'gutshot', 'high_card',
                                                   'air', 'unknown')),
        lambda s, dp, legal: _shift(s, {CALL}, FOLD, 0.50, legal),
    ))

    # 5) Mid/low pair facing TURN bet: NL5 villains rarely bluff turn.
    #    Validator: "Middle/bottom pair: model fold higher by 32% (GTO-aligned)".
    #    Test if pushing fold even higher helps.
    P.append((
        'more_fold_mid_pair_turn',
        lambda dp, legal: (dp.street == 'turn' and _facing_bet(legal)
                            and _hand_cat(dp) == 'mid_low_pair'),
        lambda s, dp, legal: _shift(s, {CALL}, FOLD, 0.30, legal),
    ))

    # 6) Mid/low pair facing RIVER bet: even stronger fold signal.
    P.append((
        'more_fold_mid_pair_river',
        lambda dp, legal: (dp.street == 'river' and _facing_bet(legal)
                            and _hand_cat(dp) == 'mid_low_pair'),
        lambda s, dp, legal: _shift(s, {CALL}, FOLD, 0.50, legal),
    ))

    # 7) Top pair on flop checked-to: value-bet (NL5 calls down with worse).
    P.append((
        'more_aggr_top_pair_flop_checked',
        lambda dp, legal: (dp.street == 'flop' and not _facing_bet(legal)
                            and _hand_cat(dp) == 'top_pair'),
        lambda s, dp, legal: _shift(s, {CALL}, RAISE_HALF, 0.30, legal),
    ))

    # 8) Overpair facing bet: value-raise (NL5 stations call overpairs down).
    P.append((
        'more_aggr_overpair_facing_bet',
        lambda dp, legal: (_facing_bet(legal) and _hand_cat(dp) == 'overpair'),
        lambda s, dp, legal: _shift(s, {CALL}, RAISE_HALF, 0.25, legal),
    ))

    # 9) Two pair facing bet: lean toward raise for value.
    P.append((
        'more_aggr_two_pair_facing_bet',
        lambda dp, legal: (_facing_bet(legal) and _hand_cat(dp) == 'two_pair'),
        lambda s, dp, legal: _shift(s, {CALL}, RAISE_HALF, 0.25, legal),
    ))

    # 10) Set/full house: slow-play less, raise more (NL5 pays off sets).
    P.append((
        'more_aggr_set_or_better',
        lambda dp, legal: _hand_cat(dp) in ('set', 'monster', 'straight'),
        lambda s, dp, legal: _shift(s, {CALL}, RAISE_HALF, 0.20, legal),
    ))

    # 11) Reverse-direction sanity check: force MORE calls with mid_pair facing
    #     bet. Should produce NEGATIVE delta — confirms the perturbation
    #     framework is wired correctly.
    P.append((
        'SANITY_more_call_mid_pair_facing_bet',
        lambda dp, legal: (_facing_bet(legal)
                            and _hand_cat(dp) == 'mid_low_pair'
                            and dp.street in ('turn', 'river')),
        lambda s, dp, legal: _shift(s, {FOLD}, CALL, 0.40, legal),
    ))

    return P


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap (load model + HH + decision points). Borrowed from validate_vs_hh.main()
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap(model_path: Path, hh_dir: Path, config: dict, max_hands: int,
              use_bh: bool, betting_line: bool, no_cache: bool):
    """Load model + parse HH + build decision points. Returns the bundle needed
    for repeated compute_showdown_ev calls.
    """
    num_players, legacy = vhh._autodetect_players(model_path)
    print(f'Loading model: {model_path}')
    deep_cfr = vhh._load_model(model_path, config, num_players, legacy,
                                model_num_players=None,
                                behavioral_stats=use_bh,
                                betting_line=betting_line)

    print(f'Parsing HH from {hh_dir} (max_hands={max_hands})...')
    all_6p, showdown, total_hands, file_count = vhh._parse_incremental(
        hh_dir, num_players, max_hands, no_cache)
    print(f'  Total hands: {total_hands}, 6p: {len(all_6p)}, showdown: {len(showdown)}')

    print(f'Building decision points from {len(showdown)} showdown hands...')
    builder = ObservationBuilder(num_players=num_players, legacy_game_state=legacy)
    all_decision_points = []
    for hand in showdown:
        dps = builder.build_decision_points(hand, behavioral_stats=use_bh)
        if dps:
            all_decision_points.extend(dps)
    print(f'  {len(all_decision_points)} decision points')

    pf_ranges = PreflopRanges(num_players=num_players)

    sfn = lambda obs, idx, lg: vhh._avg_strategy(deep_cfr, obs, idx, lg)

    return {
        'deep_cfr': deep_cfr,
        'sfn': sfn,
        'showdown': showdown,
        'all_decision_points': all_decision_points,
        'num_players': num_players,
        'pf_ranges': pf_ranges,
        'all_6p_count': len(all_6p),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-bucket counting (separate one-pass scan, not gated by perturbation)
# ──────────────────────────────────────────────────────────────────────────────

def count_buckets(decision_points, perturbations):
    """Pre-count how many DPs each perturbation fires on. Useful for filtering
    out micro-buckets where the delta would be noise.
    """
    counts = defaultdict(int)
    for dp in decision_points:
        legal = dp.legal_actions
        if not legal or dp.street == 'preflop':
            continue
        for name, match_fn, _apply in perturbations:
            try:
                if match_fn(dp, legal):
                    counts[name] += 1
            except Exception:
                pass
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Strategy-perturbation sweep over HH validation to find leaks')
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--hh-dir', type=Path,
                    default=Path('data/poker_archive/'))
    ap.add_argument('--config', type=Path,
                    default=Path('configs/training_config.json'))
    ap.add_argument('--max-hands', type=int, default=50000)
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--behavioral-stats', action='store_true', default=True)
    ap.add_argument('--betting-line', action='store_true', default=True)
    ap.add_argument('--output-dir', type=Path, default=None,
                    help='Defaults to logs/research_weaknesses/<timestamp>/')
    ap.add_argument('--min-bucket-size', type=int, default=100,
                    help='Skip perturbations with bucket size < this')
    ap.add_argument('--fold-equity', type=float, default=0.30)
    ap.add_argument('--continuation-equity', type=float, default=0.50)
    args = ap.parse_args()

    if not args.model.exists():
        sys.exit(f'Model not found: {args.model}')
    if not args.hh_dir.exists():
        sys.exit(f'HH dir not found: {args.hh_dir}')

    output_dir = args.output_dir or Path(
        f'logs/research_weaknesses/{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output: {output_dir}')

    config = load_config(args.config) if args.config.exists() else {}

    bundle = bootstrap(args.model, args.hh_dir, config, args.max_hands,
                       args.behavioral_stats, args.betting_line, args.no_cache)

    perturbations = build_perturbations()
    print(f'\nCounting bucket sizes for {len(perturbations)} perturbations...')
    bucket_counts = count_buckets(bundle['all_decision_points'], perturbations)
    for name, _, _ in perturbations:
        print(f'  {name:<48s} n={bucket_counts[name]}')

    parse_stats = {
        'all_6p_hands': bundle['all_6p_count'],
        'filtered_hands': len(bundle['showdown']),
    }

    # ── Baseline ──
    print('\n[baseline] Running showdown EV (no perturbation)...')
    vhh._STRATEGY_PERTURBATION = None
    t0 = time.time()
    ev_base = vhh.compute_showdown_ev(
        bundle['showdown'], bundle['all_decision_points'],
        bundle['sfn'], bundle['deep_cfr'], bundle['num_players'],
        preflop_ranges=bundle['pf_ranges'],
        p_fold_to_raise=args.fold_equity,
        continuation_equity=args.continuation_equity)
    base_bb100 = _bb_per_100(ev_base)
    print(f'  baseline BB/100: {base_bb100:+.2f}  '
          f'(N={ev_base["total_showdown_players"]}, '
          f'{time.time()-t0:.1f}s)')

    # ── Perturbations ──
    results = []
    for name, match_fn, apply_fn in perturbations:
        n_bucket = bucket_counts[name]
        if n_bucket < args.min_bucket_size:
            print(f'\n[skip] {name}  (bucket too small: n={n_bucket})')
            continue

        def perturb(dp, base_strat, legal,
                    _match=match_fn, _apply=apply_fn):
            try:
                if _match(dp, legal):
                    return _apply(base_strat, dp, legal)
            except Exception:
                pass
            return base_strat

        vhh._STRATEGY_PERTURBATION = perturb
        t0 = time.time()
        ev_p = vhh.compute_showdown_ev(
            bundle['showdown'], bundle['all_decision_points'],
            bundle['sfn'], bundle['deep_cfr'], bundle['num_players'],
            preflop_ranges=bundle['pf_ranges'],
            p_fold_to_raise=args.fold_equity,
            continuation_equity=args.continuation_equity)
        bb100 = _bb_per_100(ev_p)
        delta = bb100 - base_bb100
        print(f'[run]  {name:<48s} n={n_bucket:>5d}  '
              f'BB/100 {bb100:+.2f}  Δ={delta:+.2f}  '
              f'({time.time()-t0:.1f}s)')
        results.append({
            'perturbation': name,
            'bucket_size': n_bucket,
            'bb_per_100': round(bb100, 3),
            'delta_bb_per_100': round(delta, 3),
            'baseline_bb_per_100': round(base_bb100, 3),
            'avoided_losses_sum': round(sum(ev_p['avoided_losses']), 2),
            'missed_gains_sum': round(sum(ev_p['missed_gains']), 2),
            'fold_correct': ev_p['fold_correct'],
            'fold_wrong': ev_p['fold_wrong'],
            'continue_wins': ev_p['continue_wins'],
            'continue_losses': ev_p['continue_losses'],
        })

    # Reset hook
    vhh._STRATEGY_PERTURBATION = None

    # ── Save + summarize ──
    results.sort(key=lambda r: r['delta_bb_per_100'], reverse=True)
    csv_path = output_dir / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        if results:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    print(f'\nResults written: {csv_path}')

    print('\n' + '=' * 80)
    print('RANKED BY Δ BB/100 (descending — positive = perturbation helped)')
    print('=' * 80)
    print(f'  baseline: {base_bb100:+.2f} BB/100')
    print(f'  {"perturbation":<48s} {"n":>6s} {"Δ BB/100":>10s} {"BB/100":>10s}')
    print('  ' + '-' * 78)
    for r in results:
        print(f"  {r['perturbation']:<48s} "
              f"{r['bucket_size']:>6d} "
              f"{r['delta_bb_per_100']:>+10.2f} "
              f"{r['bb_per_100']:>+10.2f}")
    print('=' * 80)

    # Brief interpretation hint
    print('\nInterpretation:')
    print('  Δ > 0  → forcing this shift on the bucket improved HH EV → leak fixable')
    print('           by a shaper / fine-tune that pushes the model in this direction.')
    print('  Δ < 0  → model already plays this bucket correctly (or better than the shift).')
    print('  SANITY entries should produce Δ < 0; if not, the framework is mis-wired.\n')


if __name__ == '__main__':
    main()
