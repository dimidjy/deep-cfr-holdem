#!/usr/bin/env python3
"""
Unified analysis pipeline runner.

Parses HH files ONCE (with multiprocessing), caches the result, then runs
one or more analysis modules on the shared data with shared models.

Usage:
  # Build the database only (parsing step):
  python3 scripts/analysis/run_analysis.py --build-db --max-hands 20000 --workers 40

  # Run all analyses (loads from cache if available):
  python3 scripts/analysis/run_analysis.py --analyze all

  # Run a single analysis:
  python3 scripts/analysis/run_analysis.py --analyze benchmark
  python3 scripts/analysis/run_analysis.py --analyze per_spot_ev
  python3 scripts/analysis/run_analysis.py --analyze full_hand_replay
  python3 scripts/analysis/run_analysis.py --analyze population_stats

  # Build DB and run analyses in one shot:
  python3 scripts/analysis/run_analysis.py --build-db --analyze all --max-hands 20000 --workers 40

  # Override model paths:
  python3 scripts/analysis/run_analysis.py --analyze benchmark \\
      --model-facing-bet models/best/cql_6p_facing_bet_v7.pt \\
      --model-checked-to models/best/cql_6p_checked_to_v3.pt \\
      --model-facing-bet-3bet models/cql_6p_facing_bet_3bet.pt
"""

import argparse
import json
import os
import sys
import time
import warnings
import logging
from datetime import datetime
from pathlib import Path

# ── Project setup (BEFORE any torch/CUDA imports) ──────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Suppress noisy warnings during parsing
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)


# ── Default model paths ────────────────────────────────────────────────────
DEFAULT_MODEL_FB = 'models/best/cql_6p_facing_bet_v7.pt'
DEFAULT_MODEL_CT = 'models/best/cql_6p_checked_to_v3.pt'
DEFAULT_MODEL_FB_3BET = 'models/cql_6p_facing_bet_3bet.pt'
DEFAULT_HH_DIR = 'data/poker_archive/'
DEFAULT_GOOD_PLAYERS = 'data/good_players.txt'
DEFAULT_BAD_PLAYERS = 'data/bad_players.txt'

VALID_ANALYSES = ['all', 'benchmark', 'per_spot_ev', 'full_hand_replay', 'population_stats']


def _resolve_path(p: str) -> str:
    """Resolve a path relative to project root if not absolute."""
    if os.path.isabs(p):
        return p
    return str(_PROJECT_ROOT / p)


def _ensure_results_dir() -> Path:
    """Create and return the results directory."""
    results_dir = _SCRIPT_DIR / 'results'
    results_dir.mkdir(exist_ok=True)
    return results_dir


def main():
    parser = argparse.ArgumentParser(
        description='Unified poker analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    parser.add_argument('--build-db', action='store_true',
                        help='Build/rebuild the analysis database (parse HH files)')
    parser.add_argument('--analyze', type=str, default=None,
                        choices=VALID_ANALYSES,
                        help='Run analysis: all, benchmark, per_spot_ev, '
                             'full_hand_replay, population_stats')

    # Data options
    parser.add_argument('--hh-dir', type=str, default=DEFAULT_HH_DIR,
                        help=f'HH directory (default: {DEFAULT_HH_DIR})')
    parser.add_argument('--max-hands', type=int, default=20000,
                        help='Max showdown hands to parse (default: 20000)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Parser workers (0 = auto)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force re-parse, ignore disk cache')

    # Model options
    parser.add_argument('--model-facing-bet', type=str, default=DEFAULT_MODEL_FB,
                        help=f'Facing-bet model (default: {DEFAULT_MODEL_FB})')
    parser.add_argument('--model-checked-to', type=str, default=DEFAULT_MODEL_CT,
                        help=f'Checked-to model (default: {DEFAULT_MODEL_CT})')
    parser.add_argument('--model-facing-bet-3bet', type=str, default=DEFAULT_MODEL_FB_3BET,
                        help=f'3bet facing-bet model (default: {DEFAULT_MODEL_FB_3BET})')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for model inference (default: cpu)')

    # Player lists
    parser.add_argument('--good-players', type=str, default=DEFAULT_GOOD_PLAYERS,
                        help=f'Good players file (default: {DEFAULT_GOOD_PLAYERS})')
    parser.add_argument('--bad-players', type=str, default=DEFAULT_BAD_PLAYERS,
                        help=f'Bad players file (default: {DEFAULT_BAD_PLAYERS})')

    # Analysis-specific
    parser.add_argument('--min-sample', type=int, default=20,
                        help='Min DPs per tier for benchmark (default: 20)')
    parser.add_argument('--min-player-hands', type=int, default=100,
                        help='Min hands per player for population stats (default: 100)')

    args = parser.parse_args()

    if not args.build_db and not args.analyze:
        parser.error('Specify --build-db, --analyze, or both')

    t_start = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Step 1: Build/load analysis database ────────────────────────────
    # IMPORTANT: This happens BEFORE any torch/CUDA import to avoid
    # mp.Pool deadlocks with CUDA-initialized forked processes.

    from scripts.analysis.analysis_db import (
        build_analysis_db, load_player_names, AnalysisDB,
    )

    need_population = (args.analyze in ('all', 'population_stats'))

    print('=' * 72)
    print(f'  ANALYSIS PIPELINE  [{timestamp}]')
    print('=' * 72)

    print(f'\n[1/3] Building analysis database...')
    db = build_analysis_db(
        hh_dir=args.hh_dir,
        max_hands=args.max_hands,
        workers=args.workers,
        no_cache=args.no_cache,
        include_all_hands=need_population,
    )

    print(f'  Showdown hands: {db.total_showdown_hands}')
    print(f'  Decision points: {db.total_decision_points}')
    if db.all_hands:
        print(f'  All hands (incl non-showdown): {len(db.all_hands)}')

    if args.build_db and not args.analyze:
        print(f'\nDatabase built in {time.time()-t_start:.1f}s. '
              f'Use --analyze to run analyses.')
        return

    if not args.analyze:
        return

    # ── Step 2: Load models ─────────────────────────────────────────────
    # NOW safe to import torch (mp.Pool is done)

    print(f'\n[2/3] Loading models...')

    fb_path = _resolve_path(args.model_facing_bet)
    ct_path = _resolve_path(args.model_checked_to)
    fb3_path = _resolve_path(args.model_facing_bet_3bet)

    # Check model files exist
    if not os.path.exists(fb_path):
        print(f'ERROR: Facing-bet model not found: {fb_path}')
        return
    if not os.path.exists(ct_path):
        print(f'ERROR: Checked-to model not found: {ct_path}')
        return
    if not os.path.exists(fb3_path):
        print(f'WARNING: 3bet model not found: {fb3_path}, skipping 3bet routing')
        fb3_path = None

    from scripts.analysis.model_wrapper import ModelWrapper
    model = ModelWrapper(
        model_fb_path=fb_path,
        model_ct_path=ct_path,
        model_fb_3bet_path=fb3_path,
        device=args.device,
    )

    # Load player lists
    good_players = load_player_names(args.good_players)
    bad_players = load_player_names(args.bad_players)
    print(f'  Good players: {len(good_players)}, Bad players: {len(bad_players)}')

    # ── Step 3: Run analyses ────────────────────────────────────────────
    print(f'\n[3/3] Running analyses...')

    analyses_to_run = []
    if args.analyze == 'all':
        analyses_to_run = ['benchmark', 'per_spot_ev']
        # full_hand_replay and population_stats disabled by default
        # (inflated EV numbers / broken with showdown-only data)
        # Run explicitly: --analyze full_hand_replay  or  --analyze population_stats
    else:
        analyses_to_run = [args.analyze]

    results_dir = _ensure_results_dir()
    all_results = {}
    all_results_raw = {}

    for analysis_name in analyses_to_run:
        print(f'\n{"="*72}')
        print(f'  Running: {analysis_name}')
        print(f'{"="*72}')
        t_analysis = time.time()

        try:
            if analysis_name == 'benchmark':
                from scripts.analysis.analyzers.benchmark_vs_winners import analyze, print_report
                results = analyze(
                    db=db,
                    model=model,
                    good_players=good_players,
                    min_sample=args.min_sample,
                )
                print_report(results)

            elif analysis_name == 'per_spot_ev':
                from scripts.analysis.analyzers.per_spot_ev import analyze, print_report
                results = analyze(db=db, model=model)
                print_report(results)

            elif analysis_name == 'full_hand_replay':
                from scripts.analysis.analyzers.full_hand_replay import analyze, print_report
                results = analyze(db=db, model=model)
                print_report(results)

            elif analysis_name == 'population_stats':
                from scripts.analysis.analyzers.population_stats import analyze, print_report
                results = analyze(
                    db=db,
                    model=model,
                    good_players=good_players,
                    bad_players=bad_players,
                    min_player_hands=args.min_player_hands,
                    max_model_hands=args.max_hands,
                )
                print_report(results)

            else:
                print(f'  Unknown analysis: {analysis_name}')
                continue

            elapsed = time.time() - t_analysis
            print(f'\n  [{analysis_name}] completed in {elapsed:.1f}s')

            # Store raw results for text report, serialized for JSON
            all_results_raw[analysis_name] = results
            all_results[analysis_name] = _make_serializable(results)

        except Exception as e:
            print(f'\n  ERROR in {analysis_name}: {e}')
            import traceback
            traceback.print_exc()
            all_results[analysis_name] = {'error': str(e)}

    # ── Save results ────────────────────────────────────────────────────
    results_file = results_dir / f'analysis_{timestamp}.json'
    txt_file = results_dir / f'analysis_{timestamp}.txt'
    try:
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
    except Exception as e:
        print(f'\nWARNING: Could not save results JSON: {e}')

    # Generate human-readable text report from results
    try:
        report = _generate_text_report(all_results_raw, timestamp)
        with open(txt_file, 'w') as f:
            f.write(report)
        print(f'\nResults saved to:')
        print(f'  {results_file}')
        print(f'  {txt_file}')
    except Exception as e:
        print(f'\nWARNING: Could not save text report: {e}')

    total_time = time.time() - t_start
    print(f'\n{"="*72}')
    print(f'  PIPELINE COMPLETE  [{total_time:.1f}s total]')
    print(f'{"="*72}')


def _generate_text_report(all_results: dict, timestamp: str) -> str:
    """Generate human-readable text report by calling each analyzer's print_report
    with output captured to string."""
    import io
    import sys

    lines = []
    lines.append(f'POKER MODEL ANALYSIS REPORT — {timestamp}')
    lines.append('=' * 72)
    lines.append('')

    for name, results in all_results.items():
        if isinstance(results, dict) and 'error' in results:
            lines.append(f'[{name}] ERROR: {results["error"]}')
            lines.append('')
            continue

        # Re-run print_report to capture its output
        try:
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf

            if name == 'benchmark':
                from scripts.analysis.analyzers.benchmark_vs_winners import print_report
            elif name == 'per_spot_ev':
                from scripts.analysis.analyzers.per_spot_ev import print_report
            elif name == 'full_hand_replay':
                from scripts.analysis.analyzers.full_hand_replay import print_report
            elif name == 'population_stats':
                from scripts.analysis.analyzers.population_stats import print_report
            else:
                sys.stdout = old_stdout
                continue

            print_report(results)
            sys.stdout = old_stdout
            lines.append(buf.getvalue())

        except Exception as e:
            sys.stdout = old_stdout
            lines.append(f'[{name}] Report generation failed: {e}')
            lines.append('')

    lines.append('=' * 72)
    lines.append('END OF REPORT')
    return '\n'.join(lines)


def _make_serializable(obj):
    """Convert results dict to JSON-serializable form (strip numpy, large lists)."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Skip large internal data
            if k in ('all_results', 'records', 'grouped', 'player_bb100',
                      'good_in_data', 'model_strategy'):
                continue
            result[k] = _make_serializable(v)
        return result
    elif isinstance(obj, (list, tuple)):
        if len(obj) > 100:
            return f'[{len(obj)} items]'
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)[:50]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy array
        arr = obj.tolist()
        if isinstance(arr, list) and len(arr) > 20:
            return f'[array of {len(arr)}]'
        return arr
    else:
        return str(obj)


if __name__ == '__main__':
    main()
