"""
Automated fold-ratio search for supervised poker model.

Parses hand history data ONCE, then trains models at different fold-teacher
ratios (e.g. 10%, 20%, ..., 90% fold teachers). Each model is validated
automatically using spot validation. Results are compared in a summary table
to find the optimal balance between good-player imitation and fold learning.

Usage:
  python3 scripts/search_fold_ratio.py \
    --good-players data/test_good_players.txt \
    --bad-players data/test_bad_players.txt \
    --ratios 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9

  # Quick test with fewer epochs and games
  python3 scripts/search_fold_ratio.py \
    --good-players data/test_good_players.txt \
    --bad-players data/test_bad_players.txt \
    --epochs 20 --val-games 300
"""
import sys
import os
import warnings
import logging

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import time
from pathlib import Path

from train_supervised import (
    collect_decision_points,
    train_with_ratio,
)
from validate_spots import validation_summary_line


def main():
    parser = argparse.ArgumentParser(
        description='Search for optimal fold-teacher ratio in supervised training'
    )
    parser.add_argument('--hh-dir', type=str, default='data/poker_archive/',
                        help='Directory with PokerStars HH files')
    parser.add_argument('--good-players', type=str, required=True,
                        help='Path to .txt file with winning player names')
    parser.add_argument('--bad-players', type=str, required=True,
                        help='Path to .txt file with losing player names')
    parser.add_argument('--max-hands', type=int, default=0,
                        help='Max showdown hands to use (0=all)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force re-parse HH files')

    # Search parameters
    parser.add_argument('--ratios', type=str,
                        default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
                        help='Comma-separated fold ratios to test (default: 0.0-0.9 by 0.1)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Epochs per ratio (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32768,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden-layers', type=str, default='512,512,256',
                        help='Hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fold-weight', type=float, default=0.15,
                        help='Loss weight for fold-teacher samples')

    # Validation
    parser.add_argument('--val-games', type=int, default=500,
                        help='Games for spot validation per ratio')

    # Output
    parser.add_argument('--output-dir', type=str, default='models/ratio_search/',
                        help='Directory for ratio search checkpoints')
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    ratios = [float(r) for r in args.ratios.split(',')]
    hidden_layers = [int(x) for x in args.hidden_layers.split(',')]

    # Load player lists
    good_players = set(
        line.strip() for line in Path(args.good_players).read_text().splitlines()
        if line.strip() and not line.strip().startswith('#')
    )
    bad_players = set(
        line.strip() for line in Path(args.bad_players).read_text().splitlines()
        if line.strip() and not line.strip().startswith('#')
    )

    num_players = 6

    print('=' * 70)
    print('  FOLD RATIO SEARCH')
    print('=' * 70)
    print(f'  Good players:  {len(good_players)}')
    print(f'  Bad players:   {len(bad_players)}')
    print(f'  Ratios to test: {ratios}')
    print(f'  Epochs/ratio:  {args.epochs}')
    print(f'  Val games:     {args.val_games}')
    print('=' * 70)

    # ── Step 1: Parse data ONCE ──────────────────────────────────────────
    print(f'\n  STEP 1: Collecting decision points (one-time)...\n')
    t0 = time.time()

    all_dps = collect_decision_points(
        args.hh_dir, num_players, args.max_hands,
        good_players, bad_players, args.no_cache,
        postflop_only=True,
    )

    n_good = sum(1 for dp in all_dps if not getattr(dp, '_is_fold_teacher', False))
    n_fold = sum(1 for dp in all_dps if getattr(dp, '_is_fold_teacher', False))
    print(f'\n  Data collected in {time.time() - t0:.1f}s')
    print(f'  Good player samples: {n_good}')
    print(f'  Fold teacher samples: {n_fold}')
    print(f'  Natural ratio: {n_fold / max(n_good + n_fold, 1):.1%} fold teachers')

    if len(all_dps) < 100:
        print('Not enough data!')
        sys.exit(1)

    # ── Step 2: Train + validate each ratio ──────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for i, ratio in enumerate(ratios):
        print(f'\n{"="*70}')
        print(f'  RATIO {i+1}/{len(ratios)}: {ratio:.0%} fold teachers, {1-ratio:.0%} good')
        print(f'{"="*70}')

        model_path = output_dir / f'ratio_{ratio:.2f}.pt'
        t1 = time.time()

        result = train_with_ratio(
            decision_points=all_dps,
            fold_ratio=ratio,
            output_path=str(model_path),
            num_players=num_players,
            hidden_layers=hidden_layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            fold_weight=args.fold_weight,
            val_games=args.val_games,
            device_str=args.device,
            use_preflop_ranges=True,
        )

        elapsed = time.time() - t1

        if result is None:
            print(f'  SKIPPED (not enough data)')
            continue

        results[ratio] = result
        vr = result['val_results']
        summary = vr.get('_summary', {})
        passed = summary.get('passed', 0)
        total = summary.get('total', 0)

        print(f'  Done in {elapsed:.0f}s — epoch {result["best_epoch"]}, '
              f'val_loss={result["best_val_loss"]:.4f}, val_acc={result["best_val_acc"]:.1%}')
        print(f'  Spots: {passed}/{total} passed')
        print(f'  {validation_summary_line(vr)}')

    # ── Step 3: Summary table ────────────────────────────────────────────
    print(f'\n\n{"="*70}')
    print(f'  RESULTS SUMMARY')
    print(f'{"="*70}')
    print(f'{"Ratio":>8} {"Good":>7} {"Fold":>7} {"Total":>7} '
          f'{"Epoch":>6} {"ValLoss":>8} {"ValAcc":>8} {"Spots":>8} {"Details"}')
    print('-' * 100)

    best_ratio = None
    best_passed = -1
    best_loss = float('inf')

    for ratio in sorted(results.keys()):
        r = results[ratio]
        vr = r['val_results']
        summary = vr.get('_summary', {})
        passed = summary.get('passed', 0)
        total = summary.get('total', 0)

        # Pick best: most spots passed, then lowest val_loss as tiebreaker
        if passed > best_passed or (passed == best_passed and r['best_val_loss'] < best_loss):
            best_passed = passed
            best_loss = r['best_val_loss']
            best_ratio = ratio

        # Collect per-spot fold rates for the detail column
        spot_details = []
        for key, data in vr.items():
            if key.startswith('_') or not isinstance(data, dict):
                continue
            if data.get('n', 0) >= 5:
                tag = 'ok' if data.get('passed') else 'BAD'
                spot_details.append(f"{key}:f={data['fold_pct']:.0%}({tag})")

        detail_str = ' '.join(spot_details[:6])  # first 6 spots

        marker = ' <-- BEST' if ratio == best_ratio else ''
        print(f'{ratio:>7.0%} {r["n_good"]:>7} {r["n_fold"]:>7} {r["n_total"]:>7} '
              f'{r["best_epoch"]:>6} {r["best_val_loss"]:>8.4f} {r["best_val_acc"]:>7.1%} '
              f'{passed:>3}/{total:<4}{marker}')
        print(f'         {detail_str}')

    print('-' * 100)

    if best_ratio is not None:
        print(f'\n  BEST RATIO: {best_ratio:.0%} fold teachers / {1-best_ratio:.0%} good players')
        print(f'  Spots passed: {best_passed}')
        print(f'  Model: {output_dir / f"ratio_{best_ratio:.2f}.pt"}')

        # Copy best model to a convenient name
        import shutil
        best_src = output_dir / f'ratio_{best_ratio:.2f}.pt'
        best_dst = output_dir / 'best_ratio_model.pt'
        shutil.copy2(str(best_src), str(best_dst))
        print(f'  Copied to: {best_dst}')

    print(f'\n{"="*70}')


if __name__ == '__main__':
    main()
