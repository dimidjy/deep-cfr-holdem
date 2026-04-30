#!/usr/bin/env python3
"""Burst-training loop with automatic best-checkpoint selection.

Runs N cycles of `iters_per_cycle` iterations each. Every cycle:
  1. Trains via train_gpu.py for `iters_per_cycle` iterations
  2. Saves checkpoints every `iters_per_cycle / checkpoints_per_cycle` iters
     (e.g. 300 iters / 3 checkpoints → save every 100 iters)
  3. Evaluates each saved checkpoint with validate_spot_logs.py
  4. Picks the highest-scoring checkpoint as the resume point for next cycle
  5. Each cycle's outputs go to a dedicated subfolder

Use this to combat shaper-driven drift: each cycle re-anchors training on
the best-so-far checkpoint, preventing one-sided force from accumulating
past the optimum.

Example:
  python3 scripts/train_burst_loop.py \\
      --start-checkpoint models/best_pool/checkpoint_iter_4200_best.pt \\
      --start-iter 4200 \\
      --num-cycles 5 \\
      --iters-per-cycle 300 \\
      --checkpoints-per-cycle 3 \\
      --output-base models/v19 \\
      --kbest-seed models/best_pool/checkpoint_iter_2800.pt \\
                   models/best_pool/checkpoint_iter_3600.pt \\
                   models/best_pool/checkpoint_iter_3800.pt \\
      --workers 54

Output structure:
  models/v19/
    cycle_1/
      checkpoint_iter_4300.pt + buffer + metadata
      checkpoint_iter_4400.pt + buffer + metadata
      checkpoint_iter_4500.pt + buffer + metadata
      cycle_results.json   (scores + best pick)
    cycle_2/
      checkpoint_iter_4600.pt ... 4800.pt
      cycle_results.json
    ...
    burst_summary.json     (full timeline across cycles)
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_score_from_validate_output(stdout: str) -> float | None:
    """Extract the spot-agreement percentage from validate_spot_logs output.

    Looks for a line like:
      [spots 1597] score=1445.0/1597 (90.5%)  exact=1415  partial=60
    """
    for line in stdout.splitlines():
        m = re.search(r'score=[\d.]+/\d+ \(([\d.]+)%\)', line)
        if m:
            return float(m.group(1))
    return None


def _eval_checkpoint(ckpt_path: Path) -> float | None:
    """Run validate_spot_logs.py on a checkpoint, return % score (or None)."""
    cmd = [
        'python3',
        str(_PROJECT_ROOT / 'scripts/validate_spot_logs.py'),
        '--model', str(ckpt_path),
    ]
    print(f'  Evaluating {ckpt_path.name} (spot logs) ...', flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(_PROJECT_ROOT))
    dt = time.time() - t0
    if result.returncode != 0:
        print(f'    [FAILED in {dt:.1f}s]: {result.stderr.splitlines()[-3:]}')
        return None
    score = _parse_score_from_validate_output(result.stdout)
    if score is None:
        print(f'    [no score parsed in {dt:.1f}s]')
        return None
    print(f'    spot logs = {score:.2f}%  ({dt:.1f}s)')
    return score


# ──────────────────────────────────────────────────────────────────────────────
# Composite scoring: spot logs + HH validation + self-play GTO compliance
# ──────────────────────────────────────────────────────────────────────────────
# GTO bands per CLAUDE.md (project root) — used for self-play compliance score.
# Postflop stats only. Preflop (VPIP/PFR/3Bet) is forced by GTO range override
# in DecisionEngine, so it's always near-target regardless of model quality.
# Including preflop stats would inflate every model's GTO score by ~50%.
GTO_RANGES = {
    'cbet':      (55.0, 70.0),
    'af':        (2.0, 3.5),
    'wtsd':      (25.0, 32.0),
}

# Composite weights — money first, decision quality second, robustness third.
DEFAULT_WEIGHTS = {'hh': 0.50, 'spot': 0.30, 'gto': 0.20}

# HH BB/h normalization anchor: 0 = break-even, +2.0 BB/h showdown = score 100.
# Top current-gen models score ~+1.36 → 68 on this scale.
HH_BBH_CEILING = 2.0


def _eval_hh_validation(ckpt_path: Path, max_hands: int = 10000,
                        behavioral_stats: bool = True,
                        betting_line: bool = True) -> tuple:
    """Run validate_vs_hh.py, parse showdown BB/h and HH agreement.

    Returns (bb_h_showdown, agreement_pct) or (None, None) on failure.
    """
    cmd = [
        'python3',
        str(_PROJECT_ROOT / 'scripts/validate_vs_hh.py'),
        '--model', str(ckpt_path),
        '--max-hands', str(max_hands),
    ]
    if behavioral_stats:
        cmd.append('--behavioral-stats')
    if betting_line:
        cmd.append('--betting-line')
    print(f'  Evaluating {ckpt_path.name} (HH {max_hands}) ...', flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(_PROJECT_ROOT))
    dt = time.time() - t0
    if result.returncode != 0:
        print(f'    [HH FAILED in {dt:.1f}s]: '
              f'{result.stderr.splitlines()[-3:] if result.stderr else "no stderr"}')
        return None, None
    bb_h = None
    agreement = None
    for line in result.stdout.splitlines():
        m = re.search(r'Showdown-only edge:\s+([+-]?[\d.]+)\s+BB/hand', line)
        if m and bb_h is None:
            bb_h = float(m.group(1))
        m = re.search(r'agreement=([\d.]+)%', line)
        if m and agreement is None:
            agreement = float(m.group(1))
    if bb_h is None:
        print(f'    [HH no BB/h parsed in {dt:.1f}s]')
        return None, None
    print(f'    HH BB/h = {bb_h:+.2f}  agreement = '
          f'{agreement:.1f}%  ({dt:.1f}s)' if agreement is not None
          else f'    HH BB/h = {bb_h:+.2f}  ({dt:.1f}s)')
    return bb_h, agreement


def _eval_self_play(ckpt_path: Path, hands: int = 10000) -> dict | None:
    """Run self_play.py, return dict of GTO-relevant stats (or None on fail)."""
    cmd = [
        'python3',
        str(_PROJECT_ROOT / 'scripts/analysis/self_play.py'),
        '--model', str(ckpt_path),
        '--hands', str(hands),
        '--no-guards',
    ]
    print(f'  Evaluating {ckpt_path.name} (self-play {hands}) ...', flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(_PROJECT_ROOT))
    dt = time.time() - t0
    if result.returncode != 0:
        print(f'    [self-play FAILED in {dt:.1f}s]')
        return None
    patterns = {
        'vpip':      r'^\s+VPIP:\s+([\d.]+)%',
        'pfr':       r'^\s+PFR:\s+([\d.]+)%',
        'three_bet': r'^\s+3Bet:\s+([\d.]+)%',
        'cbet':      r'^\s+CBet:\s+([\d.]+)%',
        'af':        r'^\s+AF \(agg factor\):\s+([\d.]+)',
        'wtsd':      r'^\s+WTSD \(went to SD\):\s+([\d.]+)%',
    }
    stats = {}
    for line in result.stdout.splitlines():
        for key, pat in patterns.items():
            if key in stats:
                continue
            m = re.match(pat, line)
            if m:
                stats[key] = float(m.group(1))
    if len(stats) < 5:
        print(f'    [self-play parsed only {len(stats)}/6 stats in {dt:.1f}s]')
        return None
    in_band = sum(1 for k, (lo, hi) in GTO_RANGES.items()
                  if k in stats and lo <= stats[k] <= hi)
    print(f'    self-play GTO in-band: {in_band}/{len(GTO_RANGES)}  '
          f'(CBet={stats.get("cbet", 0):.1f} AF={stats.get("af", 0):.2f} '
          f'WTSD={stats.get("wtsd", 0):.1f})  ({dt:.1f}s)')
    return stats


def _gto_score(self_play_stats: dict | None) -> float:
    """0-100 score: fraction of GTO stats in band."""
    if not self_play_stats:
        return 0.0
    in_band = sum(1 for k, (lo, hi) in GTO_RANGES.items()
                  if k in self_play_stats and lo <= self_play_stats[k] <= hi)
    return in_band / len(GTO_RANGES) * 100.0


def _hh_score(bb_h_showdown: float | None) -> float:
    """0-100 score: HH BB/h normalized.
    0   = break-even (0 BB/h)
    100 = +HH_BBH_CEILING BB/h (currently 2.0 = aspirational)
    """
    if bb_h_showdown is None:
        return 0.0
    return max(0.0, min(100.0, bb_h_showdown / HH_BBH_CEILING * 100.0))


def _composite_score(spot_pct: float | None,
                     bb_h_showdown: float | None,
                     self_play_stats: dict | None,
                     weights: dict = DEFAULT_WEIGHTS) -> float:
    """Weighted blend of HH (50%), spot logs (30%), GTO compliance (20%)."""
    spot = spot_pct if spot_pct is not None else 0.0
    hh   = _hh_score(bb_h_showdown)
    gto  = _gto_score(self_play_stats)
    return weights['hh'] * hh + weights['spot'] * spot + weights['gto'] * gto


def _find_cycle_checkpoints(cycle_dir: Path,
                            min_iter: int,
                            max_iter: int) -> list[Path]:
    """Return checkpoints in cycle_dir whose iter is in (min_iter, max_iter]."""
    checkpoints = []
    for p in sorted(cycle_dir.glob('checkpoint_iter_*.pt')):
        m = re.match(r'checkpoint_iter_(\d+)\.pt$', p.name)
        if not m:
            continue
        it = int(m.group(1))
        if min_iter < it <= max_iter:
            checkpoints.append(p)
    return checkpoints


def _run_cycle(*,
               cycle_idx: int,
               start_iter: int,
               end_iter: int,
               iters_per_cycle: int,
               checkpoint_interval: int,
               resume_path: Path,
               cycle_dir: Path,
               config_path: Path,
               kbest_seed: list[str],
               workers: int,
               selector: str = 'composite',
               hh_hands: int = 10000,
               self_play_hands: int = 10000,
               composite_weights: dict = None) -> dict:
    """Run one training cycle and return cycle stats."""
    cycle_dir.mkdir(parents=True, exist_ok=True)
    print()
    print('=' * 70)
    print(f'CYCLE {cycle_idx}: iter {start_iter} → {end_iter}'
          f'  ({iters_per_cycle} iters, checkpoints every {checkpoint_interval})')
    print(f'  resume:  {resume_path}')
    print(f'  output:  {cycle_dir}')
    print('=' * 70)

    cmd = [
        'python3',
        str(_PROJECT_ROOT / 'scripts/train_gpu.py'),
        '--config', str(config_path),
        '--resume', str(resume_path),
        '--num-iterations', str(end_iter),
        '--checkpoint-interval', str(checkpoint_interval),
        # Save buffer at every checkpoint so any selected "best" can be
        # resumed from in the next cycle without warm-up cost.
        '--buffer-save-interval', str(checkpoint_interval),
        '--models-dir', str(cycle_dir),
        '--workers', str(workers),
        '--kbest-update-interval', '0',
    ]
    if kbest_seed:
        cmd.append('--kbest-seed')
        cmd.extend(kbest_seed)

    print(f'  command: {" ".join(cmd)}\n', flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    dt = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f'Cycle {cycle_idx} training failed (exit {result.returncode})')
    print(f'\n  Cycle {cycle_idx} training completed in {dt/60:.1f} min')

    # Find checkpoints saved during this cycle
    new_checkpoints = _find_cycle_checkpoints(cycle_dir, start_iter, end_iter)
    print(f'  Saved {len(new_checkpoints)} checkpoints: '
          f'{[p.name for p in new_checkpoints]}')

    # Evaluate each — spot logs always; HH and self-play if composite mode
    print(f'\n  Evaluating cycle checkpoints (selector={selector})...')
    weights = composite_weights or DEFAULT_WEIGHTS
    metrics: dict[str, dict] = {}
    scores: dict[str, float] = {}  # primary score used for selection
    for cp in new_checkpoints:
        print(f'\n  --- {cp.name} ---')
        spot_pct = _eval_checkpoint(cp)

        bb_h = None
        agreement = None
        sp_stats = None
        if selector == 'composite':
            bb_h, agreement = _eval_hh_validation(cp, max_hands=hh_hands)
            sp_stats = _eval_self_play(cp, hands=self_play_hands)

        if selector == 'composite':
            primary = _composite_score(spot_pct, bb_h, sp_stats, weights=weights)
        else:
            primary = spot_pct if spot_pct is not None else 0.0

        metrics[cp.name] = {
            'spot_pct': spot_pct,
            'hh_bb_h_showdown': bb_h,
            'hh_agreement_pct': agreement,
            'self_play': sp_stats,
            'gto_score': _gto_score(sp_stats),
            'hh_score': _hh_score(bb_h),
            'composite': _composite_score(spot_pct, bb_h, sp_stats, weights=weights),
            'selector_score': primary,
        }
        scores[cp.name] = primary

        print(f'    --> primary score = {primary:.2f}  '
              f'(spot={spot_pct or 0:.1f}  hh={_hh_score(bb_h):.1f}  '
              f'gto={_gto_score(sp_stats):.1f})')

    if not scores:
        raise RuntimeError(f'Cycle {cycle_idx} produced no evaluable checkpoints')

    # Pick winner by primary (composite or spot-only)
    best_name = max(scores, key=lambda k: scores[k])
    best_path = cycle_dir / best_name
    best_metrics = metrics[best_name]
    print(f'\n  Best in cycle {cycle_idx}: {best_name}  '
          f'(primary={scores[best_name]:.2f})')
    print(f'    spot logs       : {best_metrics["spot_pct"]:.1f}%'
          if best_metrics["spot_pct"] is not None else '    spot logs       : <failed>')
    if best_metrics["hh_bb_h_showdown"] is not None:
        print(f'    HH BB/h         : {best_metrics["hh_bb_h_showdown"]:+.2f}'
              f'  (score {best_metrics["hh_score"]:.1f})')
        if best_metrics["hh_agreement_pct"] is not None:
            print(f'    HH agreement    : {best_metrics["hh_agreement_pct"]:.1f}%')
    print(f'    GTO in band     : {int(best_metrics["gto_score"] / 100 * len(GTO_RANGES))}'
          f'/{len(GTO_RANGES)}  (score {best_metrics["gto_score"]:.1f})')
    print(f'    composite       : {best_metrics["composite"]:.2f}')

    cycle_summary = {
        'cycle': cycle_idx,
        'start_iter': start_iter,
        'end_iter': end_iter,
        'resume_from': str(resume_path),
        'output_dir': str(cycle_dir),
        'duration_minutes': dt / 60,
        'selector': selector,
        'composite_weights': weights,
        'checkpoints': sorted(scores.keys()),
        'metrics': metrics,
        # Backward-compat fields
        'scores_pct': {k: v.get('spot_pct') for k, v in metrics.items()},
        'best_checkpoint': best_name,
        'best_score_pct': metrics[best_name].get('spot_pct'),
        'best_composite': metrics[best_name].get('composite'),
        'best_path': str(best_path),
    }
    with open(cycle_dir / 'cycle_results.json', 'w') as f:
        json.dump(cycle_summary, f, indent=2)
    return cycle_summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--start-checkpoint', required=True, type=Path,
                   help='Initial resume checkpoint (with matching _buffer.npz alongside)')
    p.add_argument('--start-iter', required=True, type=int,
                   help='Iteration counter that the start checkpoint represents')
    p.add_argument('--num-cycles', type=int, default=5,
                   help='How many burst cycles to run (default: 5)')
    p.add_argument('--iters-per-cycle', type=int, default=300,
                   help='Iterations per cycle (default: 300)')
    p.add_argument('--checkpoints-per-cycle', type=int, default=3,
                   help='Checkpoints saved per cycle (default: 3)')
    p.add_argument('--output-base', type=Path, default=Path('models/v19'),
                   help='Base directory under which cycle_N/ folders are created')
    p.add_argument('--config', type=Path,
                   default=Path('configs/training_config.json'),
                   help='Training config path')
    p.add_argument('--kbest-seed', nargs='+', default=None,
                   help='Checkpoints for K-Best opponent pool (frozen across cycles)')
    p.add_argument('--workers', type=int, default=54,
                   help='Worker count passed to train_gpu.py')
    p.add_argument('--selector', choices=['composite', 'spot'], default='composite',
                   help='Checkpoint selector. composite (default) blends spot '
                        'logs + HH validation BB/h + self-play GTO compliance. '
                        '"spot" reverts to legacy spot-log-only selection.')
    p.add_argument('--hh-hands', type=int, default=10000,
                   help='Hand count for HH validation per checkpoint (composite '
                        'mode only). 10k = ~1-2 min per ckpt; 50k for higher '
                        'fidelity at ~5-10 min.')
    p.add_argument('--self-play-hands', type=int, default=10000,
                   help='Hand count for self-play GTO check per checkpoint.')
    p.add_argument('--weight-hh', type=float, default=DEFAULT_WEIGHTS['hh'],
                   help=f'Composite weight for HH score (default {DEFAULT_WEIGHTS["hh"]}).')
    p.add_argument('--weight-spot', type=float, default=DEFAULT_WEIGHTS['spot'],
                   help=f'Composite weight for spot-log score '
                        f'(default {DEFAULT_WEIGHTS["spot"]}).')
    p.add_argument('--weight-gto', type=float, default=DEFAULT_WEIGHTS['gto'],
                   help=f'Composite weight for GTO compliance '
                        f'(default {DEFAULT_WEIGHTS["gto"]}).')
    args = p.parse_args()
    composite_weights = {
        'hh':   args.weight_hh,
        'spot': args.weight_spot,
        'gto':  args.weight_gto,
    }
    w_sum = sum(composite_weights.values())
    if abs(w_sum - 1.0) > 0.001:
        print(f'WARNING: composite weights sum to {w_sum:.3f}, not 1.0. '
              f'Using anyway — scores will be off-scale.')

    if args.iters_per_cycle % args.checkpoints_per_cycle != 0:
        print(f'WARNING: iters_per_cycle ({args.iters_per_cycle}) is not '
              f'evenly divisible by checkpoints_per_cycle '
              f'({args.checkpoints_per_cycle}). '
              f'Some cycles may save extra/missing checkpoints.')

    checkpoint_interval = args.iters_per_cycle // args.checkpoints_per_cycle

    if not args.start_checkpoint.exists():
        print(f'ERROR: start checkpoint does not exist: {args.start_checkpoint}')
        sys.exit(1)
    buffer_path = args.start_checkpoint.with_name(
        args.start_checkpoint.stem + '_buffer.npz')
    if not buffer_path.exists():
        print(f'WARNING: no buffer at {buffer_path} — training will start with empty buffer')

    args.output_base.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_base / 'burst_summary.json'

    print('=' * 70)
    print('BURST-TRAINING LOOP')
    print('=' * 70)
    print(f'  Start checkpoint   : {args.start_checkpoint}')
    print(f'  Start iteration    : {args.start_iter}')
    print(f'  Cycles             : {args.num_cycles}')
    print(f'  Iters per cycle    : {args.iters_per_cycle}')
    print(f'  Checkpoints/cycle  : {args.checkpoints_per_cycle} '
          f'(every {checkpoint_interval} iters)')
    print(f'  Total iterations   : {args.num_cycles * args.iters_per_cycle} '
          f'(end at iter {args.start_iter + args.num_cycles * args.iters_per_cycle})')
    print(f'  Output base        : {args.output_base}')
    print(f'  Config             : {args.config}')
    if args.kbest_seed:
        print(f'  K-Best pool        : {[Path(p).name for p in args.kbest_seed]}')
    print(f'  Selector           : {args.selector}')
    if args.selector == 'composite':
        print(f'    HH hands         : {args.hh_hands}')
        print(f'    Self-play hands  : {args.self_play_hands}')
        print(f'    Weights (hh/spot/gto) : '
              f'{args.weight_hh} / {args.weight_spot} / {args.weight_gto}')
    print('=' * 70, flush=True)

    timeline = []
    current_resume = args.start_checkpoint
    current_iter = args.start_iter

    for cycle_idx in range(1, args.num_cycles + 1):
        cycle_dir = args.output_base / f'cycle_{cycle_idx}'
        end_iter = current_iter + args.iters_per_cycle

        cycle_summary = _run_cycle(
            cycle_idx=cycle_idx,
            start_iter=current_iter,
            end_iter=end_iter,
            iters_per_cycle=args.iters_per_cycle,
            checkpoint_interval=checkpoint_interval,
            resume_path=current_resume,
            cycle_dir=cycle_dir,
            config_path=args.config,
            kbest_seed=args.kbest_seed,
            workers=args.workers,
            selector=args.selector,
            hh_hands=args.hh_hands,
            self_play_hands=args.self_play_hands,
            composite_weights=composite_weights,
        )
        timeline.append(cycle_summary)

        # Write cumulative summary after each cycle so a crash doesn't lose state
        with open(summary_path, 'w') as f:
            json.dump({
                'start_checkpoint': str(args.start_checkpoint),
                'start_iter': args.start_iter,
                'num_cycles_planned': args.num_cycles,
                'cycles_completed': len(timeline),
                'iters_per_cycle': args.iters_per_cycle,
                'checkpoints_per_cycle': args.checkpoints_per_cycle,
                'kbest_seed': args.kbest_seed,
                'cycles': timeline,
            }, f, indent=2)

        # Set up next cycle: resume from this cycle's best
        current_resume = Path(cycle_summary['best_path'])
        current_iter = end_iter

        # Print running summary
        print(f'\nCumulative best scores by cycle:')
        for c in timeline:
            print(f'  cycle {c["cycle"]}: {c["best_checkpoint"]} '
                  f'→ {c["best_score_pct"]:.2f}%')

    print()
    print('=' * 70)
    print('BURST-TRAINING LOOP COMPLETE')
    print('=' * 70)
    print(f'Summary written to: {summary_path}')
    overall_best = max(timeline, key=lambda c: c['best_score_pct'])
    print(f"Overall best: {overall_best['best_checkpoint']} "
          f"(cycle {overall_best['cycle']}, "
          f"score {overall_best['best_score_pct']:.2f}%)")
    print(f'Path: {overall_best["best_path"]}')


if __name__ == '__main__':
    main()
