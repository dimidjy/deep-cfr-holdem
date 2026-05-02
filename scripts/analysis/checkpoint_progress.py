#!/usr/bin/env python3
"""
Run self-play on Deep CFR checkpoints and print a progression table.

Usage:
  python3 scripts/analysis/checkpoint_progress.py models/deep_cfr/
  python3 scripts/analysis/checkpoint_progress.py models/deep_cfr/ --step 200 --hands 3000
  python3 scripts/analysis/checkpoint_progress.py models/deep_cfr/ --from-iter 1000 --to-iter 2400
"""

import sys
import os
import re
import argparse
import json
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _PROJECT_ROOT)

from src.decision_engine import DecisionEngine, DecisionConfig
from scripts.analysis.self_play import run_self_play, generate_json


GTO_RANGES = {
    'vpip':  (20, 25),
    'pfr':   (16, 21),
    'cbet':  (55, 70),
    'af':    (2.0, 3.5),
    'wtsd':  (25, 35),
}


def _in_range(val, key):
    lo, hi = GTO_RANGES[key]
    return lo <= val <= hi


def find_checkpoints(model_dir, step, from_iter, to_iter):
    """Find checkpoint files matching the step/range filters."""
    pat = re.compile(r'checkpoint_iter_(\d+)\.pt$')
    results = []
    for f in Path(model_dir).glob('checkpoint_iter_*.pt'):
        m = pat.search(f.name)
        if not m:
            continue
        it = int(m.group(1))
        if it < from_iter or it > to_iter:
            continue
        if it % step != 0:
            continue
        results.append((it, f))
    results.sort()
    return results


def run_eval(checkpoint_path, hands, players):
    """Run self-play on a single checkpoint, return stats dict."""
    dc = DecisionConfig(
        use_preflop_ranges=True,
        raise_randomize=False,
    )
    engine = DecisionEngine(dc)
    engine.load_models(
        facing_bet=str(checkpoint_path),
        checked_to=str(checkpoint_path),
    )
    engine.load_preflop_ranges()

    # Suppress per-run prints from self_play
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        raw_stats = run_self_play(engine, hands, players)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    return generate_json(raw_stats, players, hands)


def main():
    parser = argparse.ArgumentParser(
        description='Run self-play on checkpoints and show progression table')
    parser.add_argument('model_dir', type=str,
                        help='Directory with checkpoint_iter_*.pt files')
    parser.add_argument('--step', type=int, default=100,
                        help='Iteration step between checkpoints (default: 100)')
    parser.add_argument('--hands', type=int, default=5000,
                        help='Hands per self-play run (default: 5000)')
    parser.add_argument('--players', type=int, default=6,
                        help='Number of players (default: 6)')
    parser.add_argument('--from-iter', type=int, default=0,
                        help='Start from this iteration (default: 0)')
    parser.add_argument('--to-iter', type=int, default=999999,
                        help='Stop at this iteration (default: all)')
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.model_dir, args.step,
                                   args.from_iter, args.to_iter)
    if not checkpoints:
        print(f'No checkpoints found in {args.model_dir} '
              f'(step={args.step}, range={args.from_iter}-{args.to_iter})')
        sys.exit(1)

    print(f'Found {len(checkpoints)} checkpoints '
          f'(step={args.step}, {checkpoints[0][0]}→{checkpoints[-1][0]})')
    print(f'Running {args.hands} hands per checkpoint, {args.players}-max, no guards')
    print()

    # Header
    hdr = (f'{"Iter":>6} | {"VPIP":>6} | {"PFR":>6} | {"CBet":>6} | '
           f'{"AF":>6} | {"WTSD":>6} | {"W$SD":>6} | '
           f'{"Fold":>6} | {"Check":>6} | {"Call":>6} | {"Bet/R":>6} | Flags')
    print(hdr)
    print('-' * len(hdr) + '----------')

    results = []
    for it, path in checkpoints:
        try:
            d = run_eval(path, args.hands, args.players)
        except Exception as e:
            print(f'{it:>6} | ERROR: {e}')
            continue

        vpip = d['vpip']
        pfr  = d['pfr']
        cbet = d['cbet']
        af   = d['af']
        wtsd = d['wtsd']
        wsd  = d['wsd']

        act = d.get('postflop_actions', {})
        fold_pct = act.get('fold_pct', 0)
        check_pct = act.get('check_pct', 0)
        call_pct = act.get('call_pct', 0)
        bet_pct = act.get('bet_raise_pct', 0)

        flags = []
        for key, val in [('vpip', vpip), ('pfr', pfr), ('cbet', cbet),
                         ('af', af), ('wtsd', wtsd)]:
            if not _in_range(val, key):
                flags.append(key.upper())

        flag_str = ', '.join(flags) if flags else 'OK'
        print(f'{it:>6} | {vpip:>5.1f}% | {pfr:>5.1f}% | {cbet:>5.1f}% | '
              f'{af:>5.2f} | {wtsd:>5.1f}% | {wsd:>5.1f}% | '
              f'{fold_pct:>5.1f}% | {check_pct:>5.1f}% | {call_pct:>5.1f}% | '
              f'{bet_pct:>5.1f}% | {flag_str}')

        d['iteration'] = it
        results.append(d)

    # Save results
    if results:
        out_dir = Path(__file__).parent / 'results'
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = out_dir / f'checkpoint_progress_{ts}.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

        print()
        print(f'GTO ranges:  VPIP {GTO_RANGES["vpip"]}  PFR {GTO_RANGES["pfr"]}  '
              f'CBet {GTO_RANGES["cbet"]}  AF {GTO_RANGES["af"]}  WTSD {GTO_RANGES["wtsd"]}')
        print(f'Action dist = postflop only (Fold / Check / Call / Bet+Raise)')
        print(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
