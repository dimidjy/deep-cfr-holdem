#!/usr/bin/env python3
"""Strip a Deep CFR training checkpoint down to what inference needs.

Training checkpoints carry optimizer and LR-scheduler state for every player
(~3x the size of the networks). Self-play, tournaments, the GUI and the spot
evaluator only read the networks, so published checkpoints drop the rest.

Usage:
  python3 scripts/export_checkpoint.py models/v3/checkpoint_iter_4000.pt \
      dist/checkpoint_iter_4000.pt
"""
import argparse
import sys
from pathlib import Path

import torch

KEEP = ('iteration', 'advantage_networks', 'strategy_networks')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('src', type=Path)
    ap.add_argument('dst', type=Path)
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location='cpu')
    missing = [k for k in KEEP if k not in ckpt]
    if missing:
        sys.exit(f'{args.src} is not a Deep CFR checkpoint (missing {missing})')

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: ckpt[k] for k in KEEP}, args.dst)
    print(f'{args.src} ({args.src.stat().st_size / 1e6:.0f} MB) -> '
          f'{args.dst} ({args.dst.stat().st_size / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
