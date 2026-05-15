#!/usr/bin/env python3
"""Average weights across multiple Deep CFR checkpoints (model soup / SWA).

Usage:
  python3 scripts/make_model_soup.py \\
      --inputs models/best_pool/checkpoint_iter_3900.pt \\
               models/best_pool/checkpoint_iter_4000.pt \\
               models/best_pool/checkpoint_iter_4100.pt \\
               models/best_pool/checkpoint_iter_4200.pt \\
      --output models/best_pool/soup_3900_4200.pt
"""
import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--label-iter', type=int, default=None,
                    help='Iteration label written into the soup checkpoint. '
                         'Defaults to mean of input iterations.')
    args = ap.parse_args()

    ckpts = []
    for p in args.inputs:
        print(f'Loading {p}')
        ckpts.append(torch.load(p, map_location='cpu', weights_only=False))

    iters = [c.get('iteration', 0) for c in ckpts]
    print(f'Source iterations: {iters}')

    soup = ckpts[0]
    for key in ('advantage_networks', 'strategy_networks'):
        if key not in soup:
            continue
        for net_idx in range(len(soup[key])):
            for param_name in soup[key][net_idx]:
                stacked = torch.stack([
                    c[key][net_idx][param_name].float() for c in ckpts
                ])
                soup[key][net_idx][param_name] = stacked.mean(dim=0)
        print(f'  averaged {key}: {len(soup[key])} net(s)')

    soup['iteration'] = args.label_iter if args.label_iter is not None else int(sum(iters) / len(iters))
    for k in ('advantage_schedulers', 'strategy_schedulers'):
        soup.pop(k, None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup, args.output)
    print(f'Wrote {args.output} (label iteration={soup["iteration"]})')


if __name__ == '__main__':
    main()
