#!/usr/bin/env python3
"""Benchmark a checkpoint against a ladder of fixed opponents.

6-max, 100BB: the model sits in seats 0/2/4 and the opponent in seats 1/3/5,
with the button rotating every hand. Opponents:

  * scripted baselines: uniform random, always call/check, always pot-raise
  * the model itself with an archetype skew (calling station, nit, maniac,
    weak-tight): same network, distorted action frequencies
  * the unmodified model (mirror match, expected ~0)

The model's result is reported in BB/100 per seat with a 95% confidence
interval. Each hand contributes one sample: the summed net of the three model
seats, which keeps the seat-to-seat correlation inside a hand out of the
error estimate.

Usage:
  python3 scripts/benchmark_baselines.py --model models/best/checkpoint_iter_4000.pt \
      --hands 50000 --output docs/results/baselines.json
"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from src.archetypes import get_archetype_skew  # noqa: E402
from src.decision_engine import (  # noqa: E402
    CALL, NUM_ACTIONS, RAISE_POT, DecisionConfig, DecisionEngine)
from self_play import ArchetypeEngine, run_self_play  # noqa: E402

MODEL_SEATS = (0, 2, 4)


class ScriptedEngine:
    """Opponent that ignores the cards: picks actions by a fixed rule."""

    def __init__(self, rule: str):
        self._rule = rule

    def decide(self, ctx):
        legal = list(ctx.legal_actions)
        if self._rule == 'random':
            action = int(np.random.choice(legal))
        elif self._rule == 'call':
            action = CALL if CALL in legal else legal[0]
        else:  # 'raise'
            action = next((a for a in (RAISE_POT, CALL) if a in legal), legal[0])
        strategy = [0.0] * NUM_ACTIONS
        strategy[action] = 1.0
        return action, strategy

    def info(self):
        return f'scripted:{self._rule}'


def load_engine(model: str, equity_model: str) -> DecisionEngine:
    if equity_model and Path(equity_model).exists():
        import torch
        from src.engine.fast_enhanced_engine import load_equity_model
        load_equity_model(equity_model,
                          device='cuda' if torch.cuda.is_available() else 'cpu')
    engine = DecisionEngine(DecisionConfig())
    engine.load_models(facing_bet=model, checked_to=model)
    engine.load_preflop_ranges()
    return engine


def opponents(engine: DecisionEngine):
    yield 'random', 'Random actions', ScriptedEngine('random')
    yield 'always_call', 'Always call', ScriptedEngine('call')
    yield 'always_raise', 'Always pot-raise', ScriptedEngine('raise')
    for name, label in (('calling_station', 'Model + calling-station skew'),
                        ('maniac', 'Model + maniac skew'),
                        ('nit', 'Model + nit skew'),
                        ('weak_tight', 'Model + weak-tight skew')):
        yield name, label, ArchetypeEngine(engine, get_archetype_skew(name))
    yield 'mirror', 'Itself (mirror match)', engine


def bb100_with_ci(stats) -> dict:
    per_hand = np.sum([stats[s].hand_won_bb for s in MODEL_SEATS], axis=0)
    per_seat = per_hand / len(MODEL_SEATS)
    n = len(per_seat)
    mean = float(per_seat.mean()) * 100
    half = 1.96 * float(per_seat.std(ddof=1)) / np.sqrt(n) * 100
    return {'hands': n, 'bb100': round(mean, 2), 'ci95': round(half, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--model', default='models/best/checkpoint_iter_4000.pt')
    ap.add_argument('--equity-model', default='models/equity_model_v3.pt')
    ap.add_argument('--hands', type=int, default=50000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--only', nargs='+', default=None,
                    help='run a subset of opponents by key')
    ap.add_argument('--output', type=Path, default=Path('docs/results/baselines.json'))
    args = ap.parse_args()

    engine = load_engine(args.model, args.equity_model)
    results = []
    for key, label, opp in opponents(engine):
        if args.only and key not in args.only:
            continue
        np.random.seed(args.seed)
        t0 = time.time()
        stats = run_self_play(engine, args.hands, num_players=6, engine_b=opp)
        row = {'opponent': key, 'label': label, **bb100_with_ci(stats)}
        results.append(row)
        print(f"{label:<32} {row['bb100']:>+9.1f} ± {row['ci95']:<6.1f} BB/100 "
              f"({row['hands']:,} hands, {time.time() - t0:.0f}s)", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'model': Path(args.model).name,
        'date': date.today().isoformat(),
        'hands_per_match': args.hands,
        'format': '6-max NLHE, 100BB, model in seats 0/2/4',
        'results': results,
    }, indent=2) + '\n')
    print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
