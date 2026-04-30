"""Score a DeepCFR model against manually-annotated spot JSONs.

Only looks at spots that have `right_action` set (via the annotator tool).
Unannotated and deleted spots are ignored.

Scoring per spot (action match between model's top choice and right_action):
    +1.0    exact match
    +0.5    both are raise actions but different sizes (e.g. Raise½ vs RaisePot)
    +0.0    anything else

Aggregate score = sum of per-spot points. Percentage = total_points / spot_count.

Action codes:
    0 = Fold
    1 = Check / Call
    2 = Raise½
    3 = RaisePot
    4 = All-in
    5 = Raise Min (2BB)

Usage:
    python3 scripts/validate_spot_logs.py --model models/v12/checkpoint_iter_2600.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Action indices grouped for partial-credit scoring.
# Any two RAISE actions score 0.5 when they differ (same sizing direction,
# different magnitude). Fold/Check don't get partial credit.
_RAISE_ACTIONS = {2, 3, 4, 5}

_ACTION_NAMES = {
    0: 'Fold', 1: 'Check/Call', 2: 'Raise½', 3: 'RaisePot',
    4: 'AllIn', 5: 'RaiseMin',
}

# Valid obs sizes we know how to score against. Each entry corresponds to a
# historical or current schema. New rows must be added here whenever the obs
# layout changes, otherwise spots from that schema get silently dropped.
#   50/51 — engine/agent base schema (pre-betting-line)
#   53/54 — engine/agent with the betting-line block (3 extra dims)
#   55/56 — engine/agent with opponent profile (5 extra dims)
_VALID_OBS_SIZES = {50, 51, 53, 54, 55, 56}


def score_action(model_action: int, right_action: int) -> float:
    """Partial-credit scoring between model's argmax and the annotated right action.

    Exact match → 1.0
    Both raise sizes (but different) → 0.5
    Anything else → 0.0
    """
    if model_action == right_action:
        return 1.0
    if model_action in _RAISE_ACTIONS and right_action in _RAISE_ACTIONS:
        return 0.5
    return 0.0


def _load_spot(path: Path) -> Optional[Dict]:
    """Load one spot JSON. Return None unless it's annotated, not deleted,
    and has a usable obs_vector."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if data.get('deleted'):
        return None
    if not isinstance(data.get('right_action'), int):
        return None
    if not isinstance(data.get('obs_vector'), list):
        return None
    if len(data['obs_vector']) not in _VALID_OBS_SIZES:
        return None
    return data


def load_annotated_spots(spots_dir: str) -> List[Dict]:
    """Load every annotated, non-deleted spot with a valid obs_vector."""
    path = Path(spots_dir)
    if not path.is_dir():
        return []
    spots = []
    for p in sorted(path.glob('*.json')):
        data = _load_spot(p)
        if data is not None:
            spots.append(data)
    return spots


def score_model(deep_cfr, spots: List[Dict], num_players: int = 6) -> Dict:
    """Run the model against each annotated spot and aggregate partial-credit
    scores. Returns dict with totals + per-street + per-action breakdowns."""
    if not spots:
        return {'n': 0}

    model_obs_size = getattr(deep_cfr, 'observation_size', None)

    total_score = 0.0
    n_scored = 0
    exact_matches = 0   # full +1.0 hits
    partial_matches = 0  # +0.5 hits

    per_street: Dict[str, Dict[str, float]] = {
        s: {'score': 0.0, 'n': 0} for s in ('preflop', 'flop', 'turn', 'river')
    }
    # Per right_action breakdown — how well does the model match each target?
    per_action: Dict[int, Dict[str, float]] = {
        a: {'score': 0.0, 'n': 0} for a in range(6)
    }

    for spot in spots:
        right = int(spot['right_action'])
        obs = np.asarray(spot['obs_vector'], dtype=np.float32)
        if model_obs_size and len(obs) > model_obs_size:
            obs = obs[:model_obs_size]
        elif model_obs_size and len(obs) < model_obs_size:
            # Spot log was generated under an older obs builder. Zero-pad up
            # to the model's expected size so newer-architecture models can
            # still be scored against legacy spot logs (the missing features
            # default to 0, which biases scores DOWN — actual model quality
            # may be higher once the agent's obs_builder is upgraded).
            padded = np.zeros(model_obs_size, dtype=np.float32)
            padded[:len(obs)] = obs
            obs = padded

        gs = spot.get('game_state', {})
        legal = gs.get('legal_actions') or list(range(6))
        street = gs.get('street', 'unknown')
        player_idx = gs.get('my_seat', 0) % num_players

        try:
            strategy = deep_cfr.get_strategy(obs, player_idx, legal)
        except Exception:
            continue

        model_action = int(np.argmax(strategy))
        points = score_action(model_action, right)

        total_score += points
        n_scored += 1
        if points == 1.0:
            exact_matches += 1
        elif points == 0.5:
            partial_matches += 1

        if street in per_street:
            per_street[street]['score'] += points
            per_street[street]['n'] += 1
        if right in per_action:
            per_action[right]['score'] += points
            per_action[right]['n'] += 1

    return {
        'n': n_scored,
        'total_score': total_score,
        'max_score': float(n_scored),
        'percent': (100.0 * total_score / n_scored) if n_scored else 0.0,
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'per_street': per_street,
        'per_action': per_action,
    }


def print_report(stats: Dict) -> None:
    n = stats.get('n', 0)
    if n == 0:
        print('  [spots] no annotated logs found')
        return

    print(f"  [spots {n}] "
          f"score={stats['total_score']:.1f}/{n} "
          f"({stats['percent']:.1f}%)  "
          f"exact={stats['exact_matches']}  "
          f"partial={stats['partial_matches']}")

    # Per-street breakdown
    parts = []
    for s in ('preflop', 'flop', 'turn', 'river'):
        ps = stats['per_street'].get(s, {})
        if ps.get('n', 0) > 0:
            pct = 100.0 * ps['score'] / ps['n']
            parts.append(f"{s[:2]}={pct:.0f}%({ps['n']})")
    if parts:
        print('  by street:  ' + '  '.join(parts))

    # Per-right-action breakdown — how well does the model match each
    # target action type?
    print('  by right action:')
    for a, pa in stats['per_action'].items():
        if pa.get('n', 0) > 0:
            pct = 100.0 * pa['score'] / pa['n']
            name = _ACTION_NAMES.get(a, f'a{a}')
            print(f'    {name:<12} n={pa["n"]:<3} score={pa["score"]:.1f}/{pa["n"]}  ({pct:.0f}%)')


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--model', required=True,
                    help='Path to DeepCFR .pt checkpoint')
    ap.add_argument('--spots-dir', default='logs/action_debug',
                    help='Directory of spot JSONs (default: logs/action_debug)')
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.training import DeepCFR
    from src.utils import load_config
    from src.engine.factory import make_engine

    config_path = Path(__file__).resolve().parent.parent / 'configs/training_config.json'
    config = load_config(str(config_path))
    engine = make_engine(num_players=config['game']['num_players'])
    obs_size = engine.get_observation_space_size()
    act_size = engine.get_action_space_size()

    deep_cfr = DeepCFR(
        observation_size=obs_size, action_size=act_size,
        num_players=config['game']['num_players'],
        hidden_layers=config['network']['hidden_layers'],
        learning_rate=config['training']['learning_rate'],
        buffer_size=config['training']['buffer_size'],
        dropout=config['network'].get('dropout', 0.1),
        device='cpu',
    )
    deep_cfr.load_models(args.model)

    spots = load_annotated_spots(args.spots_dir)
    print(f'Loaded {len(spots)} annotated spots from {args.spots_dir}')
    stats = score_model(deep_cfr, spots,
                         num_players=config['game']['num_players'])
    print_report(stats)


if __name__ == '__main__':
    main()
