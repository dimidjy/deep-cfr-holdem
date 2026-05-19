"""Rebuild obs_vector + features for spot JSONs that pre-date the obs-in-debug feature.

Older spots under logs/action_debug/ only have `game_state` — no `obs_vector`.
The validator requires obs_vector to score them. This script reconstructs
obs_vector (and the named `features` dict) by running the agent's obs_builder
over each spot's game_state, then writes them back into the JSON in place.

Usage:
    # Dry run (default): print what would be rebuilt
    python3 scripts/rebuild_spot_obs.py

    # Execute: actually write obs_vector/features back into JSONs
    python3 scripts/rebuild_spot_obs.py --execute

    # Only rebuild annotated spots (saves time)
    python3 scripts/rebuild_spot_obs.py --annotated-only --execute
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.game_state import GameState
from src.agent.obs_builder import ObservationBuilder, feature_names


def rebuild_one(data: dict, builder: ObservationBuilder, names: list) -> bool:
    """Rebuild obs_vector + features on the data dict (mutates in place).
    Returns True on success. False = skip this spot (unusable input)."""
    gs_dict = data.get('game_state')
    if not gs_dict:
        return False

    try:
        gs = GameState.from_dict(gs_dict)
    except Exception:
        return False

    try:
        obs = builder.build(gs)
    except Exception:
        return False
    if obs is None:
        return False

    obs_list = [float(v) for v in obs]
    data['obs_vector'] = obs_list

    # Named features map — only if dim matches
    if len(names) == len(obs_list):
        data['features'] = dict(zip(names, obs_list))
    else:
        data['features'] = {f'f{i}': v for i, v in enumerate(obs_list)}
        data['features_warning'] = (
            f'feature_names count {len(names)} != obs {len(obs_list)}')

    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--spots-dir', default='logs/action_debug',
                    help='Directory of spot JSONs (default: logs/action_debug)')
    ap.add_argument('--execute', action='store_true',
                    help='Actually write updated JSONs (otherwise dry-run)')
    ap.add_argument('--annotated-only', action='store_true',
                    help='Only rebuild spots with right_action set (skip unannotated)')
    ap.add_argument('--force', action='store_true',
                    help='Rebuild even if obs_vector already present')
    args = ap.parse_args()

    spots_dir = Path(args.spots_dir)
    if not spots_dir.is_dir():
        print(f'Directory not found: {spots_dir}')
        return

    # Build with same flags used in the agent (6-max, pad_to_6p, behavioral_stats,
    # threat_range_equity) so obs matches what validate_spot_logs expects.
    # v41+ models are 54-dim with threat_range_equity; older 53-dim still works.
    builder = ObservationBuilder(
        num_players=6, pad_to_6p=True,
        behavioral_stats=True, equity_vs_odds=False,
        threat_range_equity=True,
    )
    names = feature_names(num_players=6, pad_to_6p=True,
                          behavioral_stats=True, equity_vs_odds=False,
                          threat_range_equity=True)
    print(f'obs dim expected: {len(names)}')

    total = 0
    candidates = 0
    rebuilt = 0
    skipped_has_obs = 0
    skipped_no_gs = 0
    errors = 0

    paths = list(spots_dir.glob('*.json'))
    for p in paths:
        total += 1
        try:
            data = json.loads(p.read_text())
        except Exception:
            errors += 1
            continue

        if args.annotated_only and not isinstance(data.get('right_action'), int):
            continue
        if data.get('deleted'):
            continue  # skip soft-deleted

        already = isinstance(data.get('obs_vector'), list)
        if already and not args.force:
            skipped_has_obs += 1
            continue

        if 'game_state' not in data:
            skipped_no_gs += 1
            continue

        candidates += 1
        if rebuild_one(data, builder, names):
            if args.execute:
                try:
                    p.write_text(json.dumps(data, indent=2, default=str))
                except Exception:
                    errors += 1
                    continue
            rebuilt += 1
        else:
            errors += 1

    print()
    print(f'Total JSONs scanned:        {total}')
    print(f'Already had obs_vector:     {skipped_has_obs}')
    print(f'Missing game_state:         {skipped_no_gs}')
    print(f'Candidates to rebuild:      {candidates}')
    print(f'Successfully rebuilt:       {rebuilt}')
    print(f'Failed / errors:            {errors}')
    if not args.execute:
        print()
        print('[DRY RUN] Pass --execute to actually write the files.')


if __name__ == '__main__':
    main()
