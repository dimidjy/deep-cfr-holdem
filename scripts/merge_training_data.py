"""
Merge training data from multiple player counts into one universal dataset.

Pads all observations to 6-player format (41 dims) so a single model
can be trained on data from 2-6 player tables.

Usage:
  python3 scripts/merge_training_data.py \
    --inputs data/training_data_6p.npz data/training_data_5p.npz \
             data/training_data_4p.npz data/training_data_3p.npz \
             data/training_data_2p.npz \
    --num-players 6 5 4 3 2 \
    --output data/training_data_universal.npz
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
from pathlib import Path
from src.utils.obs_padding import pad_observation_to_6p


def main():
    parser = argparse.ArgumentParser(description='Merge training data from multiple player counts')
    parser.add_argument('--inputs', nargs='+', required=True,
                        help='Input .npz files (one per player count)')
    parser.add_argument('--num-players', nargs='+', type=int, required=True,
                        help='Player count for each input file (same order)')
    parser.add_argument('--output', type=str, default='data/training_data_universal.npz',
                        help='Output .npz file')
    args = parser.parse_args()

    if len(args.inputs) != len(args.num_players):
        print('Error: --inputs and --num-players must have same number of arguments')
        sys.exit(1)

    all_obs = []
    all_actions = []
    all_legal_masks = []
    all_weights = []
    all_streets = []

    target_obs_dim = 41  # 6-player observation dimension

    for input_path, np_count in zip(args.inputs, args.num_players):
        path = Path(input_path)
        if not path.exists():
            print(f'Warning: {path} not found, skipping')
            continue

        print(f'Loading {path} ({np_count}p)...')
        data = np.load(str(path))

        observations = data['observations']
        actions = data['actions']
        legal_masks = data['legal_masks']
        weights = data['weights']
        streets = data['streets']

        n_samples = len(observations)
        obs_dim = observations.shape[1]
        print(f'  {n_samples} samples, obs_dim={obs_dim}')

        # Pad observations to 6p format if needed
        if np_count != 6:
            print(f'  Padding {np_count}p observations to 6p format ({obs_dim} -> {target_obs_dim})...')
            padded = np.zeros((n_samples, target_obs_dim), dtype=np.float32)
            for i in range(n_samples):
                padded[i] = pad_observation_to_6p(observations[i], np_count)
            observations = padded
            print(f'  Padded to {observations.shape}')
        else:
            if obs_dim != target_obs_dim:
                print(f'  Warning: expected {target_obs_dim} dims, got {obs_dim}')

        all_obs.append(observations)
        all_actions.append(actions)
        all_legal_masks.append(legal_masks)
        all_weights.append(weights)
        all_streets.append(streets)

        # Print action distribution
        action_names = {0: 'fold', 1: 'call', 2: 'raise_half', 3: 'raise_pot', 4: 'all_in'}
        unique, counts = np.unique(actions, return_counts=True)
        dist = {action_names.get(a, str(a)): c for a, c in zip(unique, counts)}
        print(f'  Actions: {dist}')

    if not all_obs:
        print('No data loaded!')
        sys.exit(1)

    # Concatenate
    print('\nMerging...')
    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    legal_masks = np.concatenate(all_legal_masks, axis=0)
    weights = np.concatenate(all_weights, axis=0)
    streets = np.concatenate(all_streets, axis=0)

    # Re-normalize weights to mean=1
    weights = weights / weights.mean()

    # Shuffle
    print('Shuffling...')
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(observations))
    observations = observations[indices]
    actions = actions[indices]
    legal_masks = legal_masks[indices]
    weights = weights[indices]
    streets = streets[indices]

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        legal_masks=legal_masks,
        weights=weights,
        streets=streets,
    )

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f'\nSaved {len(observations)} samples to {output_path} ({file_size_mb:.1f} MB)')
    print(f'  Observation dim: {observations.shape[1]}')

    # Final stats
    action_names = {0: 'fold', 1: 'call', 2: 'raise_half', 3: 'raise_pot', 4: 'all_in'}
    print(f'\n  Action distribution:')
    unique, counts = np.unique(actions, return_counts=True)
    for a, c in zip(unique, counts):
        name = action_names.get(a, str(a))
        print(f'    {name:<14} {c:>8}  ({c/len(actions)*100:.1f}%)')

    street_names = {0: 'flop', 1: 'turn', 2: 'river'}
    print(f'\n  Street distribution:')
    unique, counts = np.unique(streets, return_counts=True)
    for s, c in zip(unique, counts):
        name = street_names.get(s, str(s))
        print(f'    {name:<14} {c:>8}  ({c/len(streets)*100:.1f}%)')

    fold_pct = (actions == 0).sum() / len(actions) * 100
    print(f'\n  Fold ratio: {fold_pct:.1f}%')


if __name__ == '__main__':
    main()
