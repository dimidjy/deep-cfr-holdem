"""
Migrate a checkpoint from old obs_size to new obs_size by padding the first
layer weight matrix with zeros for the new input dimensions.

The model keeps all learned weights for existing features. New feature columns
start at zero (no influence), and the network learns to use them over time.

Usage:
  python3 scripts/migrate_checkpoint.py \
      --src models/checkpoint_iter_800.pt \
      --dst models/checkpoint_iter_800_migrated.pt
"""
import sys
import os
import argparse
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.engine import EnhancedPokerEngine


def _make_engine(num_players, with_betting_history):
    return EnhancedPokerEngine(
        num_players=num_players,
        add_position_features=True,
        add_pot_odds_features=True,
        add_hand_strength_features=True,
        add_stack_features=True,
        add_equity_features=True,
        add_street_features=True,
        add_game_state_features=True,
        add_betting_history_features=with_betting_history,
    )


def migrate(src_path: str, dst_path: str):
    checkpoint = torch.load(src_path, map_location='cpu')

    # Detect old obs_size from first network's first layer
    old_weight = checkpoint['advantage_networks'][0]['network.0.weight']
    old_obs_size = old_weight.shape[1]
    num_players = len(checkpoint['advantage_networks'])

    # Detect new obs_size
    new_engine = _make_engine(num_players, with_betting_history=True)
    new_obs_size = new_engine.get_observation_space_size()

    if old_obs_size == new_obs_size:
        print(f'No migration needed — obs_size is already {old_obs_size}.')
        return

    added_dims = new_obs_size - old_obs_size
    if added_dims < 0:
        print(f'ERROR: new obs_size ({new_obs_size}) < old obs_size ({old_obs_size}). Check your engine config.')
        return

    print(f'Migrating: obs_size {old_obs_size} → {new_obs_size} (+{added_dims} dims)')
    print(f'Players: {num_players}')

    def pad_state_dict(state_dict):
        new_sd = {}
        for k, v in state_dict.items():
            # First linear layer weight: shape (hidden, obs_size) → pad last dim
            if k == 'network.0.weight' and v.shape[1] == old_obs_size:
                pad = torch.zeros(v.shape[0], added_dims, dtype=v.dtype)
                new_sd[k] = torch.cat([v, pad], dim=1)
                print(f'  Padded {k}: {list(v.shape)} → {list(new_sd[k].shape)}')
            else:
                new_sd[k] = v
        return new_sd

    new_checkpoint = dict(checkpoint)
    new_checkpoint['advantage_networks'] = [
        pad_state_dict(sd) for sd in checkpoint['advantage_networks']
    ]
    new_checkpoint['strategy_networks'] = [
        pad_state_dict(sd) for sd in checkpoint['strategy_networks']
    ]

    # Optimizer states encode param shapes indirectly via momentum buffers.
    # We reset them so Adam starts fresh (loses momentum but keeps LR schedule).
    # This is the safest option — momentum for the padded dimensions would be
    # meaningless anyway.
    for opt_list_key in ('advantage_optimizers', 'strategy_optimizers'):
        new_opt_states = []
        for opt_state in checkpoint[opt_list_key]:
            new_opt = dict(opt_state)
            new_opt['state'] = {}   # clear momentum buffers
            new_opt_states.append(new_opt)
        new_checkpoint[opt_list_key] = new_opt_states

    torch.save(new_checkpoint, dst_path)
    print(f'\nSaved migrated checkpoint to: {dst_path}')
    print('Optimizer momentum was reset (networks weights preserved).')


def main():
    parser = argparse.ArgumentParser(description='Migrate checkpoint to new obs_size')
    parser.add_argument('--src', required=True, help='Source checkpoint path')
    parser.add_argument('--dst', required=True, help='Destination checkpoint path')
    args = parser.parse_args()

    if not Path(args.src).exists():
        print(f'ERROR: source checkpoint not found: {args.src}')
        sys.exit(1)

    migrate(args.src, args.dst)


if __name__ == '__main__':
    main()
