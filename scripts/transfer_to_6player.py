"""
Transfer a trained 2-player Deep CFR model to a 6-player architecture.

Usage (after Phase 1 training completes):
    python3 scripts/transfer_to_6player.py \
        --src  models/final_model_enhanced.pt \
        --dst  models/6player_transferred.pt

Then update configs/training_config.json:
    "num_players": 6
    "num_iterations": 1000   (fine-tuning, not from scratch)

And run Phase 2:
    python3 scripts/train_gpu.py --resume models/6player_transferred.pt

How it works
------------
Obs layout: [base(54) | pos_one_hot(N) | rel_pos(1) | is_btn(1) | pot_odds(3) | hand(5) | stack(2)]
  2-player: 54 + 2 + 1 + 1 + 3 + 5 + 2 = 68 dims
  6-player: 54 + 6 + 1 + 1 + 3 + 5 + 2 = 72 dims

The 4 extra dims (positions 56-59 in 6-player obs) are the extra 4 position one-hot entries.
All other weights (hidden layers, output layer) are identical shapes and copied directly.
The input layer weight matrix is padded with zeros for the 4 new position columns.

For the 6-player model (6 networks), player 0 and 1 copy from the 2-player networks directly.
Players 2-5 are initialized from player 0 (they'll diverge during fine-tuning).
"""
import sys
import os
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training import DeepCFR
from src.engine import EnhancedPokerEngine
from src.utils import load_config

# Number of obs dims where the 4 extra 6-player position one-hot dims are inserted
# 2-player obs: [0:54 base | 54:56 pos_oh | 56 rel_pos | 57 is_btn | 58:68 rest]
# 6-player obs: [0:54 base | 54:60 pos_oh | 60 rel_pos | 61 is_btn | 62:72 rest]
OBS_SPLIT = 56        # cols 0..55 are shared (base + first 2 pos dims)
EXTRA_COLS = 4        # 4 extra one-hot dims for 6-player positions 2-5
SRC_PLAYERS = 2
DST_PLAYERS = 6


def _pad_input_weight(w2: torch.Tensor, src_obs: int, dst_obs: int) -> torch.Tensor:
    """
    Expand input weight matrix from [hidden, src_obs] to [hidden, dst_obs].
    The 4 new columns are inserted at position OBS_SPLIT (zero-initialised).
    """
    assert w2.shape[1] == src_obs, f"Expected src weight cols={src_obs}, got {w2.shape[1]}"
    hidden = w2.shape[0]
    w6 = torch.zeros(hidden, dst_obs, dtype=w2.dtype)
    w6[:, :OBS_SPLIT]       = w2[:, :OBS_SPLIT]          # base obs + pos_oh[0,1]
    # w6[:, OBS_SPLIT:OBS_SPLIT+EXTRA_COLS] = 0           # pos_oh[2,3,4,5] — already zero
    w6[:, OBS_SPLIT + EXTRA_COLS:] = w2[:, OBS_SPLIT:]   # rel_pos, is_btn, pot_odds, hand, stack
    return w6


def _transfer_state_dict(src_sd: dict, src_obs: int, dst_obs: int) -> dict:
    """Return a new state dict with the input layer weight padded for 6-player."""
    dst_sd = {}
    for k, v in src_sd.items():
        if k == 'network.0.weight':
            dst_sd[k] = _pad_input_weight(v, src_obs, dst_obs)
        else:
            dst_sd[k] = v.clone()
    return dst_sd


def main():
    parser = argparse.ArgumentParser(description='Transfer 2-player model → 6-player')
    parser.add_argument('--src', required=True, help='Path to 2-player checkpoint (.pt)')
    parser.add_argument('--dst', required=True, help='Output path for 6-player checkpoint (.pt)')
    parser.add_argument('--config', default=None, help='Config path (auto-detected if omitted)')
    args = parser.parse_args()

    config_path = args.config or str(Path(__file__).parent.parent / 'configs/training_config.json')
    config = load_config(config_path)

    # Build engines to get exact obs/act sizes for both player counts
    def _make_engine(n):
        e = EnhancedPokerEngine(
            num_players=n,
            add_position_features=True,
            add_pot_odds_features=True,
            add_hand_strength_features=True,
            add_stack_features=True,
        )
        return e.get_observation_space_size(), e.get_action_space_size()

    src_obs, src_act = _make_engine(SRC_PLAYERS)
    dst_obs, dst_act = _make_engine(DST_PLAYERS)
    assert src_act == dst_act, "Action space must be the same"
    assert dst_obs - src_obs == EXTRA_COLS, (
        f"Expected obs size diff of {EXTRA_COLS}, got {dst_obs - src_obs}"
    )

    print(f"Source: {SRC_PLAYERS}-player  obs={src_obs}  act={src_act}")
    print(f"Target: {DST_PLAYERS}-player  obs={dst_obs}  act={dst_act}")
    print(f"Loading 2-player checkpoint: {args.src}")

    src_ckpt = torch.load(args.src, map_location='cpu')
    src_adv   = src_ckpt['advantage_networks']   # list of 2 state dicts
    src_strat = src_ckpt['strategy_networks']    # list of 2 state dicts
    src_iter  = src_ckpt.get('iteration', 0)

    print(f"Source checkpoint iteration: {src_iter}")

    hidden_layers = config['network']['hidden_layers']
    dropout       = config['network'].get('dropout', 0.1)

    # Build a fresh 6-player DeepCFR just to get properly-shaped state dicts
    dst_deep_cfr = DeepCFR(
        observation_size=dst_obs,
        action_size=dst_act,
        num_players=DST_PLAYERS,
        hidden_layers=hidden_layers,
        dropout=dropout,
    )

    print(f"Transferring weights to 6-player model ({DST_PLAYERS} networks)...")
    for p in range(DST_PLAYERS):
        # Players 0 and 1 come from the matching 2-player networks.
        # Players 2-5 are cloned from player 0 (they diverge during fine-tuning).
        src_p = min(p, SRC_PLAYERS - 1)

        adv_sd   = _transfer_state_dict(src_adv[src_p],   src_obs, dst_obs)
        strat_sd = _transfer_state_dict(src_strat[src_p], src_obs, dst_obs)

        dst_deep_cfr.advantage_networks[p].load_state_dict(adv_sd)
        dst_deep_cfr.strategy_networks[p].load_state_dict(strat_sd)
        print(f"  player {p}  ← src player {src_p}  (input layer padded with zeros at cols {OBS_SPLIT}-{OBS_SPLIT+EXTRA_COLS-1})")

    # Save as a proper DeepCFR checkpoint.
    # Reset iteration to 0 so Phase 2 starts its own fresh iteration counter
    # (otherwise the training loop would start from e.g. 1750 and immediately
    # think it's "done" if num_iterations < src_iter).
    dst_deep_cfr.iteration = 0
    dst_deep_cfr.save_models(args.dst)
    print(f"\nSaved 6-player checkpoint to: {args.dst}")
    print("\nNext steps:")
    print("  1. Edit configs/training_config.json  →  \"num_players\": 6")
    print("  2. Set \"num_iterations\": 1000  (fine-tuning only)")
    print(f"  3. python3 scripts/train_gpu.py --resume {args.dst}")


if __name__ == '__main__':
    main()
