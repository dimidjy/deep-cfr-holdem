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
Obs layout (feature-only, no raw card encoding):
  [pos_one_hot(N) | rel_pos(1) | is_btn(1) | pot_odds(3) | hand(5) | stack(2) | equity(4) | street(4) | game_state(3)]
  2-player: 2 + 1 + 1 + 3 + 5 + 2 + 4 + 4 + 3 = 25 dims
  6-player: 6 + 1 + 1 + 3 + 5 + 2 + 4 + 4 + 3 = 29 dims

The 4 extra dims are the extra position one-hot entries at indices 2-5 in 6-player obs.
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
# 2-player obs: [0:2 pos_oh | 2 rel_pos | 3 is_btn | 4:25 rest]
# 6-player obs: [0:6 pos_oh | 6 rel_pos | 7 is_btn | 8:29 rest]
OBS_SPLIT = 2         # first 2 cols are shared (pos_oh[0], pos_oh[1])
EXTRA_COLS = 4        # 4 extra one-hot dims for 6-player positions 2-5
SRC_PLAYERS = 2
DST_PLAYERS = 6


def _pad_input_weight(w2: torch.Tensor, src_obs: int, dst_obs: int,
                      src_players: int, dst_players: int) -> torch.Tensor:
    """
    Expand input weight matrix from [hidden, src_obs] to [hidden, dst_obs].

    Feature layout (in order):
      pos_oh(N), rel_pos(1), is_btn(1), pot_odds(3), hand(5), stack(2),
      [equity(4) — optional], street(4), game_state(3)

    Without equity: obs = N + 19
    With    equity: obs = N + 23

    Handles two possible gaps:
      1. Extra pos_oh columns (dst_players - src_players), always 4.
      2. Equity columns (4) present in dst but absent in src (or vice versa).
    """
    assert w2.shape[1] == src_obs, f"Expected src weight cols={src_obs}, got {w2.shape[1]}"
    hidden = w2.shape[0]
    w6 = torch.zeros(hidden, dst_obs, dtype=w2.dtype)

    src_has_equity = (src_obs - src_players) == 23
    dst_has_equity = (dst_obs - dst_players) == 23

    sc, dc = 0, 0  # src/dst column cursors

    # pos_oh: copy first src_players cols, remaining dst_players-src_players stay zero
    w6[:, dc:dc + src_players] = w2[:, sc:sc + src_players]
    sc += src_players
    dc += dst_players

    # rel_pos(1) + is_btn(1) + pot_odds(3) + hand(5) + stack(2) = 12 cols — identical
    w6[:, dc:dc + 12] = w2[:, sc:sc + 12]
    sc += 12
    dc += 12

    # equity (4 cols) — may be absent in source
    if src_has_equity and dst_has_equity:
        w6[:, dc:dc + 4] = w2[:, sc:sc + 4]
        sc += 4
        dc += 4
    elif dst_has_equity:
        dc += 4          # insert zeros in dst; src has no equity cols
    elif src_has_equity:
        sc += 4          # skip equity in src; dst doesn't use it

    # street(4) + game_state(3) = 7 cols — identical
    w6[:, dc:dc + 7] = w2[:, sc:sc + 7]
    sc += 7
    dc += 7

    assert sc == src_obs, f"Column mapping error: consumed {sc} src cols, expected {src_obs}"
    assert dc == dst_obs, f"Column mapping error: filled {dc} dst cols, expected {dst_obs}"
    return w6


def _transfer_state_dict(src_sd: dict, src_obs: int, dst_obs: int,
                         src_players: int, dst_players: int) -> dict:
    """Return a new state dict with the input layer weight padded for 6-player."""
    dst_sd = {}
    for k, v in src_sd.items():
        if k == 'network.0.weight':
            dst_sd[k] = _pad_input_weight(v, src_obs, dst_obs, src_players, dst_players)
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
            add_equity_features=True,
            add_street_features=True,
            add_game_state_features=True,
        )
        return e.get_observation_space_size(), e.get_action_space_size()

    _, src_act = _make_engine(SRC_PLAYERS)
    dst_obs, dst_act = _make_engine(DST_PLAYERS)
    assert src_act == dst_act, "Action space must be the same"

    print(f"Loading 2-player checkpoint: {args.src}")

    src_ckpt = torch.load(args.src, map_location='cpu')
    src_adv   = src_ckpt['advantage_networks']   # list of 2 state dicts
    src_strat = src_ckpt['strategy_networks']    # list of 2 state dicts
    src_iter  = src_ckpt.get('iteration', 0)

    # Read the actual obs size from the checkpoint weights (may differ from engine if
    # treys was unavailable during training — equity features would be absent)
    src_obs = src_adv[0]['network.0.weight'].shape[1]

    print(f"Source: {SRC_PLAYERS}-player  obs={src_obs}  act={src_act}  (from checkpoint)")
    print(f"Target: {DST_PLAYERS}-player  obs={dst_obs}  act={dst_act}")
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

        adv_sd   = _transfer_state_dict(src_adv[src_p],   src_obs, dst_obs, SRC_PLAYERS, DST_PLAYERS)
        strat_sd = _transfer_state_dict(src_strat[src_p], src_obs, dst_obs, SRC_PLAYERS, DST_PLAYERS)

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
