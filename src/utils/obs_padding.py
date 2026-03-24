"""
Observation padding for using a 6-player model at any table size (2-6 players).

The observation vector layout is:
  Position(N+2) + Features(33)

Where Position is a one-hot of N player slots + is_dealer + is_blind.
The 33 feature dims (pot_odds, hand_strength, equity, etc.) are identical
regardless of player count.

This module maps N-player positions to their 6-player equivalents and
zero-pads the unused position slots.
"""

import numpy as np

# Maps (num_players, rel_pos) → 6p rel_pos
# rel_pos: 0=BTN, 1=SB, 2=BB, then EP positions
_POS_MAP_TO_6P = {
    6: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},  # identity
    5: {0: 0, 1: 1, 2: 2, 3: 4, 4: 5},          # HJ→4(MP), CO→5
    4: {0: 0, 1: 1, 2: 2, 3: 5},                 # CO→5
    3: {0: 0, 1: 1, 2: 2},                        # BTN, SB, BB only
    2: {0: 0, 1: 2},                              # BTN→0, BB→2
}

# 6p observation size
_OBS_DIM_6P = 41
_POS_BLOCK_6P = 8  # 6 + 2 (is_dealer, is_blind)


def pad_observation_to_6p(obs: np.ndarray, num_players: int) -> np.ndarray:
    """Pad an N-player observation to 6-player format.

    Args:
        obs: Observation vector from N-player engine (dim = N+2+33)
        num_players: Actual number of players (2-6)

    Returns:
        6-player observation vector (dim = 41)
    """
    if num_players == 6:
        return obs

    pos_size_src = num_players + 2
    features = obs[pos_size_src:]  # 33 dims (same for all player counts)

    # Build 6p position block
    pos_6p = np.zeros(_POS_BLOCK_6P, dtype=np.float32)

    # Map one-hot position (first num_players dims)
    one_hot = obs[:num_players]
    if one_hot.sum() > 0:
        active_pos = int(np.argmax(one_hot))
        mapped_pos = _POS_MAP_TO_6P[num_players].get(active_pos, 0)
        pos_6p[mapped_pos] = 1.0

    # Copy is_dealer and is_blind to their 6p positions
    pos_6p[6] = obs[num_players]      # is_dealer
    pos_6p[7] = obs[num_players + 1]  # is_blind

    return np.concatenate([pos_6p, features]).astype(np.float32)


def map_player_idx_to_6p(player_idx: int, num_players: int) -> int:
    """Map N-player player_idx to 6p equivalent."""
    return _POS_MAP_TO_6P.get(num_players, {}).get(player_idx, player_idx)
