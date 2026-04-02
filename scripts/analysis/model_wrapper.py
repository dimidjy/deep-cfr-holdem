"""
Shared model wrapper for 3-model routing in analysis scripts.

Handles:
- Loading facing_bet, checked_to, and optional 3bet models
- Routing decisions based on obs[10] (pot_odds) and pot size
- Observation padding and equity_vs_odds computation
- Excluding fold for checked_to model

Usage:
    from scripts.analysis.model_wrapper import ModelWrapper
    model = ModelWrapper(
        model_fb_path='models/best/cql_6p_facing_bet_v7.pt',
        model_ct_path='models/best/cql_6p_checked_to_v3.pt',
        model_fb_3bet_path='models/cql_6p_facing_bet_3bet.pt',
    )
    action, strategy = model.get_action(obs, legal_actions)
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── Project setup ───────────────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _PROJECT_ROOT)

import torch
from scripts.train_cql import QNetwork, CQLModelWrapper

# ── Constants ───────────────────────────────────────────────────────────────
FOLD = 0
CALL = 1
RAISE_HALF = 2
RAISE_POT = 3
ALL_IN = 4

ACTION_NAMES = ['Fold', 'Call', 'RaiseHalf', 'RaisePot', 'AllIn']

IDX_POT_ODDS = 10   # obs[10] = pot_odds (0 when not facing a bet)
IDX_EQUITY = 18      # obs[18] = hand_strength / equity vs random
IDX_POT_NORM = 8     # obs[8] = pot_normalized

# 3bet pot threshold: pot_bb > 12
_3BET_POT_BB_THRESHOLD = 12


# ── Model loading ──────────────────────────────────────────────────────────

def _load_single_model(path: str, device: str = 'cpu', exclude_fold: bool = False):
    """Load a CQL model from checkpoint.

    Returns: (CQLModelWrapper, obs_size, action_size, metadata_dict)
    """
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get('model_type') not in ('cql', 'iql', None):
        raise ValueError(f"Expected CQL/IQL model, got: {checkpoint.get('model_type')}")

    obs_size = checkpoint['obs_size']
    action_size = checkpoint['action_size']
    hidden = checkpoint.get('hidden_layers', [512, 512, 256])
    dropout = checkpoint.get('dropout', 0.1)
    temperature = checkpoint.get('temperature', 1.0)
    num_players = checkpoint.get('num_players', 6)

    q_net = QNetwork(obs_size, action_size, hidden, dropout)
    q_net.load_state_dict(checkpoint['model_state_dict'])
    q_net.eval()

    wrapper = CQLModelWrapper(
        q_net=q_net,
        obs_size=obs_size,
        action_size=action_size,
        num_players=num_players,
        device=device,
        iteration=checkpoint.get('epoch', 0),
        temperature=temperature,
        exclude_fold=exclude_fold,
    )

    has_bh = checkpoint.get('behavioral_stats', False) or (obs_size - 41) >= 8
    has_evo = (obs_size - 41) >= 9

    meta = {
        'obs_size': obs_size,
        'action_size': action_size,
        'has_behavioral_stats': has_bh,
        'has_evo': has_evo,
        'epoch': checkpoint.get('epoch', 0),
        'temperature': temperature,
    }

    return wrapper, obs_size, action_size, meta


# ── Observation padding ────────────────────────────────────────────────────

def pad_observation(obs: np.ndarray, target_size: int) -> np.ndarray:
    """Pad observation to target size, computing equity_vs_odds for the last dim."""
    if len(obs) >= target_size:
        return obs[:target_size]
    padded = np.zeros(target_size, dtype=np.float32)
    padded[:len(obs)] = obs
    # Compute equity_vs_odds for the extra dimension
    if len(obs) >= 19 and target_size > len(obs):
        pot_odds = obs[IDX_POT_ODDS]
        equity = obs[IDX_EQUITY]
        evo_idx = target_size - 1
        padded[evo_idx] = np.clip(equity - (1.0 - pot_odds), -1.0, 1.0)
    return padded


# ── ModelWrapper ───────────────────────────────────────────────────────────

class ModelWrapper:
    """Unified model wrapper with 3-model routing for analysis scripts.

    Routes decisions to the appropriate model based on game state:
    - facing_bet: when obs[10] > 0.001 (pot_odds nonzero)
    - facing_bet_3bet: when facing_bet AND pot > 12BB
    - checked_to: when not facing a bet (fold excluded from legal actions)
    """

    def __init__(
        self,
        model_fb_path: str,
        model_ct_path: str,
        model_fb_3bet_path: Optional[str] = None,
        device: str = 'cpu',
    ):
        print(f'[ModelWrapper] Loading models...')

        # Load facing_bet model
        self.model_fb, self.obs_size_fb, self.act_size_fb, self.meta_fb = (
            _load_single_model(model_fb_path, device))
        print(f'  facing_bet: obs={self.obs_size_fb}, '
              f'actions={self.act_size_fb}, '
              f'bh={"ON" if self.meta_fb["has_behavioral_stats"] else "OFF"}')

        # Load checked_to model
        self.model_ct, self.obs_size_ct, self.act_size_ct, self.meta_ct = (
            _load_single_model(model_ct_path, device, exclude_fold=False))
        print(f'  checked_to: obs={self.obs_size_ct}, '
              f'actions={self.act_size_ct}, '
              f'bh={"ON" if self.meta_ct["has_behavioral_stats"] else "OFF"}')

        # Load optional 3bet model
        self.model_fb_3bet = None
        self.obs_size_fb3 = None
        self.meta_fb3 = None
        if model_fb_3bet_path:
            self.model_fb_3bet, self.obs_size_fb3, _, self.meta_fb3 = (
                _load_single_model(model_fb_3bet_path, device))
            print(f'  3bet facing_bet: obs={self.obs_size_fb3}')

        self.device = device
        print(f'[ModelWrapper] All models loaded.')

    def is_facing_bet(self, obs: np.ndarray) -> bool:
        """Check if the observation indicates facing a bet."""
        return float(obs[IDX_POT_ODDS]) > 0.001 if len(obs) > IDX_POT_ODDS else False

    def is_3bet_pot(self, obs: np.ndarray) -> bool:
        """Check if pot is large enough to be a 3bet pot."""
        pot_bb = obs[IDX_POT_NORM] * 600 if len(obs) > IDX_POT_NORM else 0
        return pot_bb > _3BET_POT_BB_THRESHOLD

    def get_action(
        self,
        obs: np.ndarray,
        legal_actions: List[int],
        player_idx: int = 0,
    ) -> Tuple[int, np.ndarray]:
        """Get model's preferred action and full strategy.

        Automatically routes to the correct model based on game state.
        For checked_to: excludes fold from legal actions.

        Args:
            obs: Observation array from obs_builder.
            legal_actions: List of legal action indices (0-4).
            player_idx: Player index (default 0, used for model inference).

        Returns:
            (best_action_idx, strategy_array)
        """
        facing_bet = self.is_facing_bet(obs)

        if facing_bet:
            # Select 3bet model for large pots if available
            if self.model_fb_3bet is not None and self.is_3bet_pot(obs):
                model = self.model_fb_3bet
                target_size = self.obs_size_fb3
            else:
                model = self.model_fb
                target_size = self.obs_size_fb
            model_legal = [a for a in legal_actions if a < 5] or [CALL]
        else:
            model = self.model_ct
            target_size = self.obs_size_ct
            # Exclude fold for checked_to spots
            model_legal = [a for a in legal_actions if a != FOLD and a < 5] or [CALL]

        # Pad observation
        padded = pad_observation(obs, target_size)

        # Get strategy from model
        strategy = model.get_strategy(padded, player_idx, model_legal)

        # Pick best action
        best_action = max(model_legal, key=lambda a: strategy[a])
        return best_action, strategy

    def get_strategy(
        self,
        obs: np.ndarray,
        legal_actions: List[int],
        player_idx: int = 0,
    ) -> np.ndarray:
        """Get model's full strategy (probability over actions).

        Same routing as get_action but returns only the strategy array.
        """
        _, strategy = self.get_action(obs, legal_actions, player_idx)
        return strategy

    def info(self) -> str:
        """Return a short description of loaded models."""
        parts = [
            f'FB(obs={self.obs_size_fb})',
            f'CT(obs={self.obs_size_ct})',
        ]
        if self.model_fb_3bet:
            parts.append(f'3BET(obs={self.obs_size_fb3})')
        return ', '.join(parts)
