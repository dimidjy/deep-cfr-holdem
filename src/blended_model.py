"""Blended model: fold decisions from Deep CFR, call/raise from supervised model.

Usage:
    from src.blended_model import BlendedModel

    blended = BlendedModel(
        supervised_path='models/ratio_0.05.pt',
        cfr_path='models/checkpoint_iter_1950.pt',
        config=config,
        num_players=6,
    )
    # blended.get_strategy(obs, player_idx, legal_actions) → np.ndarray
    # Works as drop-in replacement for DeepCFR in GUI, validation, etc.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional

from src.training.deep_cfr import StrategyNetwork, DeepCFR
from src.utils import load_config

FOLD = 0


class BlendedModel:
    """Blends two models: fold probability from CFR, action distribution from supervised."""

    def __init__(self, supervised_path: str, cfr_path: str,
                 config: dict, num_players: int = 6,
                 device: str = 'cpu', fold_blend: float = 1.0,
                 legacy_cfr: bool = False,
                 min_fold_weight: float = 0.0):
        """
        Args:
            supervised_path: path to supervised .pt checkpoint
            cfr_path: path to Deep CFR .pt checkpoint
            config: training config dict (for network architecture)
            num_players: number of players
            device: 'cpu' or 'cuda'
            fold_blend: how much CFR fold to use (1.0 = full CFR fold, 0.0 = no fold)
            legacy_cfr: True if CFR model uses legacy obs format (3-dim game_state)
            min_fold_weight: minimum CFR fold influence (0.0=off, 0.3=always 30%+ CFR fold)
        """
        self.device = torch.device(device)
        self.num_players = num_players
        self.fold_blend = fold_blend
        self.legacy_cfr = legacy_cfr
        self.min_fold_weight = min_fold_weight

        # Load supervised model
        sup_ckpt = torch.load(supervised_path, map_location=device)
        obs_size = sup_ckpt.get('obs_size', 38)
        self.action_size = sup_ckpt.get('action_size', 5)
        hidden = sup_ckpt.get('hidden_layers', config['network']['hidden_layers'])
        dropout = sup_ckpt.get('dropout', 0.1)

        self.sup_model = StrategyNetwork(obs_size, self.action_size, hidden, dropout)
        self.sup_model.load_state_dict(sup_ckpt['model_state_dict'])
        self.sup_model.eval()
        self.sup_model.to(self.device)

        # Load Deep CFR model
        from src.engine import EnhancedPokerEngine
        engine = EnhancedPokerEngine(num_players=num_players, legacy_game_state=legacy_cfr)

        self.cfr_model = DeepCFR(
            observation_size=engine.get_observation_space_size(),
            action_size=engine.get_action_space_size(),
            num_players=num_players,
            hidden_layers=config['network']['hidden_layers'],
            device=device,
        )
        self.cfr_model.load_models(cfr_path)

        # Expose attributes that validation/GUI scripts expect
        self.iteration = self.cfr_model.iteration
        self.strategy_networks = [self.sup_model] * num_players
        self.advantage_networks = self.cfr_model.advantage_networks

        # If CFR uses legacy obs, we need to convert current→legacy observations
        # Game_state block starts at index 30 (after Position8+PotOdds3+Hand5+Stack2+Equity8+Street4)
        # Current: 6 dims [alive, call, aggr, facing_bet, bet_to_pot, can_raise]
        # Legacy:  3 dims [alive, call, aggr]
        # Drop indices 33,34,35 to convert current→legacy
        self._gs_start = 30  # game_state block start
        self._gs_extra = 3   # extra dims in current vs legacy

        sup_name = Path(supervised_path).stem
        cfr_name = Path(cfr_path).stem
        print(f"  Blended model: fold from {cfr_name}, call/raise from {sup_name}")
        print(f"  Fold blend: {fold_blend}")
        if legacy_cfr:
            print(f"  CFR model uses legacy obs ({self.cfr_model.advantage_networks[0].network[0].in_features} dims)")

    def get_strategy(self, observation: np.ndarray, player_idx: int,
                     legal_actions: List[int]) -> np.ndarray:
        """Blend: fold prob from CFR, call/raise distribution from supervised.

        Logic:
        - fold_blend controls how much CFR fold to mix in (0.0=none, 1.0=full)
        - CFR fold is taken directly — no gating on supervised fold
        - Final fold = sup_fold + fold_blend * cfr_fold * (1 - sup_fold)
        - Call/raise distribution comes entirely from supervised model
        """

        # Get both strategies
        sup_strategy = self._get_model_strategy(self.sup_model, observation, legal_actions)

        # Convert observation for CFR model if it uses legacy format
        if self.legacy_cfr:
            cfr_obs = self._to_legacy_obs(observation)
        else:
            cfr_obs = observation
        cfr_strategy = self.cfr_model.get_strategy(cfr_obs, player_idx, legal_actions)

        # If fold isn't legal, just use supervised
        if FOLD not in legal_actions:
            return sup_strategy

        cfr_fold = cfr_strategy[FOLD]
        sup_fold = sup_strategy[FOLD]

        # Direct blend: CFR fold fills the gap between supervised fold and 1.0
        # fold_blend=0.5 means CFR fold contributes 50% of its signal
        fold_prob = sup_fold + self.fold_blend * cfr_fold * (1.0 - sup_fold)
        fold_prob = min(fold_prob, 0.95)  # never fold more than 95%

        # Distribute remaining probability using supervised call/raise distribution
        remaining_mass = 1.0 - fold_prob
        non_fold_strategy = sup_strategy.copy()
        non_fold_strategy[FOLD] = 0.0
        non_fold_sum = non_fold_strategy.sum()

        result = np.zeros(self.action_size, dtype=np.float32)
        if non_fold_sum > 0:
            result = non_fold_strategy * (remaining_mass / non_fold_sum)
        else:
            return cfr_strategy

        result[FOLD] = fold_prob

        # Zero out illegal actions and renormalize
        legal_mask = np.zeros(self.action_size, dtype=np.float32)
        legal_mask[legal_actions] = 1.0
        result *= legal_mask
        s = result.sum()
        if s > 0:
            result /= s
        else:
            result[legal_actions] = 1.0 / len(legal_actions)

        return result

    def _to_legacy_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert current 41-dim obs to legacy 38-dim by dropping extra game_state dims.

        Removes indices 33-35 (facing_bet, bet_to_pot, can_raise).
        """
        # Keep: 0-32 (pre-gs + 3 legacy gs dims) + 36-40 (betting history)
        return np.concatenate([obs[:self._gs_start + 3],
                               obs[self._gs_start + 6:]])

    def _get_model_strategy(self, model: StrategyNetwork,
                            observation: np.ndarray,
                            legal_actions: List[int]) -> np.ndarray:
        """Get strategy from a StrategyNetwork with masked softmax."""
        obs_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(obs_t).cpu().numpy()[0]

        masked = np.full(self.action_size, -1e9, dtype=np.float32)
        for a in legal_actions:
            masked[a] = logits[a]
        masked -= masked.max()
        exp_s = np.exp(masked)
        strategy = exp_s / exp_s.sum()

        legal_mask = np.zeros(self.action_size, dtype=np.float32)
        legal_mask[legal_actions] = 1.0
        strategy *= legal_mask
        s = strategy.sum()
        if s > 0:
            strategy /= s
        else:
            strategy[legal_actions] = 1.0 / len(legal_actions)
        return strategy
