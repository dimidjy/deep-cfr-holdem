"""Supervised (StrategyNetwork) model adapter."""

from typing import List

import numpy as np
import torch

from .base import ModelAdapter


class SupervisedAdapter(ModelAdapter):
    """Wraps a supervised StrategyNetwork for inference."""

    def __init__(self, net, obs_size: int, action_size: int):
        self.net = net
        self.obs_size = obs_size
        self.action_size = action_size
        self.iteration = 0

    def get_strategy(self, obs: np.ndarray, player_idx: int,
                     legal_actions: List[int]) -> np.ndarray:
        obs_padded = self._pad_obs(obs)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_padded).unsqueeze(0)
            logits = self.net(obs_t).numpy()[0]
        return self._normalize(logits, legal_actions)

    def _pad_obs(self, obs: np.ndarray) -> np.ndarray:
        if len(obs) == self.obs_size:
            return obs.astype(np.float32)
        if len(obs) > self.obs_size:
            return obs[:self.obs_size].astype(np.float32)
        padded = np.zeros(self.obs_size, dtype=np.float32)
        padded[:len(obs)] = obs
        return padded

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> 'SupervisedAdapter':
        from src.training.deep_cfr import StrategyNetwork

        checkpoint = torch.load(path, map_location='cpu')
        obs_size = checkpoint.get('obs_size', 42)
        action_size = checkpoint.get('action_size', 5)
        hidden = checkpoint.get('hidden_layers', [512, 512, 256])
        dropout = checkpoint.get('dropout', 0.1)

        net = StrategyNetwork(obs_size, action_size, hidden, dropout)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()

        adapter = cls(net, obs_size, action_size)
        adapter.iteration = checkpoint.get('epoch', 0)
        return adapter
