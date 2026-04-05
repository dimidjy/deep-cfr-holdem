"""CQL (Conservative Q-Learning) model adapter."""

from typing import List

import numpy as np
import torch

from .base import ModelAdapter


class CQLAdapter(ModelAdapter):
    """Wraps a CQL QNetwork for inference.

    CQL outputs Q-values per action. Strategy is derived via softmax
    over Q-values of legal actions.
    """

    def __init__(self, q_net, obs_size: int, action_size: int,
                 temperature: float = 1.0, exclude_fold: bool = False):
        self.q_net = q_net
        self.obs_size = obs_size
        self.action_size = action_size
        self.temperature = temperature
        self.exclude_fold = exclude_fold
        self.iteration = 0

    def get_strategy(self, obs: np.ndarray, player_idx: int,
                     legal_actions: List[int]) -> np.ndarray:
        obs_padded = self._pad_obs(obs)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_padded).unsqueeze(0)
            q_values = self.q_net(obs_t).numpy()[0]

        legal = list(legal_actions)
        if self.exclude_fold and 0 in legal and len(legal) > 1:
            legal = [a for a in legal if a != 0]

        q_legal = q_values[legal]
        if self.temperature != 1.0 and self.temperature > 0:
            q_legal = q_legal / self.temperature

        # Softmax
        q_legal = q_legal - q_legal.max()
        exp_q = np.exp(q_legal)
        probs = exp_q / exp_q.sum()

        strategy = np.zeros(self.action_size, dtype=np.float32)
        strategy[legal] = probs
        return strategy

    def _pad_obs(self, obs: np.ndarray) -> np.ndarray:
        if len(obs) == self.obs_size:
            return obs.astype(np.float32)
        if len(obs) > self.obs_size:
            return obs[:self.obs_size].astype(np.float32)
        padded = np.zeros(self.obs_size, dtype=np.float32)
        padded[:len(obs)] = obs
        return padded

    @classmethod
    def from_checkpoint(cls, path: str, exclude_fold: bool = False,
                        **kwargs) -> 'CQLAdapter':
        from scripts.train_cql import QNetwork

        checkpoint = torch.load(path, map_location='cpu')
        obs_size = checkpoint['obs_size']
        action_size = checkpoint['action_size']
        hidden = checkpoint.get('hidden_layers', [512, 512, 256])
        dropout = checkpoint.get('dropout', 0.1)
        temperature = checkpoint.get('temperature', 1.0)

        q_net = QNetwork(obs_size, action_size, hidden, dropout)
        q_net.load_state_dict(checkpoint['model_state_dict'])
        q_net.eval()

        adapter = cls(q_net, obs_size, action_size, temperature, exclude_fold)
        adapter.iteration = checkpoint.get('epoch', 0)
        return adapter
