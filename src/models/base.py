"""Abstract base class for all model adapters."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class ModelAdapter(ABC):
    """Unified interface for poker model inference.

    All model types (CQL, Deep CFR, Supervised) implement this interface
    so the DecisionEngine doesn't need to know which type it's using.
    """

    obs_size: int
    action_size: int
    iteration: int = 0

    @abstractmethod
    def get_strategy(self, obs: np.ndarray, player_idx: int,
                     legal_actions: List[int]) -> np.ndarray:
        """Return action probability distribution over all actions.

        Args:
            obs: observation vector (will be padded/truncated to obs_size)
            player_idx: player index (used by Deep CFR which has per-player nets)
            legal_actions: list of legal action indices

        Returns:
            numpy array of shape (action_size,) with probabilities summing to 1
        """

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, path: str, **kwargs) -> 'ModelAdapter':
        """Load model from a checkpoint file."""

    @staticmethod
    def _normalize(probs: np.ndarray, legal_actions: List[int]) -> np.ndarray:
        """Zero out illegal actions and normalize to sum to 1."""
        n = len(probs)
        # Expand output if legal_actions reference indices beyond model output
        max_action = max(legal_actions) if legal_actions else 0
        if max_action >= n:
            masked = np.zeros(max_action + 1, dtype=probs.dtype)
            masked[:n] = 0.0  # will be filled below
        else:
            masked = np.zeros_like(probs)
        valid = [a for a in legal_actions if a < n]
        if valid:
            masked[valid] = np.maximum(probs[valid], 0.0)
        s = masked.sum()
        if s > 0:
            masked /= s
        else:
            masked[legal_actions] = 1.0 / len(legal_actions)
        return masked
