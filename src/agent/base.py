from abc import ABC, abstractmethod
from typing import List, Tuple

from src.agent.game_state import GameState


class AbstractCore(ABC):

    @abstractmethod
    def configure(self, config: dict) -> None:
        """Load models, set parameters."""

    @abstractmethod
    def decide(self, game_state: GameState) -> Tuple[int, List[float]]:
        """Return (chosen_action_idx, strategy_probabilities)."""

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return metadata about loaded model."""
