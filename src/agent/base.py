from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from src.agent.game_state import GameState


class AbstractCore(ABC):

    @abstractmethod
    def configure(self, config: dict) -> None:
        """Load models, set parameters."""

    @abstractmethod
    def decide(self, game_state: GameState,
               table_id: Optional[int] = None
               ) -> Tuple[int, List[float]]:
        """Return (chosen_action_idx, strategy_probabilities).

        `table_id` lets stateful cores keep separate per-hand state
        per game — e.g. tracking whether hero is the
        last aggressor across streets, which the obs builder cannot
        reconstruct from a single snapshot.
        """

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return metadata about loaded model."""
