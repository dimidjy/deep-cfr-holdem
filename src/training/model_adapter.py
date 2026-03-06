"""
Adapter to use models trained with different player counts
"""
import numpy as np
from typing import Dict


class ModelAdapter:
    """Adapts a model trained with N players to work with M players"""
    
    def __init__(self, trained_players: int, target_players: int):
        self.trained_players = trained_players
        self.target_players = target_players
        
        # Calculate observation sizes (no raw card encoding — features only)
        # Layout: position(N+2) + pot_odds(3) + hand_strength(5) + stack(2)
        #         + equity(4) + street(4) + game_state(3) = N+23
        self.trained_obs_size = trained_players + 23
        self.target_obs_size = target_players + 23

    def adapt_observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert observation from target player count to trained player count"""
        if self.trained_players == self.target_players:
            return obs

        # Position features (one-hot encoding)
        pos_start = 0
        pos_end = self.target_players
        target_position = obs[pos_start:pos_end]

        # Find which position is active
        active_pos = np.argmax(target_position) if target_position.sum() > 0 else 0

        # Create new position encoding for trained player count
        trained_position = np.zeros(self.trained_players)
        mapped_pos = int(active_pos * self.trained_players / self.target_players)
        trained_position[mapped_pos] = 1.0

        # Relative position and button flag
        rel_pos = obs[pos_end]
        is_button = obs[pos_end + 1]

        # All remaining features are player-count-independent
        rest = obs[pos_end + 2:]  # pot_odds(3) + hand(5) + stack(2) + equity(4) + street(4) + game_state(2)

        # Combine into adapted observation
        adapted_obs = np.concatenate([
            trained_position,
            [rel_pos, is_button],
            rest,
        ])
        
        return adapted_obs
    
    def get_strategy(self, model, obs: np.ndarray, legal_actions: np.ndarray) -> Dict[int, float]:
        """Get strategy from model with adapted observation"""
        adapted_obs = self.adapt_observation(obs)
        return model.get_strategy(adapted_obs, legal_actions)
