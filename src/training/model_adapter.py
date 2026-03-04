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
        
        # Calculate observation sizes
        # Base: 52 (cards) + 2 (chips/pot) = 54
        # Position features: N (one-hot) + 2 (relative pos, is button)
        # Pot odds: 3
        # Hand strength: 5 (is_pair, is_suited, high_rank, low_rank, rank_gap)
        # Stack features: 2 (m_ratio, relative_stack)
        self.trained_obs_size = 54 + trained_players + 2 + 3 + 5 + 2
        self.target_obs_size = 54 + target_players + 2 + 3 + 5 + 2
        
    def adapt_observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert observation from target player count to trained player count"""
        if self.trained_players == self.target_players:
            return obs
        
        # Split observation into components
        base_obs = obs[:54]  # Cards, chips, pot
        
        # Position features (one-hot encoding)
        pos_start = 54
        pos_end = 54 + self.target_players
        target_position = obs[pos_start:pos_end]
        
        # Find which position is active
        active_pos = np.argmax(target_position) if target_position.sum() > 0 else 0
        
        # Create new position encoding for trained player count
        trained_position = np.zeros(self.trained_players)
        # Map position proportionally
        mapped_pos = int(active_pos * self.trained_players / self.target_players)
        trained_position[mapped_pos] = 1.0
        
        # Relative position and button flag
        rel_pos = obs[pos_end]
        is_button = obs[pos_end + 1]
        
        # Pot odds features (same for any player count)
        pot_start = pos_end + 2
        pot_features = obs[pot_start:pot_start + 3]

        # Hand strength features (5 dims: is_pair, is_suited, high_rank, low_rank, rank_gap)
        hand_features = obs[pot_start + 3:pot_start + 8]

        # Stack features (2 dims: m_ratio, relative_stack)
        stack_features = obs[pot_start + 8:pot_start + 10]

        # Combine into adapted observation
        adapted_obs = np.concatenate([
            base_obs,
            trained_position,
            [rel_pos, is_button],
            pot_features,
            hand_features,
            stack_features
        ])
        
        return adapted_obs
    
    def get_strategy(self, model, obs: np.ndarray, legal_actions: np.ndarray) -> Dict[int, float]:
        """Get strategy from model with adapted observation"""
        adapted_obs = self.adapt_observation(obs)
        return model.get_strategy(adapted_obs, legal_actions)
