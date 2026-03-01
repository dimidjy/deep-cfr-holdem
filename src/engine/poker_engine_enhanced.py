import numpy as np
from pettingzoo.classic import texas_holdem_no_limit_v6
from typing import Dict, List, Tuple, Optional, Any
from .poker_engine import PokerEngine


class EnhancedPokerEngine(PokerEngine):
    
    def __init__(self, num_players: int = 2, add_position_features: bool = True,
                 add_pot_odds_features: bool = True, add_hand_strength_features: bool = True):
        super().__init__(num_players)
        self.add_position_features = add_position_features
        self.add_pot_odds_features = add_pot_odds_features
        self.add_hand_strength_features = add_hand_strength_features
        
        self.base_obs_size = super().get_observation_space_size()
        self.enhanced_obs_size = self._calculate_enhanced_size()
    
    def _calculate_enhanced_size(self) -> int:
        size = self.base_obs_size
        
        if self.add_position_features:
            size += self.num_players
            size += 1
            size += 1
        
        if self.add_pot_odds_features:
            size += 1
            size += 1
            size += 1
        
        if self.add_hand_strength_features:
            size += 1
            size += 1
        
        return size
    
    def get_observation_space_size(self) -> int:
        return self.enhanced_obs_size
    
    def _get_position_features(self, current_agent: str) -> np.ndarray:
        features = []
        
        try:
            game = self._get_game()
            if game:
                agent_idx = self.agents.index(current_agent)
                dealer_idx = game.dealer_id
                
                position_one_hot = np.zeros(self.num_players)
                position_one_hot[agent_idx] = 1
                features.extend(position_one_hot)
                
                relative_position = (agent_idx - dealer_idx) % self.num_players
                normalized_position = relative_position / self.num_players
                features.append(normalized_position)
                
                is_button = 1.0 if agent_idx == dealer_idx else 0.0
                features.append(is_button)
        except:
            features = [0.0] * (self.num_players + 2)
        
        return np.array(features, dtype=np.float32)
    
    def _get_pot_odds_features(self, current_agent: str) -> np.ndarray:
        features = []
        
        try:
            game = self._get_game()
            if game:
                agent_idx = self.agents.index(current_agent)
                player = game.players[agent_idx]
                
                pot = sum(p.in_chips for p in game.players)
                my_chips = player.remained_chips
                
                pot_normalized = min(pot / 100.0, 1.0)
                features.append(pot_normalized)
                
                stack_to_pot = my_chips / max(pot, 1)
                stack_to_pot_normalized = min(stack_to_pot, 2.0) / 2.0
                features.append(stack_to_pot_normalized)
                
                if pot > 0:
                    pot_odds = pot / max(pot + my_chips, 1)
                else:
                    pot_odds = 0.0
                features.append(pot_odds)
        except:
            features = [0.0, 0.0, 0.0]
        
        return np.array(features, dtype=np.float32)
    
    def _get_hand_strength_features(self, current_agent: str) -> np.ndarray:
        features = []
        
        try:
            game = self._get_game()
            if game:
                agent_idx = self.agents.index(current_agent)
                player_cards = game.players[agent_idx].hand
                
                if len(player_cards) == 2:
                    ranks = [self._rank_to_value(card.rank) for card in player_cards]
                    suits = [card.suit for card in player_cards]
                    
                    is_pair = 1.0 if ranks[0] == ranks[1] else 0.0
                    features.append(is_pair)
                    
                    is_suited = 1.0 if suits[0] == suits[1] else 0.0
                    features.append(is_suited)
                else:
                    features = [0.0, 0.0]
        except:
            features = [0.0, 0.0]
        
        return np.array(features, dtype=np.float32)
    
    def _rank_to_value(self, rank: str) -> int:
        rank_map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        return rank_map.get(rank, 0)
    
    def get_observation(self) -> np.ndarray:
        base_obs = super().get_observation()
        
        if not self.current_agent:
            return base_obs
        
        enhanced_features = []
        
        if self.add_position_features:
            position_features = self._get_position_features(self.current_agent)
            enhanced_features.extend(position_features)
        
        if self.add_pot_odds_features:
            pot_odds_features = self._get_pot_odds_features(self.current_agent)
            enhanced_features.extend(pot_odds_features)
        
        if self.add_hand_strength_features:
            hand_strength_features = self._get_hand_strength_features(self.current_agent)
            enhanced_features.extend(hand_strength_features)
        
        if enhanced_features:
            enhanced_obs = np.concatenate([base_obs, np.array(enhanced_features, dtype=np.float32)])
            return enhanced_obs
        
        return base_obs
