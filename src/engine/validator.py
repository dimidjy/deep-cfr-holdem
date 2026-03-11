import numpy as np
from typing import Dict, List, Any, Optional


class PokerValidator:
    
    @staticmethod
    def validate_action(action: int, legal_actions: List[int]) -> bool:
        return action in legal_actions
    
    @staticmethod
    def validate_observation(observation) -> bool:
        # Fast engine: observation is a numpy array or None
        if isinstance(observation, np.ndarray):
            return True
        if observation is None:
            return True
        # Legacy PettingZoo format: dict with 'observation' and 'action_mask'
        if isinstance(observation, dict):
            required_keys = ['observation', 'action_mask']
            if not all(key in observation for key in required_keys):
                return False
            if not isinstance(observation['observation'], np.ndarray):
                return False
            if not isinstance(observation['action_mask'], np.ndarray):
                return False
            return True
        return False
    
    @staticmethod
    def validate_game_state(state: Dict[str, Any]) -> bool:
        required_keys = ['observation', 'current_player', 'legal_actions']
        
        if not all(key in state for key in required_keys):
            return False
        
        if not state['legal_actions'] and not state.get('termination', False):
            return False
        
        return True
    
    @staticmethod
    def validate_pot_calculation(bets: List[int], expected_pot: int) -> bool:
        return sum(bets) == expected_pot
    
    @staticmethod
    def validate_hand_ranking(cards: List[int]) -> Optional[str]:
        if len(cards) < 5:
            return None
        
        ranks = [card % 13 for card in cards]
        suits = [card // 13 for card in cards]
        
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        counts = sorted(rank_counts.values(), reverse=True)
        is_flush = len(set(suits)) == 1
        
        sorted_ranks = sorted(ranks)
        is_straight = all(sorted_ranks[i] + 1 == sorted_ranks[i+1] for i in range(4))
        is_wheel = sorted_ranks == [0, 1, 2, 3, 12]
        
        if is_straight or is_wheel:
            if is_flush:
                return "straight_flush"
            return "straight"
        
        if is_flush:
            return "flush"
        
        if counts == [4, 1]:
            return "four_of_a_kind"
        if counts == [3, 2]:
            return "full_house"
        if counts == [3, 1, 1]:
            return "three_of_a_kind"
        if counts == [2, 2, 1]:
            return "two_pair"
        if counts == [2, 1, 1, 1]:
            return "pair"
        
        return "high_card"
    
    @staticmethod
    def validate_chips_conservation(initial_chips: Dict[str, int], 
                                   final_chips: Dict[str, int], 
                                   pot: int) -> bool:
        total_initial = sum(initial_chips.values())
        total_final = sum(final_chips.values()) + pot
        
        return abs(total_initial - total_final) < 0.01
