import numpy as np
from typing import List, Dict, Any


def compute_regret_matching_strategy(regrets: np.ndarray, 
                                     legal_actions: List[int],
                                     action_size: int) -> np.ndarray:
    
    positive_regrets = np.maximum(regrets, 0)
    
    legal_mask = np.zeros(action_size)
    legal_mask[legal_actions] = 1
    positive_regrets = positive_regrets * legal_mask
    
    regret_sum = positive_regrets.sum()
    
    if regret_sum > 0:
        strategy = positive_regrets / regret_sum
    else:
        strategy = legal_mask / legal_mask.sum()
    
    return strategy


def compute_exploitability(strategy_profile: Dict[int, np.ndarray],
                          game_values: Dict[int, float]) -> float:
    
    total_exploitability = 0.0
    
    for player_idx, value in game_values.items():
        total_exploitability += abs(value)
    
    return total_exploitability / len(game_values)


def normalize_strategy(strategy: np.ndarray, legal_actions: List[int]) -> np.ndarray:
    
    normalized = strategy.copy()
    
    legal_sum = sum(strategy[action] for action in legal_actions)
    
    if legal_sum > 0:
        for action in legal_actions:
            normalized[action] = strategy[action] / legal_sum
    else:
        uniform_prob = 1.0 / len(legal_actions)
        for action in legal_actions:
            normalized[action] = uniform_prob
    
    return normalized


def compute_best_response_value(opponent_strategy: np.ndarray,
                               payoff_matrix: np.ndarray) -> float:
    
    expected_values = payoff_matrix @ opponent_strategy
    
    return np.max(expected_values)


def sample_action_from_strategy(strategy: np.ndarray, 
                               legal_actions: List[int]) -> int:
    
    legal_probs = strategy[legal_actions]
    legal_probs_sum = legal_probs.sum()
    
    if legal_probs_sum > 0:
        legal_probs = legal_probs / legal_probs_sum
    else:
        legal_probs = np.ones(len(legal_actions)) / len(legal_actions)
    
    return np.random.choice(legal_actions, p=legal_probs)


def compute_strategy_entropy(strategy: np.ndarray) -> float:
    
    probs = strategy[strategy > 0]
    
    if len(probs) == 0:
        return 0.0
    
    return -np.sum(probs * np.log(probs + 1e-10))
