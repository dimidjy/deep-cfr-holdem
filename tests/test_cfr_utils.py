import pytest
import numpy as np
from src.training.cfr_utils import (
    compute_regret_matching_strategy,
    normalize_strategy,
    sample_action_from_strategy,
    compute_strategy_entropy
)


class TestCFRUtils:
    
    def test_compute_regret_matching_strategy_positive(self):
        regrets = np.array([1.0, 2.0, 3.0, 0.0])
        legal_actions = [0, 1, 2, 3]
        action_size = 4
        
        strategy = compute_regret_matching_strategy(regrets, legal_actions, action_size)
        
        assert np.isclose(strategy.sum(), 1.0)
        assert all(strategy[i] >= 0 for i in legal_actions)
        assert strategy[2] > strategy[1] > strategy[0]
    
    def test_compute_regret_matching_strategy_negative(self):
        regrets = np.array([-1.0, -2.0, -3.0, -0.5])
        legal_actions = [0, 1, 2, 3]
        action_size = 4
        
        strategy = compute_regret_matching_strategy(regrets, legal_actions, action_size)
        
        assert np.isclose(strategy.sum(), 1.0)
        assert all(strategy[i] >= 0 for i in legal_actions)
    
    def test_compute_regret_matching_strategy_mixed(self):
        regrets = np.array([1.0, -1.0, 2.0, 0.0])
        legal_actions = [0, 1, 2, 3]
        action_size = 4
        
        strategy = compute_regret_matching_strategy(regrets, legal_actions, action_size)
        
        assert np.isclose(strategy.sum(), 1.0)
        assert strategy[1] == 0.0
    
    def test_normalize_strategy(self):
        strategy = np.array([2.0, 3.0, 5.0, 0.0])
        legal_actions = [0, 1, 2]
        
        normalized = normalize_strategy(strategy, legal_actions)
        
        legal_sum = sum(normalized[action] for action in legal_actions)
        assert np.isclose(legal_sum, 1.0)
    
    def test_normalize_strategy_zero_sum(self):
        strategy = np.array([0.0, 0.0, 0.0, 0.0])
        legal_actions = [0, 1, 2]
        
        normalized = normalize_strategy(strategy, legal_actions)
        
        legal_sum = sum(normalized[action] for action in legal_actions)
        assert np.isclose(legal_sum, 1.0)
        assert all(np.isclose(normalized[action], 1.0/3) for action in legal_actions)
    
    def test_sample_action_from_strategy(self):
        strategy = np.array([0.5, 0.3, 0.2, 0.0])
        legal_actions = [0, 1, 2]
        
        samples = [sample_action_from_strategy(strategy, legal_actions) for _ in range(100)]
        
        assert all(action in legal_actions for action in samples)
        assert len(set(samples)) > 1
    
    def test_compute_strategy_entropy_uniform(self):
        strategy = np.array([0.25, 0.25, 0.25, 0.25])
        
        entropy = compute_strategy_entropy(strategy)
        
        expected_entropy = -4 * (0.25 * np.log(0.25))
        assert np.isclose(entropy, expected_entropy, atol=1e-5)
    
    def test_compute_strategy_entropy_deterministic(self):
        strategy = np.array([1.0, 0.0, 0.0, 0.0])
        
        entropy = compute_strategy_entropy(strategy)
        
        assert np.isclose(entropy, 0.0, atol=1e-5)
    
    def test_compute_strategy_entropy_mixed(self):
        strategy = np.array([0.7, 0.2, 0.1, 0.0])
        
        entropy = compute_strategy_entropy(strategy)
        
        assert entropy > 0
        assert entropy < -3 * (1/3 * np.log(1/3))
