import pytest
import torch
import numpy as np
from src.training import DeepCFR, ReplayBuffer, AdvantageNetwork, StrategyNetwork


class TestDeepCFR:
    
    def test_deep_cfr_initialization(self):
        observation_size = 72
        action_size = 6
        num_players = 2
        
        deep_cfr = DeepCFR(
            observation_size=observation_size,
            action_size=action_size,
            num_players=num_players
        )
        
        assert len(deep_cfr.advantage_networks) == num_players
        assert len(deep_cfr.strategy_networks) == num_players
        assert deep_cfr.observation_size == observation_size
        assert deep_cfr.action_size == action_size
    
    def test_advantage_network_forward(self):
        input_size = 72
        output_size = 6
        hidden_layers = [128, 128]
        
        network = AdvantageNetwork(input_size, output_size, hidden_layers)
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, input_size)
        
        output = network(input_tensor)
        
        assert output.shape == (batch_size, output_size)
    
    def test_strategy_network_forward(self):
        input_size = 72
        output_size = 6
        hidden_layers = [128, 128]
        
        network = StrategyNetwork(input_size, output_size, hidden_layers)
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, input_size)
        
        output = network(input_tensor)
        
        # The network returns logits; masking + softmax happen in DeepCFR.
        assert output.shape == (batch_size, output_size)
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    
    def test_replay_buffer_add(self):
        buffer = ReplayBuffer(capacity=1000, obs_size=72, act_size=6)
        
        state = np.random.randn(72)
        action = 1
        regret = 0.5
        strategy = np.random.rand(6)
        player = 0
        
        buffer.add(state, action, regret, strategy, player)
        
        assert len(buffer) == 1
    
    def test_replay_buffer_capacity(self):
        capacity = 10
        buffer = ReplayBuffer(capacity=capacity, obs_size=72, act_size=6)
        
        for i in range(20):
            buffer.add(np.random.randn(72), i % 6, 0.5, np.random.rand(6), 0)
        
        assert len(buffer) == capacity
    
    def test_replay_buffer_sample(self):
        buffer = ReplayBuffer(capacity=1000, obs_size=72, act_size=6)
        
        for i in range(100):
            buffer.add(np.random.randn(72), i % 6, 0.5, np.random.rand(6), i % 2)
        
        batch = buffer.sample(32)
        
        assert len(batch) == 32
        assert all('state' in sample for sample in batch)
        assert all('action' in sample for sample in batch)
    
    def test_get_strategy(self):
        deep_cfr = DeepCFR(
            observation_size=72,
            action_size=6,
            num_players=2
        )
        
        observation = np.random.randn(72)
        legal_actions = [0, 1, 2]
        player_idx = 0
        
        strategy = deep_cfr.get_strategy(observation, player_idx, legal_actions)
        
        assert strategy.shape == (6,)
        assert np.isclose(strategy.sum(), 1.0)
        assert all(strategy[i] > 0 for i in legal_actions)
    
    def test_get_regret_matching_strategy(self):
        deep_cfr = DeepCFR(
            observation_size=72,
            action_size=6,
            num_players=2
        )
        
        regrets = np.array([1.0, -0.5, 2.0, 0.0, -1.0, 0.5])
        legal_actions = [0, 1, 2, 3, 4, 5]
        
        strategy = deep_cfr.get_regret_matching_strategy(regrets, legal_actions)
        
        assert strategy.shape == (6,)
        assert np.isclose(strategy.sum(), 1.0)
        assert all(strategy[i] >= 0 for i in legal_actions)
    
    def test_compute_counterfactual_values(self):
        deep_cfr = DeepCFR(
            observation_size=72,
            action_size=6,
            num_players=2
        )
        
        observation = np.random.randn(72)
        legal_actions = [0, 1, 2]
        player_idx = 0
        
        advantages = deep_cfr.compute_counterfactual_values(observation, player_idx, legal_actions)
        
        assert advantages.shape == (6,)
        assert all(advantages[i] == 0 for i in range(6) if i not in legal_actions)
    
    def test_train_advantage_network(self):
        deep_cfr = DeepCFR(
            observation_size=72,
            action_size=6,
            num_players=2
        )
        
        for _ in range(150):
            state = np.random.randn(72)
            action = np.random.randint(0, 6)
            regret = np.random.randn()
            strategy = np.random.rand(6)
            strategy = strategy / strategy.sum()
            player = np.random.randint(0, 2)
            
            deep_cfr.replay_buffer.add(state, action, regret, strategy, player)
        
        deep_cfr.train_advantage_network(batch_size=32)
    
    def test_train_strategy_network(self):
        deep_cfr = DeepCFR(
            observation_size=72,
            action_size=6,
            num_players=2
        )
        
        for _ in range(150):
            state = np.random.randn(72)
            action = np.random.randint(0, 6)
            regret = np.random.randn()
            strategy = np.random.rand(6)
            strategy = strategy / strategy.sum()
            player = np.random.randint(0, 2)
            
            deep_cfr.replay_buffer.add(state, action, regret, strategy, player)
        
        deep_cfr.train_strategy_network(batch_size=32)
