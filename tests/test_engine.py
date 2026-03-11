import pytest
import numpy as np
from src.engine import PokerEngine, PokerValidator


class TestPokerEngine:
    
    def test_engine_initialization(self):
        engine = PokerEngine(num_players=2)
        assert engine is not None
        assert engine.num_players == 2
        assert len(engine.agents) == 2
    
    def test_reset(self):
        engine = PokerEngine(num_players=2)
        state = engine.reset()
        
        assert 'observation' in state
        assert 'legal_actions' in state
        assert 'current_player' in state
        assert len(state['legal_actions']) > 0
    
    def test_legal_actions_exist(self):
        engine = PokerEngine(num_players=2)
        engine.reset()
        
        legal_actions = engine.get_legal_actions()
        assert len(legal_actions) > 0
        assert all(isinstance(action, int) for action in legal_actions)
    
    def test_step_with_legal_action(self):
        engine = PokerEngine(num_players=2)
        engine.reset()
        
        legal_actions = engine.get_legal_actions()
        action = legal_actions[0]
        
        result = engine.step(action)
        
        assert 'observation' in result
        assert 'reward' in result
        assert 'done' in result
        assert 'legal_actions' in result
    
    def test_game_completion(self):
        engine = PokerEngine(num_players=2)
        engine.reset()
        
        max_steps = 1000
        steps = 0
        
        while not engine.is_done() and steps < max_steps:
            legal_actions = engine.get_legal_actions()
            if not legal_actions:
                break
            
            action = np.random.choice(legal_actions)
            engine.step(action)
            steps += 1
        
        assert steps < max_steps
    
    def test_observation_shape_consistency(self):
        engine = PokerEngine(num_players=2)
        engine.reset()
        
        obs1 = engine.get_observation()
        shape1 = obs1.shape
        
        legal_actions = engine.get_legal_actions()
        if legal_actions:
            engine.step(legal_actions[0])
            obs2 = engine.get_observation()
            shape2 = obs2.shape
            
            assert shape1 == shape2
    
    def test_action_space_size(self):
        engine = PokerEngine(num_players=2)
        engine.reset()
        
        action_space_size = engine.get_action_space_size()
        assert action_space_size > 0
    
    def test_observation_space_size(self):
        engine = PokerEngine(num_players=2)
        engine.reset()
        
        obs_space_size = engine.get_observation_space_size()
        assert obs_space_size > 0
    
    def test_rewards_structure(self):
        engine = PokerEngine(num_players=2)
        engine.reset()
        
        rewards = engine.get_rewards()
        assert len(rewards) == engine.num_players
        assert all(isinstance(r, (int, float)) for r in rewards.values())
    
    def test_multiple_games(self):
        engine = PokerEngine(num_players=2)
        
        for _ in range(5):
            engine.reset()
            
            while not engine.is_done():
                legal_actions = engine.get_legal_actions()
                if not legal_actions:
                    break
                action = np.random.choice(legal_actions)
                engine.step(action)
            
            rewards = engine.get_rewards()
            assert len(rewards) == 2
