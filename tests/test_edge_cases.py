import pytest
import numpy as np
from src.engine import PokerEngine, PokerValidator


class TestEdgeCases:
    
    def test_all_in_scenario(self):
        engine = PokerEngine(num_players=2)
        engine.reset()
        
        steps = 0
        max_steps = 500
        
        while not engine.is_done() and steps < max_steps:
            legal_actions = engine._get_legal_actions()
            if not legal_actions:
                break
            
            raise_actions = [a for a in legal_actions if a >= 2]
            if raise_actions:
                action = raise_actions[-1]
            else:
                action = legal_actions[0]
            
            engine.step(action)
            steps += 1
        
        assert steps < max_steps
        engine.close()
    
    def test_consecutive_folds(self):
        engine = PokerEngine(num_players=2)
        
        for _ in range(5):
            engine.reset()
            
            legal_actions = engine._get_legal_actions()
            if 0 in legal_actions:
                result = engine.step(0)
                
                if engine.is_done():
                    rewards = engine.get_rewards()
                    assert len(rewards) == 2
        
        engine.close()
    
    def test_check_check_scenario(self):
        engine = PokerEngine(num_players=2)
        engine.reset()
        
        steps = 0
        check_count = 0
        
        while not engine.is_done() and steps < 100:
            legal_actions = engine._get_legal_actions()
            if not legal_actions:
                break
            
            if 1 in legal_actions:
                engine.step(1)
                check_count += 1
            else:
                engine.step(legal_actions[0])
            
            steps += 1
        
        engine.close()
    
    def test_rapid_game_cycles(self):
        engine = PokerEngine(num_players=2)
        
        for game_idx in range(20):
            state = engine.reset()
            assert PokerValidator.validate_game_state(state)
            
            steps = 0
            while not engine.is_done() and steps < 100:
                legal_actions = engine._get_legal_actions()
                if not legal_actions:
                    break
                
                action = np.random.choice(legal_actions)
                engine.step(action)
                steps += 1
        
        engine.close()
    
    def test_reward_sum_zero_sum_property(self):
        engine = PokerEngine(num_players=2)
        
        for _ in range(5):
            engine.reset()
            
            while not engine.is_done():
                legal_actions = engine._get_legal_actions()
                if not legal_actions:
                    break
                
                action = np.random.choice(legal_actions)
                engine.step(action)
            
            rewards = engine.get_rewards()
            total_reward = sum(rewards.values())
            
            assert abs(total_reward) < 1e-6
        
        engine.close()
    
    def test_invalid_action_handling(self):
        engine = PokerEngine(num_players=2)
        engine.reset()
        
        legal_actions = engine._get_legal_actions()
        invalid_action = max(legal_actions) + 10
        
        assert not PokerValidator.validate_action(invalid_action, legal_actions)
    
    def test_observation_contains_action_mask(self):
        engine = PokerEngine(num_players=2)
        state = engine.reset()
        
        observation = state['observation']
        assert 'action_mask' in observation
        assert isinstance(observation['action_mask'], np.ndarray)
        
        engine.close()
