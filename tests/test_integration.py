import pytest
import numpy as np
from src.engine import PokerEngine, PokerValidator


class TestIntegration:
    
    def test_full_game_flow(self):
        engine = PokerEngine(num_players=2)
        state = engine.reset()
        
        assert PokerValidator.validate_game_state(state)
        
        step_count = 0
        max_steps = 1000
        
        while not engine.is_done() and step_count < max_steps:
            legal_actions = state['legal_actions']
            
            if not legal_actions:
                break
            
            assert PokerValidator.validate_action(legal_actions[0], legal_actions)
            
            action = np.random.choice(legal_actions)
            state = engine.step(action)
            
            step_count += 1
        
        assert step_count < max_steps
        
        rewards = engine.get_rewards()
        assert len(rewards) == 2
        
        engine.close()
    
    def test_multiple_games_consistency(self):
        engine = PokerEngine(num_players=2)
        
        game_lengths = []
        
        for game_num in range(10):
            engine.reset()
            steps = 0
            
            while not engine.is_done() and steps < 1000:
                legal_actions = engine._get_legal_actions()
                if not legal_actions:
                    break
                
                action = np.random.choice(legal_actions)
                engine.step(action)
                steps += 1
            
            game_lengths.append(steps)
        
        assert all(length < 1000 for length in game_lengths)
        assert len(set(game_lengths)) > 1
        
        engine.close()
    
    def test_all_fold_scenario(self):
        engine = PokerEngine(num_players=2)
        engine.reset()
        
        legal_actions = engine._get_legal_actions()
        
        fold_action = None
        for action in legal_actions:
            if action == 0:
                fold_action = action
                break
        
        if fold_action is not None:
            result = engine.step(fold_action)
            assert 'reward' in result
    
    def test_observation_validation_during_game(self):
        engine = PokerEngine(num_players=2)
        state = engine.reset()
        
        steps = 0
        while not engine.is_done() and steps < 100:
            assert PokerValidator.validate_observation(state['observation'])
            
            legal_actions = state['legal_actions']
            if not legal_actions:
                break
            
            action = np.random.choice(legal_actions)
            state = engine.step(action)
            steps += 1
        
        engine.close()
    
    def test_action_space_consistency(self):
        engine = PokerEngine(num_players=2)
        engine.reset()
        
        action_space_size = engine.get_action_space_size()
        
        for _ in range(10):
            legal_actions = engine._get_legal_actions()
            if not legal_actions:
                break
            
            assert all(0 <= action < action_space_size for action in legal_actions)
            
            action = np.random.choice(legal_actions)
            engine.step(action)
            
            if engine.is_done():
                engine.reset()
        
        engine.close()
    
    def test_state_retrieval(self):
        engine = PokerEngine(num_players=2)
        engine.reset()
        
        state = engine.get_state()
        
        assert 'observation' in state
        assert 'current_player' in state
        assert 'legal_actions' in state
        assert 'agents' in state
        assert PokerValidator.validate_game_state(state)
        
        engine.close()
