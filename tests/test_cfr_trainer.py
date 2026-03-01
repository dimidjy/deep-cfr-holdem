import pytest
import numpy as np
from src.engine import PokerEngine
from src.training import DeepCFR, CFRTrainer


class TestCFRTrainer:
    
    def test_trainer_initialization(self):
        engine = PokerEngine(num_players=2)
        
        observation_size = engine.get_observation_space_size()
        action_size = engine.get_action_space_size()
        
        deep_cfr = DeepCFR(
            observation_size=observation_size,
            action_size=action_size,
            num_players=2
        )
        
        trainer = CFRTrainer(engine, deep_cfr)
        
        assert trainer.engine is not None
        assert trainer.deep_cfr is not None
        assert trainer.iteration == 0
        
        engine.close()
    
    def test_run_traversal(self):
        engine = PokerEngine(num_players=2)
        
        observation_size = engine.get_observation_space_size()
        action_size = engine.get_action_space_size()
        
        deep_cfr = DeepCFR(
            observation_size=observation_size,
            action_size=action_size,
            num_players=2
        )
        
        trainer = CFRTrainer(engine, deep_cfr)
        
        reward = trainer.run_traversal(player_idx=0)
        
        assert isinstance(reward, (int, float))
        assert len(deep_cfr.replay_buffer) > 0
        
        engine.close()
    
    def test_train_iteration(self):
        engine = PokerEngine(num_players=2)
        
        observation_size = engine.get_observation_space_size()
        action_size = engine.get_action_space_size()
        
        deep_cfr = DeepCFR(
            observation_size=observation_size,
            action_size=action_size,
            num_players=2
        )
        
        config = {
            'traversals_per_iteration': 10,
            'batch_size': 32
        }
        
        trainer = CFRTrainer(engine, deep_cfr, config)
        
        initial_iteration = trainer.iteration
        trainer.train_iteration(traversals_per_iteration=10)
        
        assert trainer.iteration == initial_iteration + 1
        assert len(deep_cfr.replay_buffer) > 0
        
        engine.close()
    
    def test_evaluate(self):
        engine = PokerEngine(num_players=2)
        
        observation_size = engine.get_observation_space_size()
        action_size = engine.get_action_space_size()
        
        deep_cfr = DeepCFR(
            observation_size=observation_size,
            action_size=action_size,
            num_players=2
        )
        
        trainer = CFRTrainer(engine, deep_cfr)
        
        results = trainer.evaluate(num_games=5)
        
        assert 'total_games' in results
        assert results['total_games'] == 5
        assert 'player_0_avg_reward' in results
        assert 'player_1_avg_reward' in results
        
        engine.close()
    
    def test_multiple_iterations(self):
        engine = PokerEngine(num_players=2)
        
        observation_size = engine.get_observation_space_size()
        action_size = engine.get_action_space_size()
        
        deep_cfr = DeepCFR(
            observation_size=observation_size,
            action_size=action_size,
            num_players=2,
            hidden_layers=[64, 64]
        )
        
        config = {
            'traversals_per_iteration': 5,
            'batch_size': 16
        }
        
        trainer = CFRTrainer(engine, deep_cfr, config)
        
        for _ in range(3):
            trainer.train_iteration(traversals_per_iteration=5)
        
        assert trainer.iteration == 3
        
        engine.close()
