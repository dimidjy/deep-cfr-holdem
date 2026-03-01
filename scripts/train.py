import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine import PokerEngine
from src.training import DeepCFR, CFRTrainer
from src.utils import load_config, TrainingLogger


CONFIG_FILE = "configs/training_config.json"


def main():
    config_path = Path(__file__).parent.parent / CONFIG_FILE
    config = load_config(str(config_path))
    
    print("="*60)
    print("DEEP CFR POKER TRAINING")
    print("="*60)
    print(f"Configuration: {CONFIG_FILE}")
    print(f"Players: {config['game']['num_players']}")
    print(f"Iterations: {config['training']['num_iterations']}")
    print(f"Traversals per iteration: {config['training']['traversals_per_iteration']}")
    print("="*60)
    
    engine = PokerEngine(num_players=config['game']['num_players'])
    
    observation_size = engine.get_observation_space_size()
    action_size = engine.get_action_space_size()
    
    print(f"\nObservation size: {observation_size}")
    print(f"Action size: {action_size}")
    
    deep_cfr = DeepCFR(
        observation_size=observation_size,
        action_size=action_size,
        num_players=config['game']['num_players'],
        hidden_layers=config['network']['hidden_layers'],
        learning_rate=config['training']['learning_rate'],
        buffer_size=config['training']['buffer_size']
    )
    
    trainer = CFRTrainer(engine, deep_cfr, config['training'])
    
    logger = TrainingLogger(config['paths']['logs'])
    
    print("\nStarting training...")
    
    trainer.train(
        num_iterations=config['training']['num_iterations'],
        checkpoint_interval=config['training']['checkpoint_interval']
    )
    
    logger.log_iteration(trainer.iteration, {'buffer_size': len(deep_cfr.replay_buffer)})
    
    print("\nTraining completed!")
    
    print("\nEvaluating final model...")
    eval_results = trainer.evaluate(num_games=100)
    
    logger.log_evaluation(trainer.iteration, eval_results)
    
    print("\nEvaluation Results:")
    print(f"Games played: {eval_results['total_games']}")
    for player_idx in range(config['game']['num_players']):
        avg_reward = eval_results[f'player_{player_idx}_avg_reward']
        std_reward = eval_results[f'player_{player_idx}_std_reward']
        print(f"Player {player_idx}: {avg_reward:.2f} ± {std_reward:.2f}")
    
    final_model_path = Path(config['paths']['models']) / 'final_model.pt'
    trainer.save_checkpoint(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")
    
    engine.close()


if __name__ == "__main__":
    main()
