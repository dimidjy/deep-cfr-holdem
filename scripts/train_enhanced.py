import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine import EnhancedPokerEngine
from src.training import DeepCFR, CFRTrainer
from src.utils import load_config, TrainingLogger


CONFIG_FILE = "configs/training_config.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g. models/checkpoint_iter_400.pt)')
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / CONFIG_FILE
    config = load_config(str(config_path))
    
    print("="*60)
    print("ENHANCED DEEP CFR POKER TRAINING")
    print("="*60)
    print(f"Configuration: {CONFIG_FILE}")
    print(f"Players: {config['game']['num_players']}")
    print(f"Iterations: {config['training']['num_iterations']}")
    print(f"Traversals per iteration: {config['training']['traversals_per_iteration']}")
    print("="*60)
    print("\nENHANCED FEATURES ENABLED:")
    print("  + Explicit position encoding")
    print("  + Pot odds calculation")
    print("  + Richer hand strength (rank, gap, suited, pair)")
    print("  + Stack features (M-ratio, relative stack)")
    print("\nTRAINING IMPROVEMENTS:")
    print("  + Fixed counterfactual regret computation")
    print("  + Prioritized replay buffer with linear CFR weighting")
    print("  + Residual networks with BatchNorm + LeakyReLU")
    print("  + Gradient clipping + cosine LR schedule")
    print("  + 20 training passes per iteration")
    print("="*60)
    
    engine = EnhancedPokerEngine(
        num_players=config['game']['num_players'],
        add_position_features=True,
        add_pot_odds_features=True,
        add_hand_strength_features=True,
        add_stack_features=True,
    )

    observation_size = engine.get_observation_space_size()
    action_size = engine.get_action_space_size()
    num_players = config['game']['num_players']

    print(f"\nObservation size: {observation_size}")
    print(f"  Base observation: 54")
    print(f"  Position features: {num_players + 2}")
    print(f"  Pot odds features: 3")
    print(f"  Hand strength features: 5")
    print(f"  Stack features: 2")
    print(f"Action size: {action_size}")

    deep_cfr = DeepCFR(
        observation_size=observation_size,
        action_size=action_size,
        num_players=num_players,
        hidden_layers=config['network']['hidden_layers'],
        learning_rate=config['training']['learning_rate'],
        buffer_size=config['training']['buffer_size'],
        gradient_clip=config['training'].get('gradient_clip', 1.0),
        lr_scheduler_t_max=config['training'].get('lr_scheduler_t_max', 2000),
        dropout=config['network'].get('dropout', 0.1),
    )
    
    trainer = CFRTrainer(engine, deep_cfr, config['training'])

    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"\nResumed from: {args.resume} (iteration {trainer.iteration})")

    logger = TrainingLogger(config['paths']['logs'])

    remaining = config['training']['num_iterations'] - trainer.iteration
    print(f"\nStarting enhanced training... ({remaining} iterations remaining)")
    print("Expected benefits:")
    print("  - Correct regret signal drives meaningful learning")
    print("  - Richer features enable better decision-making")
    print("  - Stable gradients through BatchNorm + clipping")

    trainer.train(
        num_iterations=remaining,
        checkpoint_interval=config['training']['checkpoint_interval']
    )
    
    logger.log_iteration(trainer.iteration, {'buffer_size': len(deep_cfr.replay_buffer)})
    
    print("\nTraining completed!")
    
    print("\nEvaluating final model...")
    eval_results = trainer.evaluate(num_games=1000)
    
    logger.log_evaluation(trainer.iteration, eval_results)
    
    print("\nEvaluation Results:")
    print(f"Games played: {eval_results['total_games']}")
    for player_idx in range(config['game']['num_players']):
        avg_reward = eval_results[f'player_{player_idx}_avg_reward']
        std_reward = eval_results[f'player_{player_idx}_std_reward']
        print(f"Player {player_idx}: {avg_reward:.2f} ± {std_reward:.2f}")
    
    final_model_path = Path(config['paths']['models']) / 'final_model_enhanced.pt'
    trainer.save_checkpoint(str(final_model_path))
    print(f"\nEnhanced model saved to: {final_model_path}")
    
    engine.close()


if __name__ == "__main__":
    main()
