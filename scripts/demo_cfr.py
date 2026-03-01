import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine import PokerEngine
from src.training import DeepCFR, CFRTrainer


NUM_ITERATIONS = 5
TRAVERSALS_PER_ITERATION = 50


def main():
    print("="*60)
    print("DEEP CFR DEMO")
    print("="*60)
    
    engine = PokerEngine(num_players=2)
    
    observation_size = engine.get_observation_space_size()
    action_size = engine.get_action_space_size()
    
    print(f"\nObservation size: {observation_size}")
    print(f"Action size: {action_size}")
    
    deep_cfr = DeepCFR(
        observation_size=observation_size,
        action_size=action_size,
        num_players=2,
        hidden_layers=[128, 128],
        learning_rate=0.001,
        buffer_size=10000
    )
    
    config = {
        'traversals_per_iteration': TRAVERSALS_PER_ITERATION,
        'batch_size': 32
    }
    
    trainer = CFRTrainer(engine, deep_cfr, config)
    
    print(f"\nRunning {NUM_ITERATIONS} training iterations...")
    print(f"Traversals per iteration: {TRAVERSALS_PER_ITERATION}")
    
    for iteration in range(NUM_ITERATIONS):
        trainer.train_iteration(traversals_per_iteration=TRAVERSALS_PER_ITERATION)
        
        buffer_size = len(deep_cfr.replay_buffer)
        print(f"Iteration {iteration + 1}/{NUM_ITERATIONS} - Buffer size: {buffer_size}")
    
    print("\nEvaluating trained model...")
    results = trainer.evaluate(num_games=20)
    
    print("\nEvaluation Results:")
    print(f"Games played: {results['total_games']}")
    for player_idx in range(2):
        avg_reward = results[f'player_{player_idx}_avg_reward']
        std_reward = results[f'player_{player_idx}_std_reward']
        print(f"Player {player_idx}: {avg_reward:.2f} ± {std_reward:.2f}")
    
    print("\n" + "="*60)
    print("✓ Demo completed successfully!")
    print("="*60)
    
    engine.close()


if __name__ == "__main__":
    main()
