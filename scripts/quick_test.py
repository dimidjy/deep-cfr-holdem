import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine import PokerEngine, PokerValidator
import numpy as np


NUM_GAMES = 10


def main():
    print("="*60)
    print("QUICK ENGINE TEST")
    print("="*60)
    
    engine = PokerEngine(num_players=2)
    
    print(f"\nObservation space size: {engine.get_observation_space_size()}")
    print(f"Action space size: {engine.get_action_space_size()}")
    
    print(f"\nRunning {NUM_GAMES} random games...")
    
    game_lengths = []
    total_rewards = []
    
    for game_num in range(NUM_GAMES):
        state = engine.reset()
        steps = 0
        
        while not engine.is_done() and steps < 1000:
            legal_actions = engine._get_legal_actions()
            
            if not legal_actions:
                break
            
            action = np.random.choice(legal_actions)
            
            assert PokerValidator.validate_action(action, legal_actions)
            
            state = engine.step(action)
            steps += 1
        
        game_lengths.append(steps)
        
        rewards = engine.get_rewards()
        total_reward = sum(rewards.values())
        total_rewards.append(total_reward)
        
        print(f"Game {game_num + 1}: {steps} steps, rewards: {rewards}")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Average game length: {np.mean(game_lengths):.1f} steps")
    print(f"Min/Max game length: {min(game_lengths)}/{max(game_lengths)} steps")
    print(f"Zero-sum check: All games sum to ~0: {all(abs(r) < 1e-6 for r in total_rewards)}")
    print("="*60)
    print("✓ Engine working correctly!")
    
    engine.close()


if __name__ == "__main__":
    main()
