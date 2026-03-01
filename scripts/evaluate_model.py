import sys
import os
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine import PokerEngine
from src.training import DeepCFR, CFRTrainer


MODEL_PATH = "models/final_model.pt"
NUM_GAMES = 100


def main():
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    model_path = Path(__file__).parent.parent / MODEL_PATH
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Please train a model first using: python scripts/train.py")
        return
    
    engine = PokerEngine(num_players=2)
    
    observation_size = engine.get_observation_space_size()
    action_size = engine.get_action_space_size()
    
    deep_cfr = DeepCFR(
        observation_size=observation_size,
        action_size=action_size,
        num_players=2
    )
    
    print(f"\nLoading model from: {model_path}")
    deep_cfr.load_models(str(model_path))
    
    trainer = CFRTrainer(engine, deep_cfr)
    
    print(f"Evaluating over {NUM_GAMES} games...")
    results = trainer.evaluate(num_games=NUM_GAMES)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total games: {results['total_games']}")
    
    for player_idx in range(2):
        avg_reward = results[f'player_{player_idx}_avg_reward']
        std_reward = results[f'player_{player_idx}_std_reward']
        rewards = results['player_rewards'][player_idx]
        
        win_rate = sum(1 for r in rewards if r > 0) / len(rewards) * 100
        
        print(f"\nPlayer {player_idx}:")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Std deviation: {std_reward:.2f}")
        print(f"  Win rate: {win_rate:.1f}%")
    
    print("="*60)
    
    engine.close()


if __name__ == "__main__":
    main()
