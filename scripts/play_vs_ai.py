import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine import PokerEngine
from src.training import DeepCFR


MODEL_PATH = "models/final_model.pt"


def get_human_action(legal_actions):
    action_names = {
        0: "Fold",
        1: "Check/Call",
        2: "Raise Half Pot",
        3: "Raise Full Pot",
        4: "All-In"
    }
    
    print("\nLegal actions:")
    for idx, action in enumerate(legal_actions):
        print(f"  {idx}: {action_names.get(action, f'Action {action}')}")
    
    while True:
        try:
            choice = int(input("Choose action index: "))
            if 0 <= choice < len(legal_actions):
                return legal_actions[choice]
            else:
                print("Invalid choice. Try again.")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(0)


def main():
    print("="*60)
    print("PLAY VS AI")
    print("="*60)
    
    model_path = Path(__file__).parent.parent / MODEL_PATH
    
    engine = PokerEngine(num_players=2)
    
    observation_size = engine.get_observation_space_size()
    action_size = engine.get_action_space_size()
    
    deep_cfr = DeepCFR(
        observation_size=observation_size,
        action_size=action_size,
        num_players=2
    )
    
    if model_path.exists():
        print(f"Loading AI model from: {model_path}")
        deep_cfr.load_models(str(model_path))
    else:
        print("No trained model found. AI will play randomly.")
    
    print("\nYou are Player 0, AI is Player 1")
    print("="*60)
    
    state = engine.reset()
    
    while not engine.is_done():
        current_agent = state['current_player']
        agent_idx = engine.agents.index(current_agent)
        
        legal_actions = state['legal_actions']
        
        if not legal_actions:
            break
        
        print(f"\n--- {current_agent}'s turn ---")
        
        if agent_idx == 0:
            action = get_human_action(legal_actions)
        else:
            observation = engine.get_observation()
            
            if model_path.exists():
                strategy = deep_cfr.get_strategy(observation, agent_idx, legal_actions)
                action = np.random.choice(legal_actions, p=strategy[legal_actions] / strategy[legal_actions].sum())
            else:
                action = np.random.choice(legal_actions)
            
            print(f"AI chooses action: {action}")
        
        state = engine.step(action)
    
    print("\n" + "="*60)
    print("GAME OVER")
    print("="*60)
    
    rewards = engine.get_rewards()
    for idx, agent in enumerate(engine.agents):
        result = "WON" if rewards[agent] > 0 else "LOST" if rewards[agent] < 0 else "TIED"
        print(f"{agent}: {result} (Reward: {rewards[agent]:.2f})")
    
    print("="*60)
    
    engine.close()


if __name__ == "__main__":
    main()
