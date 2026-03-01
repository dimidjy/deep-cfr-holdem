import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine import PokerEngine, HandHistory


NUM_HANDS = 20
OUTPUT_FILE = "data/hand_histories.json"


def main():
    print("="*60)
    print("RECORDING HAND HISTORIES")
    print("="*60)
    
    engine = PokerEngine(num_players=2)
    hand_history = HandHistory()
    
    print(f"\nRecording {NUM_HANDS} hands...\n")
    
    for hand_num in range(NUM_HANDS):
        hand_history.start_hand()
        
        state = engine.reset()
        steps = 0
        
        while not engine.is_done() and steps < 1000:
            legal_actions = engine._get_legal_actions()
            if not legal_actions:
                break
            
            action = np.random.choice(legal_actions)
            current_player = state['current_player']
            
            hand_history.add_action(action, current_player, state)
            
            state = engine.step(action)
            steps += 1
        
        rewards = engine.get_rewards()
        hand_history.end_hand(rewards)
        
        print(f"Hand {hand_num + 1}/{NUM_HANDS}: {steps} actions, rewards: {rewards}")
    
    output_path = Path(__file__).parent.parent / OUTPUT_FILE
    hand_history.save_to_file(str(output_path))
    
    print(f"\n✓ Saved {NUM_HANDS} hands to {OUTPUT_FILE}")
    print("="*60)
    
    engine.close()


if __name__ == "__main__":
    main()
