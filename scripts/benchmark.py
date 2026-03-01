import sys
import os
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine import PokerEngine


NUM_GAMES = 1000


def main():
    print("="*60)
    print("POKER ENGINE BENCHMARK")
    print("="*60)
    
    engine = PokerEngine(num_players=2)
    
    print(f"\nRunning {NUM_GAMES} games...")
    
    start_time = time.time()
    
    total_steps = 0
    game_times = []
    
    for game_num in range(NUM_GAMES):
        game_start = time.time()
        
        engine.reset()
        steps = 0
        
        while not engine.is_done() and steps < 1000:
            legal_actions = engine._get_legal_actions()
            if not legal_actions:
                break
            
            action = np.random.choice(legal_actions)
            engine.step(action)
            steps += 1
        
        total_steps += steps
        game_times.append(time.time() - game_start)
        
        if (game_num + 1) % 100 == 0:
            elapsed = time.time() - start_time
            games_per_sec = (game_num + 1) / elapsed
            print(f"  {game_num + 1}/{NUM_GAMES} games - {games_per_sec:.1f} games/sec")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Total games: {NUM_GAMES}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Games per second: {NUM_GAMES / total_time:.2f}")
    print(f"Average game time: {np.mean(game_times)*1000:.2f} ms")
    print(f"Total steps: {total_steps}")
    print(f"Steps per second: {total_steps / total_time:.2f}")
    print(f"Average steps per game: {total_steps / NUM_GAMES:.2f}")
    print("="*60)
    
    engine.close()


if __name__ == "__main__":
    main()
