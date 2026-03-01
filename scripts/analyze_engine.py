import sys
import os
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine import PokerEngine, PokerValidator


NUM_GAMES = 100


def main():
    print("="*60)
    print("POKER ENGINE ANALYSIS")
    print("="*60)
    
    engine = PokerEngine(num_players=2)
    
    game_lengths = []
    action_distributions = defaultdict(int)
    player_rewards = {agent: [] for agent in engine.agents}
    zero_sum_violations = []
    
    print(f"\nAnalyzing {NUM_GAMES} random games...\n")
    
    for game_num in range(NUM_GAMES):
        engine.reset()
        steps = 0
        game_actions = []
        
        while not engine.is_done() and steps < 1000:
            legal_actions = engine._get_legal_actions()
            if not legal_actions:
                break
            
            action = np.random.choice(legal_actions)
            game_actions.append(action)
            action_distributions[action] += 1
            
            engine.step(action)
            steps += 1
        
        game_lengths.append(steps)
        
        rewards = engine.get_rewards()
        for agent, reward in rewards.items():
            player_rewards[agent].append(reward)
        
        total_reward = sum(rewards.values())
        if abs(total_reward) > 1e-6:
            zero_sum_violations.append((game_num, total_reward))
        
        if (game_num + 1) % 20 == 0:
            print(f"Processed {game_num + 1}/{NUM_GAMES} games...")
    
    print("\n" + "="*60)
    print("GAME LENGTH STATISTICS")
    print("="*60)
    print(f"Average: {np.mean(game_lengths):.2f} steps")
    print(f"Median: {np.median(game_lengths):.2f} steps")
    print(f"Std Dev: {np.std(game_lengths):.2f}")
    print(f"Min: {min(game_lengths)} steps")
    print(f"Max: {max(game_lengths)} steps")
    
    print("\n" + "="*60)
    print("ACTION DISTRIBUTION")
    print("="*60)
    
    action_names = {
        0: "Fold",
        1: "Check/Call",
        2: "Raise Half Pot",
        3: "Raise Full Pot",
        4: "All-In",
        5: "All-In"
    }
    
    total_actions = sum(action_distributions.values())
    for action in sorted(action_distributions.keys()):
        count = action_distributions[action]
        percentage = (count / total_actions) * 100
        name = action_names.get(action, f"Action {action}")
        print(f"{name:20s}: {count:5d} ({percentage:5.1f}%)")
    
    print("\n" + "="*60)
    print("REWARD STATISTICS")
    print("="*60)
    
    for agent, rewards in player_rewards.items():
        wins = sum(1 for r in rewards if r > 0)
        losses = sum(1 for r in rewards if r < 0)
        ties = sum(1 for r in rewards if abs(r) < 1e-6)
        
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        print(f"\n{agent}:")
        print(f"  Wins: {wins} ({wins/NUM_GAMES*100:.1f}%)")
        print(f"  Losses: {losses} ({losses/NUM_GAMES*100:.1f}%)")
        print(f"  Ties: {ties} ({ties/NUM_GAMES*100:.1f}%)")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Std Reward: {std_reward:.2f}")
    
    print("\n" + "="*60)
    print("ZERO-SUM PROPERTY CHECK")
    print("="*60)
    
    if not zero_sum_violations:
        print("✓ All games satisfy zero-sum property")
    else:
        print(f"✗ Found {len(zero_sum_violations)} violations:")
        for game_num, total in zero_sum_violations[:5]:
            print(f"  Game {game_num}: sum = {total:.6f}")
    
    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)
    
    engine.reset()
    state = engine.get_state()
    
    checks = [
        ("Game state validation", PokerValidator.validate_game_state(state)),
        ("Observation validation", PokerValidator.validate_observation(state['observation'])),
        ("Legal actions exist", len(state['legal_actions']) > 0),
        ("Action space size > 0", engine.get_action_space_size() > 0),
        ("Observation space size > 0", engine.get_observation_space_size() > 0)
    ]
    
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"{status} {check_name}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    engine.close()


if __name__ == "__main__":
    main()
