import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine import PokerEngine
from src.training import DeepCFR


NUM_GAMES = 50


def random_strategy(legal_actions):
    return np.random.choice(legal_actions)


def aggressive_strategy(legal_actions):
    raise_actions = [a for a in legal_actions if a >= 2]
    if raise_actions:
        return raise_actions[-1]
    elif 1 in legal_actions:
        return 1
    else:
        return legal_actions[0]


def conservative_strategy(legal_actions):
    if 1 in legal_actions:
        return 1
    elif 0 in legal_actions:
        return 0
    else:
        return legal_actions[0]


def play_match(engine, strategy1, strategy2, num_games):
    results = {
        'player_0_wins': 0,
        'player_1_wins': 0,
        'ties': 0,
        'player_0_rewards': [],
        'player_1_rewards': []
    }
    
    for _ in range(num_games):
        engine.reset()
        
        while not engine.is_done():
            current_agent = engine.current_agent
            agent_idx = engine.agents.index(current_agent)
            
            legal_actions = engine._get_legal_actions()
            if not legal_actions:
                break
            
            if agent_idx == 0:
                action = strategy1(legal_actions)
            else:
                action = strategy2(legal_actions)
            
            engine.step(action)
        
        rewards = engine.get_rewards()
        
        for idx, agent in enumerate(engine.agents):
            reward = rewards[agent]
            results[f'player_{idx}_rewards'].append(reward)
            
            if reward > 0:
                results[f'player_{idx}_wins'] += 1
            elif abs(reward) < 1e-6:
                results['ties'] += 1
    
    return results


def main():
    print("="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    
    engine = PokerEngine(num_players=2)
    
    strategies = {
        'Random': random_strategy,
        'Aggressive': aggressive_strategy,
        'Conservative': conservative_strategy
    }
    
    matchups = [
        ('Random', 'Aggressive'),
        ('Random', 'Conservative'),
        ('Aggressive', 'Conservative')
    ]
    
    for strat1_name, strat2_name in matchups:
        print(f"\n{strat1_name} vs {strat2_name}")
        print("-" * 40)
        
        strat1 = strategies[strat1_name]
        strat2 = strategies[strat2_name]
        
        results = play_match(engine, strat1, strat2, NUM_GAMES)
        
        p0_wins = results['player_0_wins']
        p1_wins = results['player_1_wins']
        ties = results['ties']
        
        p0_avg = np.mean(results['player_0_rewards'])
        p1_avg = np.mean(results['player_1_rewards'])
        
        print(f"{strat1_name}: {p0_wins} wins ({p0_wins/NUM_GAMES*100:.1f}%), avg: {p0_avg:.2f}")
        print(f"{strat2_name}: {p1_wins} wins ({p1_wins/NUM_GAMES*100:.1f}%), avg: {p1_avg:.2f}")
        print(f"Ties: {ties}")
    
    print("\n" + "="*60)
    
    engine.close()


if __name__ == "__main__":
    main()
