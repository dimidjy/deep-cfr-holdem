import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from typing import Dict, List, Optional
from pathlib import Path
import json


def run_traversal_worker(args):
    player_idx, seed, num_players, observation_size, action_size, device = args
    
    np.random.seed(seed)
    
    from src.engine import PokerEngine
    
    engine = PokerEngine(num_players=num_players)
    engine.reset()
    
    episode_data = []
    
    while not engine.is_done():
        current_agent = engine.current_agent
        agent_idx = engine.agents.index(current_agent)
        
        observation = engine.get_observation()
        legal_actions = engine.get_legal_actions()
        
        if not legal_actions:
            break
        
        legal_mask = np.zeros(action_size)
        legal_mask[legal_actions] = 1
        strategy = legal_mask / legal_mask.sum()
        
        legal_strategy = strategy[legal_actions]
        legal_strategy = legal_strategy / legal_strategy.sum()
        action = np.random.choice(legal_actions, p=legal_strategy)
        
        episode_data.append({
            'player': agent_idx,
            'observation': observation.copy(),
            'action': action,
            'strategy': strategy,
            'legal_actions': legal_actions.copy()
        })
        
        result = engine.step(action)
    
    rewards = engine.get_rewards()
    final_reward = rewards[engine.agents[player_idx]]
    
    engine.close()
    
    return episode_data, final_reward, player_idx


class ParallelCFRTrainer:
    
    def __init__(self, engine, deep_cfr, config: Optional[Dict] = None, num_workers: int = None):
        self.engine = engine
        self.deep_cfr = deep_cfr
        self.config = config or {}
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        self.iteration = 0
        
        print(f"Parallel trainer initialized with {self.num_workers} workers")
    
    def train_iteration(self, traversals_per_iteration: int = 1000):
        self.iteration += 1
        
        work_items = []
        for player_idx in range(self.deep_cfr.num_players):
            for i in range(traversals_per_iteration):
                seed = self.iteration * 100000 + player_idx * 10000 + i
                work_items.append((
                    player_idx,
                    seed,
                    self.deep_cfr.num_players,
                    self.deep_cfr.observation_size,
                    self.deep_cfr.action_size,
                    str(self.deep_cfr.device)
                ))
        
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(run_traversal_worker, work_items, chunksize=10)
        
        for episode_data, final_reward, player_idx in results:
            for step_data in episode_data:
                if step_data['player'] != player_idx:
                    continue
                
                observation = step_data['observation']
                action = step_data['action']
                strategy = step_data['strategy']
                
                regret = final_reward
                
                self.deep_cfr.replay_buffer.add(
                    observation, action, regret, strategy, player_idx
                )
        
        batch_size = self.config.get('batch_size', 128)
        
        for _ in range(10):
            self.deep_cfr.train_advantage_network(batch_size)
            self.deep_cfr.train_strategy_network(batch_size)
        
        self.deep_cfr.iteration = self.iteration
    
    def train(self, num_iterations: int, checkpoint_interval: int = 100):
        traversals = self.config.get('traversals_per_iteration', 1000)
        
        for iteration in range(num_iterations):
            self.train_iteration(traversals)
            
            if (iteration + 1) % checkpoint_interval == 0:
                self.save_checkpoint(f"models/checkpoint_iter_{iteration + 1}.pt")
                print(f"Iteration {iteration + 1}/{num_iterations} completed - Buffer: {len(self.deep_cfr.replay_buffer)}")
    
    def save_checkpoint(self, filepath: str):
        self.deep_cfr.save_models(filepath)
        
        metadata = {
            'iteration': self.iteration,
            'buffer_size': len(self.deep_cfr.replay_buffer),
            'config': self.config
        }
        
        metadata_path = filepath.replace('.pt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_checkpoint(self, filepath: str):
        self.deep_cfr.load_models(filepath)
        
        metadata_path = filepath.replace('.pt', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.iteration = metadata.get('iteration', 0)
    
    def evaluate(self, num_games: int = 100) -> Dict[str, float]:
        results = {
            'total_games': num_games,
            'player_rewards': {i: [] for i in range(self.deep_cfr.num_players)}
        }
        
        for _ in range(num_games):
            self.engine.reset()
            
            while not self.engine.is_done():
                current_agent = self.engine.current_agent
                agent_idx = self.engine.agents.index(current_agent)
                
                observation = self.engine.get_observation()
                legal_actions = self.engine.get_legal_actions()
                
                if not legal_actions:
                    break
                
                strategy = self.deep_cfr.get_strategy(observation, agent_idx, legal_actions)
                action = np.random.choice(legal_actions, p=strategy[legal_actions] / strategy[legal_actions].sum())
                
                self.engine.step(action)
            
            rewards = self.engine.get_rewards()
            for idx, agent in enumerate(self.engine.agents):
                results['player_rewards'][idx].append(rewards[agent])
        
        for player_idx in range(self.deep_cfr.num_players):
            rewards = results['player_rewards'][player_idx]
            results[f'player_{player_idx}_avg_reward'] = np.mean(rewards)
            results[f'player_{player_idx}_std_reward'] = np.std(rewards)
        
        return results
