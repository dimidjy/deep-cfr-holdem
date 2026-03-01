import numpy as np
from typing import Dict, List, Tuple, Optional
from src.engine import PokerEngine
from src.training.deep_cfr import DeepCFR
import json
from pathlib import Path


class CFRTrainer:
    
    def __init__(self, 
                 engine: PokerEngine,
                 deep_cfr: DeepCFR,
                 config: Optional[Dict] = None):
        
        self.engine = engine
        self.deep_cfr = deep_cfr
        self.config = config or {}
        
        self.iteration = 0
        self.cumulative_regrets = {}
        self.cumulative_strategies = {}
    
    def run_traversal(self, player_idx: int) -> float:
        self.engine.reset()
        
        episode_data = []
        total_reward = 0
        
        while not self.engine.is_done():
            current_agent = self.engine.current_agent
            agent_idx = self.engine.agents.index(current_agent)
            
            observation = self.engine.get_observation()
            legal_actions = self.engine._get_legal_actions()
            
            if not legal_actions:
                break
            
            if agent_idx == player_idx:
                advantages = self.deep_cfr.compute_counterfactual_values(
                    observation, agent_idx, legal_actions
                )
                strategy = self.deep_cfr.get_regret_matching_strategy(advantages, legal_actions)
            else:
                strategy = self.deep_cfr.get_strategy(observation, agent_idx, legal_actions)
            
            action = np.random.choice(legal_actions, p=strategy[legal_actions] / strategy[legal_actions].sum())
            
            episode_data.append({
                'player': agent_idx,
                'observation': observation.copy(),
                'action': action,
                'strategy': strategy,
                'legal_actions': legal_actions.copy()
            })
            
            result = self.engine.step(action)
            
            if agent_idx == player_idx:
                total_reward += result['reward']
        
        self.update_regrets(episode_data, player_idx, total_reward)
        
        return total_reward
    
    def update_regrets(self, episode_data: List[Dict], player_idx: int, final_reward: float):
        for step_data in episode_data:
            if step_data['player'] != player_idx:
                continue
            
            observation = step_data['observation']
            action = step_data['action']
            strategy = step_data['strategy']
            legal_actions = step_data['legal_actions']
            
            action_value = final_reward
            
            expected_value = np.sum(strategy[legal_actions] * action_value)
            
            regret = action_value - expected_value
            
            self.deep_cfr.replay_buffer.add(
                observation, action, regret, strategy, player_idx
            )
    
    def train_iteration(self, traversals_per_iteration: int = 1000):
        self.iteration += 1
        
        for player_idx in range(self.deep_cfr.num_players):
            for _ in range(traversals_per_iteration):
                self.run_traversal(player_idx)
        
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
                print(f"Iteration {iteration + 1}/{num_iterations} completed")
    
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
                legal_actions = self.engine._get_legal_actions()
                
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
