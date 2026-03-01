import torch
import torch.nn as nn
import numpy as np
from typing import List
from .deep_cfr import DeepCFR, AdvantageNetwork, StrategyNetwork


class BatchedDeepCFR(DeepCFR):
    
    def get_strategies_batch(self, observations: np.ndarray, 
                            player_indices: List[int],
                            legal_actions_list: List[List[int]]) -> List[np.ndarray]:
        if len(observations) == 0:
            return []
        
        player_batches = {}
        for i, (obs, player_idx, legal_actions) in enumerate(
            zip(observations, player_indices, legal_actions_list)
        ):
            if player_idx not in player_batches:
                player_batches[player_idx] = {'obs': [], 'legal': [], 'indices': []}
            player_batches[player_idx]['obs'].append(obs)
            player_batches[player_idx]['legal'].append(legal_actions)
            player_batches[player_idx]['indices'].append(i)
        
        results = [None] * len(observations)
        
        for player_idx, batch_data in player_batches.items():
            obs_batch = torch.FloatTensor(np.array(batch_data['obs'])).to(self.device)
            
            with torch.no_grad():
                strategies = self.strategy_networks[player_idx](obs_batch).cpu().numpy()
            
            for i, (strategy, legal_actions, orig_idx) in enumerate(
                zip(strategies, batch_data['legal'], batch_data['indices'])
            ):
                legal_mask = np.zeros(self.action_size)
                legal_mask[legal_actions] = 1
                strategy = strategy * legal_mask
                
                strategy_sum = strategy.sum()
                if strategy_sum > 0:
                    strategy = strategy / strategy_sum
                else:
                    strategy = legal_mask / legal_mask.sum()
                
                results[orig_idx] = strategy
        
        return results
    
    def compute_counterfactual_values_batch(self, observations: np.ndarray,
                                           player_indices: List[int],
                                           legal_actions_list: List[List[int]]) -> List[np.ndarray]:
        if len(observations) == 0:
            return []
        
        player_batches = {}
        for i, (obs, player_idx, legal_actions) in enumerate(
            zip(observations, player_indices, legal_actions_list)
        ):
            if player_idx not in player_batches:
                player_batches[player_idx] = {'obs': [], 'legal': [], 'indices': []}
            player_batches[player_idx]['obs'].append(obs)
            player_batches[player_idx]['legal'].append(legal_actions)
            player_batches[player_idx]['indices'].append(i)
        
        results = [None] * len(observations)
        
        for player_idx, batch_data in player_batches.items():
            obs_batch = torch.FloatTensor(np.array(batch_data['obs'])).to(self.device)
            
            with torch.no_grad():
                advantages = self.advantage_networks[player_idx](obs_batch).cpu().numpy()
            
            for i, (advantage, legal_actions, orig_idx) in enumerate(
                zip(advantages, batch_data['legal'], batch_data['indices'])
            ):
                legal_mask = np.zeros(self.action_size)
                legal_mask[legal_actions] = 1
                advantage = advantage * legal_mask
                
                results[orig_idx] = advantage
        
        return results
