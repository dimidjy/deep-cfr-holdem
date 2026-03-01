import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class AdvantageNetwork(nn.Module):
    
    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int]):
        super(AdvantageNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class StrategyNetwork(nn.Module):
    
    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int]):
        super(StrategyNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, regret: float, 
            strategy: np.ndarray, player: int):
        self.buffer.append({
            'state': state,
            'action': action,
            'regret': regret,
            'strategy': strategy,
            'player': player
        })
    
    def sample(self, batch_size: int) -> List[Dict]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class DeepCFR:
    
    def __init__(self, 
                 observation_size: int,
                 action_size: int,
                 num_players: int = 2,
                 hidden_layers: List[int] = [256, 256, 128],
                 learning_rate: float = 0.001,
                 buffer_size: int = 100000,
                 device: str = 'cpu'):
        
        self.observation_size = observation_size
        self.action_size = action_size
        self.num_players = num_players
        self.device = torch.device(device)
        
        self.advantage_networks = []
        self.strategy_networks = []
        self.advantage_optimizers = []
        self.strategy_optimizers = []
        
        for _ in range(num_players):
            adv_net = AdvantageNetwork(observation_size, action_size, hidden_layers).to(self.device)
            strat_net = StrategyNetwork(observation_size, action_size, hidden_layers).to(self.device)
            
            self.advantage_networks.append(adv_net)
            self.strategy_networks.append(strat_net)
            
            self.advantage_optimizers.append(optim.Adam(adv_net.parameters(), lr=learning_rate))
            self.strategy_optimizers.append(optim.Adam(strat_net.parameters(), lr=learning_rate))
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.iteration = 0
    
    def get_strategy(self, observation: np.ndarray, player_idx: int, 
                    legal_actions: List[int]) -> np.ndarray:
        
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            strategy = self.strategy_networks[player_idx](obs_tensor).cpu().numpy()[0]
        
        legal_mask = np.zeros(self.action_size)
        legal_mask[legal_actions] = 1
        strategy = strategy * legal_mask
        
        strategy_sum = strategy.sum()
        if strategy_sum > 0:
            strategy = strategy / strategy_sum
        else:
            strategy = legal_mask / legal_mask.sum()
        
        return strategy
    
    def get_regret_matching_strategy(self, regrets: np.ndarray, 
                                    legal_actions: List[int]) -> np.ndarray:
        
        positive_regrets = np.maximum(regrets, 0)
        
        legal_mask = np.zeros(self.action_size)
        legal_mask[legal_actions] = 1
        positive_regrets = positive_regrets * legal_mask
        
        regret_sum = positive_regrets.sum()
        
        if regret_sum > 0:
            strategy = positive_regrets / regret_sum
        else:
            strategy = legal_mask / legal_mask.sum()
        
        return strategy
    
    def compute_counterfactual_values(self, observation: np.ndarray, 
                                     player_idx: int,
                                     legal_actions: List[int]) -> np.ndarray:
        
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            advantages = self.advantage_networks[player_idx](obs_tensor).cpu().numpy()[0]
        
        legal_mask = np.zeros(self.action_size)
        legal_mask[legal_actions] = 1
        advantages = advantages * legal_mask
        
        return advantages
    
    def train_advantage_network(self, batch_size: int = 128):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        
        for player_idx in range(self.num_players):
            player_batch = [sample for sample in batch if sample['player'] == player_idx]
            
            if not player_batch:
                continue
            
            states = torch.FloatTensor([s['state'] for s in player_batch]).to(self.device)
            actions = torch.LongTensor([s['action'] for s in player_batch]).to(self.device)
            regrets = torch.FloatTensor([s['regret'] for s in player_batch]).to(self.device)
            
            self.advantage_optimizers[player_idx].zero_grad()
            
            predicted_advantages = self.advantage_networks[player_idx](states)
            predicted_action_advantages = predicted_advantages.gather(1, actions.unsqueeze(1)).squeeze()
            
            loss = nn.MSELoss()(predicted_action_advantages, regrets)
            
            loss.backward()
            self.advantage_optimizers[player_idx].step()
    
    def train_strategy_network(self, batch_size: int = 128):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        
        for player_idx in range(self.num_players):
            player_batch = [sample for sample in batch if sample['player'] == player_idx]
            
            if not player_batch:
                continue
            
            states = torch.FloatTensor([s['state'] for s in player_batch]).to(self.device)
            target_strategies = torch.FloatTensor([s['strategy'] for s in player_batch]).to(self.device)
            
            self.strategy_optimizers[player_idx].zero_grad()
            
            predicted_strategies = self.strategy_networks[player_idx](states)
            
            loss = nn.KLDivLoss(reduction='batchmean')(
                torch.log(predicted_strategies + 1e-8),
                target_strategies
            )
            
            loss.backward()
            self.strategy_optimizers[player_idx].step()
    
    def save_models(self, filepath: str):
        checkpoint = {
            'iteration': self.iteration,
            'advantage_networks': [net.state_dict() for net in self.advantage_networks],
            'strategy_networks': [net.state_dict() for net in self.strategy_networks],
            'advantage_optimizers': [opt.state_dict() for opt in self.advantage_optimizers],
            'strategy_optimizers': [opt.state_dict() for opt in self.strategy_optimizers]
        }
        
        torch.save(checkpoint, filepath)
    
    def load_models(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.iteration = checkpoint['iteration']
        
        for idx in range(self.num_players):
            self.advantage_networks[idx].load_state_dict(checkpoint['advantage_networks'][idx])
            self.strategy_networks[idx].load_state_dict(checkpoint['strategy_networks'][idx])
            self.advantage_optimizers[idx].load_state_dict(checkpoint['advantage_optimizers'][idx])
            self.strategy_optimizers[idx].load_state_dict(checkpoint['strategy_optimizers'][idx])
