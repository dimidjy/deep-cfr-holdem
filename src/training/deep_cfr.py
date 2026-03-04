import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional
from collections import deque
import random


class ResidualBlock(nn.Module):

    def __init__(self, size: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.LayerNorm(size),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(size, size),
            nn.LayerNorm(size),
        )
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x):
        return self.activation(x + self.block(x))


class AdvantageNetwork(nn.Module):

    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int],
                 dropout: float = 0.1):
        super().__init__()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(dropout))

            # Add residual block when consecutive layers have same size
            if i + 1 < len(hidden_layers) and hidden_layers[i + 1] == hidden_size:
                layers.append(ResidualBlock(hidden_size, dropout))

            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class StrategyNetwork(nn.Module):

    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int],
                 dropout: float = 0.1):
        super().__init__()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(dropout))

            if i + 1 < len(hidden_layers) and hidden_layers[i + 1] == hidden_size:
                layers.append(ResidualBlock(hidden_size, dropout))

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
            strategy: np.ndarray, player: int, iteration: int = 1):
        priority = abs(regret) + 1e-5
        self.buffer.append({
            'state': state,
            'action': action,
            'regret': regret,
            'strategy': strategy,
            'player': player,
            'iteration': max(iteration, 1),
            'priority': priority,
        })

    def sample(self, batch_size: int) -> List[Dict]:
        n = len(self.buffer)
        if n == 0:
            return []

        buf_list = list(self.buffer)
        weights = np.array(
            [e['iteration'] * e['priority'] for e in buf_list], dtype=np.float64
        )
        weights /= weights.sum()

        k = min(batch_size, n)
        indices = np.random.choice(n, size=k, replace=False, p=weights)
        return [buf_list[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class DeepCFR:

    def __init__(self,
                 observation_size: int,
                 action_size: int,
                 num_players: int = 2,
                 hidden_layers: List[int] = None,
                 learning_rate: float = 0.001,
                 buffer_size: int = 500000,
                 device: str = 'cpu',
                 lr_scheduler_t_max: int = 2000,
                 gradient_clip: float = 1.0,
                 dropout: float = 0.1):

        if hidden_layers is None:
            hidden_layers = [512, 512, 256]

        self.observation_size = observation_size
        self.action_size = action_size
        self.num_players = num_players
        self.device = torch.device(device)
        self.gradient_clip = gradient_clip

        self.advantage_networks = []
        self.strategy_networks = []
        self.advantage_optimizers = []
        self.strategy_optimizers = []
        self.advantage_schedulers = []
        self.strategy_schedulers = []

        for _ in range(num_players):
            adv_net = AdvantageNetwork(
                observation_size, action_size, hidden_layers, dropout
            ).to(self.device)
            strat_net = StrategyNetwork(
                observation_size, action_size, hidden_layers, dropout
            ).to(self.device)

            adv_opt = optim.Adam(adv_net.parameters(), lr=learning_rate)
            strat_opt = optim.Adam(strat_net.parameters(), lr=learning_rate)

            adv_sched = optim.lr_scheduler.CosineAnnealingLR(adv_opt, T_max=lr_scheduler_t_max)
            strat_sched = optim.lr_scheduler.CosineAnnealingLR(strat_opt, T_max=lr_scheduler_t_max)

            self.advantage_networks.append(adv_net)
            self.strategy_networks.append(strat_net)
            self.advantage_optimizers.append(adv_opt)
            self.strategy_optimizers.append(strat_opt)
            self.advantage_schedulers.append(adv_sched)
            self.strategy_schedulers.append(strat_sched)

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

    def train_advantage_network(self, batch_size: int = 256) -> float:
        if len(self.replay_buffer) < batch_size:
            return 0.0

        batch = self.replay_buffer.sample(batch_size)
        total_loss = 0.0
        count = 0

        for player_idx in range(self.num_players):
            player_batch = [s for s in batch if s['player'] == player_idx]

            if not player_batch:
                continue

            states = torch.FloatTensor(
                np.array([s['state'] for s in player_batch])
            ).to(self.device)
            actions = torch.LongTensor(
                np.array([s['action'] for s in player_batch])
            ).to(self.device)
            regrets = torch.FloatTensor(
                np.array([s['regret'] for s in player_batch])
            ).to(self.device)

            self.advantage_optimizers[player_idx].zero_grad()

            predicted_advantages = self.advantage_networks[player_idx](states)
            predicted_action_advantages = predicted_advantages.gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)

            loss = nn.MSELoss()(predicted_action_advantages, regrets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.advantage_networks[player_idx].parameters(),
                max_norm=self.gradient_clip
            )

            self.advantage_optimizers[player_idx].step()
            self.advantage_schedulers[player_idx].step()
            total_loss += loss.item()
            count += 1

        return total_loss / count if count > 0 else 0.0

    def train_strategy_network(self, batch_size: int = 256) -> float:
        if len(self.replay_buffer) < batch_size:
            return 0.0

        batch = self.replay_buffer.sample(batch_size)
        total_loss = 0.0
        count = 0

        for player_idx in range(self.num_players):
            player_batch = [s for s in batch if s['player'] == player_idx]

            if not player_batch:
                continue

            states = torch.FloatTensor(
                np.array([s['state'] for s in player_batch])
            ).to(self.device)
            target_strategies = torch.FloatTensor(
                np.array([s['strategy'] for s in player_batch])
            ).to(self.device)

            self.strategy_optimizers[player_idx].zero_grad()

            predicted_strategies = self.strategy_networks[player_idx](states)

            loss = nn.KLDivLoss(reduction='batchmean')(
                torch.log(predicted_strategies + 1e-8),
                target_strategies
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.strategy_networks[player_idx].parameters(),
                max_norm=self.gradient_clip
            )

            self.strategy_optimizers[player_idx].step()
            self.strategy_schedulers[player_idx].step()
            total_loss += loss.item()
            count += 1

        return total_loss / count if count > 0 else 0.0

    def save_models(self, filepath: str):
        checkpoint = {
            'iteration': self.iteration,
            'advantage_networks': [net.state_dict() for net in self.advantage_networks],
            'strategy_networks': [net.state_dict() for net in self.strategy_networks],
            'advantage_optimizers': [opt.state_dict() for opt in self.advantage_optimizers],
            'strategy_optimizers': [opt.state_dict() for opt in self.strategy_optimizers],
            'advantage_schedulers': [sch.state_dict() for sch in self.advantage_schedulers],
            'strategy_schedulers': [sch.state_dict() for sch in self.strategy_schedulers],
        }
        torch.save(checkpoint, filepath)

    def load_models(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)

        self.iteration = checkpoint['iteration']

        for idx in range(self.num_players):
            self.advantage_networks[idx].load_state_dict(
                checkpoint['advantage_networks'][idx]
            )
            self.strategy_networks[idx].load_state_dict(
                checkpoint['strategy_networks'][idx]
            )
            self.advantage_optimizers[idx].load_state_dict(
                checkpoint['advantage_optimizers'][idx]
            )
            self.strategy_optimizers[idx].load_state_dict(
                checkpoint['strategy_optimizers'][idx]
            )
            if 'advantage_schedulers' in checkpoint:
                self.advantage_schedulers[idx].load_state_dict(
                    checkpoint['advantage_schedulers'][idx]
                )
            if 'strategy_schedulers' in checkpoint:
                self.strategy_schedulers[idx].load_state_dict(
                    checkpoint['strategy_schedulers'][idx]
                )
