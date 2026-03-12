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
    """Columnar replay buffer using pre-allocated numpy arrays for fast sampling.

    Instead of a list of Python dicts, stores data in contiguous numpy arrays.
    This eliminates per-sample Python overhead during batch sampling (the main
    training bottleneck: 300 sample calls × 32768 items per iteration).
    """

    def __init__(self, capacity: int, num_players: int = 6,
                 obs_size: int = 38, act_size: int = 5):
        self.capacity = capacity
        self.num_players = num_players
        self._size = 0
        self._pos = 0

        # Pre-allocated columnar storage
        self.states = np.zeros((capacity, obs_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.regrets = np.zeros(capacity, dtype=np.float32)
        self.strategies = np.zeros((capacity, act_size), dtype=np.float32)
        self.players = np.zeros(capacity, dtype=np.int32)
        self.iterations = np.zeros(capacity, dtype=np.float64)

        # Per-player index caches
        self._player_indices = [None] * num_players
        self._player_weights = [None] * num_players
        self._dirty = True

    def add(self, state: np.ndarray, action: int, regret: float,
            strategy: np.ndarray, player: int, iteration: int = 1):
        i = self._pos
        self.states[i] = state
        self.actions[i] = action
        self.regrets[i] = regret
        self.strategies[i] = strategy
        self.players[i] = player
        self.iterations[i] = max(iteration, 1)

        self._pos = (self._pos + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1
        self._dirty = True

    def rebuild_indices(self):
        """Rebuild per-player indices and pre-compute weights."""
        players_slice = self.players[:self._size]
        iters_slice = self.iterations[:self._size]
        for p in range(self.num_players):
            mask = players_slice == p
            idx = np.nonzero(mask)[0]
            self._player_indices[p] = idx
            if len(idx) > 0:
                w = iters_slice[idx].copy()
                w /= w.sum()
                self._player_weights[p] = w
            else:
                self._player_weights[p] = None
        self._dirty = False

    def sample_player_tensors(self, batch_size: int, player_idx: int, device):
        """Sample batch and return GPU tensors directly (skips Python dicts)."""
        if self._dirty:
            self.rebuild_indices()

        idx = self._player_indices[player_idx]
        n = len(idx)
        if n == 0:
            return None

        w = self._player_weights[player_idx]
        k = min(batch_size, n)
        chosen = np.random.choice(n, size=k, replace=False, p=w)
        sel = idx[chosen]

        states_t = torch.from_numpy(self.states[sel]).to(device, non_blocking=True)
        actions_t = torch.from_numpy(self.actions[sel]).to(device, non_blocking=True)
        regrets_t = torch.from_numpy(self.regrets[sel]).to(device, non_blocking=True)
        strategies_t = torch.from_numpy(self.strategies[sel]).to(device, non_blocking=True)
        return states_t, actions_t, regrets_t, strategies_t

    def sample_player(self, batch_size: int, player_idx: int) -> List[Dict]:
        """Legacy dict-based sampling (kept for compatibility)."""
        if self._dirty:
            self.rebuild_indices()

        idx = self._player_indices[player_idx]
        n = len(idx)
        if n == 0:
            return []

        w = self._player_weights[player_idx]
        k = min(batch_size, n)
        chosen = np.random.choice(n, size=k, replace=False, p=w)
        sel = idx[chosen]
        return [{'state': self.states[i], 'action': self.actions[i],
                 'regret': self.regrets[i], 'strategy': self.strategies[i],
                 'player': self.players[i], 'iteration': self.iterations[i]}
                for i in sel]

    def sample(self, batch_size: int) -> List[Dict]:
        """Sample from the full buffer (iteration-weighted)."""
        n = self._size
        if n == 0:
            return []
        w = self.iterations[:n].copy()
        w /= w.sum()
        k = min(batch_size, n)
        sel = np.random.choice(n, size=k, replace=False, p=w)
        return [{'state': self.states[i], 'action': self.actions[i],
                 'regret': self.regrets[i], 'strategy': self.strategies[i],
                 'player': self.players[i], 'iteration': self.iterations[i]}
                for i in sel]

    def __len__(self):
        return self._size


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

            adv_sched = optim.lr_scheduler.LinearLR(adv_opt, start_factor=1.0, end_factor=0.1, total_iters=lr_scheduler_t_max)
            strat_sched = optim.lr_scheduler.LinearLR(strat_opt, start_factor=1.0, end_factor=0.1, total_iters=lr_scheduler_t_max)

            self.advantage_networks.append(adv_net)
            self.strategy_networks.append(strat_net)
            self.advantage_optimizers.append(adv_opt)
            self.strategy_optimizers.append(strat_opt)
            self.advantage_schedulers.append(adv_sched)
            self.strategy_schedulers.append(strat_sched)

        self.replay_buffer = ReplayBuffer(
            buffer_size, num_players=num_players,
            obs_size=observation_size, act_size=action_size)
        self.iteration = 0

        # AMP scaler for mixed-precision training
        self._amp_enabled = self.device.type == 'cuda'
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)

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

        total_loss = 0.0
        count = 0

        for player_idx in range(self.num_players):
            batch = self.replay_buffer.sample_player_tensors(
                batch_size, player_idx, self.device)
            if batch is None:
                continue
            states, actions, regrets, _ = batch

            self.advantage_optimizers[player_idx].zero_grad()

            with torch.cuda.amp.autocast(enabled=self._amp_enabled):
                predicted = self.advantage_networks[player_idx](states)
                pred_act = predicted.gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = nn.MSELoss()(pred_act, regrets)

            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.advantage_optimizers[player_idx])
            torch.nn.utils.clip_grad_norm_(
                self.advantage_networks[player_idx].parameters(),
                max_norm=self.gradient_clip
            )
            self._scaler.step(self.advantage_optimizers[player_idx])
            self._scaler.update()

            total_loss += loss.item()
            count += 1

        return total_loss / count if count > 0 else 0.0

    def train_strategy_network(self, batch_size: int = 256) -> float:
        if len(self.replay_buffer) < batch_size:
            return 0.0

        total_loss = 0.0
        count = 0

        for player_idx in range(self.num_players):
            batch = self.replay_buffer.sample_player_tensors(
                batch_size, player_idx, self.device)
            if batch is None:
                continue
            states, _, _, target_strategies = batch

            self.strategy_optimizers[player_idx].zero_grad()

            with torch.cuda.amp.autocast(enabled=self._amp_enabled):
                predicted = self.strategy_networks[player_idx](states)
                loss = nn.KLDivLoss(reduction='batchmean')(
                    torch.log(predicted + 1e-8), target_strategies
                )

            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.strategy_optimizers[player_idx])
            torch.nn.utils.clip_grad_norm_(
                self.strategy_networks[player_idx].parameters(),
                max_norm=self.gradient_clip
            )
            self._scaler.step(self.strategy_optimizers[player_idx])
            self._scaler.update()

            total_loss += loss.item()
            count += 1

        return total_loss / count if count > 0 else 0.0

    def step_schedulers(self):
        """Step all LR schedulers once. Call once per iteration from the main loop."""
        for sched in self.advantage_schedulers + self.strategy_schedulers:
            sched.step()

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
