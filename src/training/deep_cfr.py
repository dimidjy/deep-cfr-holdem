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
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Returns raw logits (not probabilities). Apply masked softmax externally."""
        return self.network(x)


class ReplayBuffer:
    """Columnar replay buffer with reservoir sampling.

    Uses reservoir sampling instead of FIFO: once the buffer is full, each
    new entry has a (capacity / total_seen) probability of replacing a random
    existing entry. This preserves a representative sample across ALL
    iterations, preventing strategy collapse from losing diverse old data.

    Research confirms: "with sliding-window memories, exploitability stops
    converging once the buffer becomes full. With reservoir sampling,
    convergence improves." (Deep Predictive Discounted CFR, 2025)
    """

    def __init__(self, capacity: int, num_players: int = 6,
                 obs_size: int = 38, act_size: int = 5):
        self.capacity = capacity
        self.num_players = num_players
        self._size = 0
        self._total_seen = 0  # total entries ever added (for reservoir sampling)

        # Pre-allocated columnar storage
        self.states = np.zeros((capacity, obs_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.regrets = np.zeros(capacity, dtype=np.float32)
        self.strategies = np.zeros((capacity, act_size), dtype=np.float32)
        self.legal_masks = np.zeros((capacity, act_size), dtype=np.float32)
        self.players = np.zeros(capacity, dtype=np.int32)
        self.iterations = np.zeros(capacity, dtype=np.float64)

        # Per-player index caches
        self._player_indices = [None] * num_players
        self._player_weights = [None] * num_players
        self._player_weights_sqrt = [None] * num_players
        self._dirty = True

    def add(self, state: np.ndarray, action: int, regret: float,
            strategy: np.ndarray, player: int, iteration: int = 1,
            legal_mask: np.ndarray = None):
        self._total_seen += 1

        if self._size < self.capacity:
            # Buffer not full yet — append normally
            i = self._size
            self._size += 1
        else:
            # Reservoir sampling: accept with probability capacity / total_seen
            # This ensures uniform sampling across all iterations ever seen
            j = random.randint(0, self._total_seen - 1)
            if j >= self.capacity:
                return  # reject this entry (keeps existing data)
            i = j  # replace random existing entry

        self.states[i] = state
        self.actions[i] = action
        self.regrets[i] = regret
        self.strategies[i] = strategy
        if legal_mask is not None:
            self.legal_masks[i] = legal_mask
        else:
            self.legal_masks[i] = (strategy > 0).astype(np.float32)
        self.players[i] = player
        self.iterations[i] = max(iteration, 1)
        self._dirty = True

    def rebuild_indices(self):
        """Rebuild per-player indices and pre-compute weights.

        Two weight sets:
        - _player_weights: linear iteration weighting (for advantage network)
        - _player_weights_sqrt: sqrt iteration weighting (for strategy network)
          Sqrt weighting gives more influence to older data, preventing the
          strategy network from collapsing to the current aggressive policy.
        """
        players_slice = self.players[:self._size]
        iters_slice = self.iterations[:self._size]
        for p in range(self.num_players):
            mask = players_slice == p
            idx = np.nonzero(mask)[0]
            self._player_indices[p] = idx
            if len(idx) > 0:
                # Linear weights for advantage network (standard Deep CFR)
                w = iters_slice[idx].astype(np.float64)
                w /= w.sum()
                self._player_weights[p] = w
                # Sqrt weights for strategy network (flatter, more diverse)
                w_sqrt = np.sqrt(iters_slice[idx].astype(np.float64))
                w_sqrt /= w_sqrt.sum()
                self._player_weights_sqrt[p] = w_sqrt
            else:
                self._player_weights[p] = None
                self._player_weights_sqrt[p] = None
        self._dirty = False

    def sample_player_tensors(self, batch_size: int, player_idx: int, device):
        """Sample batch with importance weights. Returns 6 tensors (+ weights).

        Uses fast uniform sampling instead of slow weighted np.random.choice.
        Iteration weighting is applied via importance weights in the loss.
        """
        if self._dirty:
            self.rebuild_indices()

        idx = self._player_indices[player_idx]
        n = len(idx)
        if n == 0:
            return None

        k = min(batch_size, n)
        chosen = np.random.randint(0, n, size=k)  # fast uniform sampling
        sel = idx[chosen]

        # Importance weights: iteration-proportional (normalize to mean=1)
        raw_w = self.iterations[sel].copy()
        raw_w /= max(raw_w.mean(), 1e-8)

        states_t = torch.from_numpy(self.states[sel]).to(device, non_blocking=True)
        actions_t = torch.from_numpy(self.actions[sel]).to(device, non_blocking=True)
        regrets_t = torch.from_numpy(self.regrets[sel]).to(device, non_blocking=True)
        strategies_t = torch.from_numpy(self.strategies[sel]).to(device, non_blocking=True)
        legal_masks_t = torch.from_numpy(self.legal_masks[sel]).to(device, non_blocking=True)
        weights_t = torch.from_numpy(raw_w.astype(np.float32)).to(device, non_blocking=True)
        return states_t, actions_t, regrets_t, strategies_t, legal_masks_t, weights_t

    def sample_player_tensors_sqrt(self, batch_size: int, player_idx: int, device):
        """Sample with sqrt importance weights (flatter, for strategy network)."""
        if self._dirty:
            self.rebuild_indices()

        idx = self._player_indices[player_idx]
        n = len(idx)
        if n == 0:
            return None

        k = min(batch_size, n)
        chosen = np.random.randint(0, n, size=k)  # fast uniform sampling
        sel = idx[chosen]

        # Sqrt importance weights (flatter than linear)
        raw_w = np.sqrt(self.iterations[sel].copy())
        raw_w /= max(raw_w.mean(), 1e-8)

        states_t = torch.from_numpy(self.states[sel]).to(device, non_blocking=True)
        actions_t = torch.from_numpy(self.actions[sel]).to(device, non_blocking=True)
        regrets_t = torch.from_numpy(self.regrets[sel]).to(device, non_blocking=True)
        strategies_t = torch.from_numpy(self.strategies[sel]).to(device, non_blocking=True)
        legal_masks_t = torch.from_numpy(self.legal_masks[sel]).to(device, non_blocking=True)
        weights_t = torch.from_numpy(raw_w.astype(np.float32)).to(device, non_blocking=True)
        return states_t, actions_t, regrets_t, strategies_t, legal_masks_t, weights_t

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

    def save(self, filepath: str):
        """Save buffer contents to disk (numpy compressed)."""
        n = self._size
        np.savez_compressed(filepath,
                            states=self.states[:n],
                            actions=self.actions[:n],
                            regrets=self.regrets[:n],
                            strategies=self.strategies[:n],
                            legal_masks=self.legal_masks[:n],
                            players=self.players[:n],
                            iterations=self.iterations[:n],
                            size=np.array([n]),
                            total_seen=np.array([self._total_seen]))

    def load(self, filepath: str):
        """Load buffer contents from disk."""
        data = np.load(filepath)
        n = int(data['size'][0])
        if n == 0:
            return

        if n > self.capacity:
            n = self.capacity

        self.states[:n] = data['states'][:n]
        self.actions[:n] = data['actions'][:n]
        self.regrets[:n] = data['regrets'][:n]
        self.strategies[:n] = data['strategies'][:n]
        self.legal_masks[:n] = data['legal_masks'][:n]
        self.players[:n] = data['players'][:n]
        self.iterations[:n] = data['iterations'][:n]
        self._size = n
        self._total_seen = int(data['total_seen'][0]) if 'total_seen' in data else n
        self._dirty = True


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

    def extend_first_layer(self, new_obs_size: int):
        """Extend all networks from current obs_size to new_obs_size.

        Used for fine-tuning: copies existing weights, near-zero-inits new
        input dimensions so the model initially ignores them and gradually
        learns their meaning. Clears replay buffer (old obs incompatible)
        and rebuilds optimizers (old momentum buffers wrong shape).
        """
        old_obs_size = self.observation_size
        extra = new_obs_size - old_obs_size
        if extra <= 0:
            return

        for net_list in [self.advantage_networks, self.strategy_networks]:
            for net in net_list:
                old_layer = net.network[0]  # first nn.Linear
                new_layer = nn.Linear(new_obs_size, old_layer.out_features)
                with torch.no_grad():
                    new_layer.weight[:, :old_obs_size] = old_layer.weight
                    new_layer.weight[:, old_obs_size:] = (
                        torch.randn(old_layer.out_features, extra) * 0.01)
                    new_layer.bias[:] = old_layer.bias
                net.network[0] = new_layer
                net.to(self.device)

        self.observation_size = new_obs_size

        # Clear buffer — old states have wrong dimensionality
        self.replay_buffer = ReplayBuffer(
            self.replay_buffer.capacity,
            num_players=self.num_players,
            obs_size=new_obs_size,
            act_size=self.action_size,
        )

        # Rebuild optimizers — old momentum buffers reference old param shapes
        lr = self.advantage_optimizers[0].param_groups[0]['lr']
        self.advantage_optimizers = [
            optim.Adam(net.parameters(), lr=lr)
            for net in self.advantage_networks
        ]
        self.strategy_optimizers = [
            optim.Adam(net.parameters(), lr=lr)
            for net in self.strategy_networks
        ]

    def get_strategy(self, observation: np.ndarray, player_idx: int,
                     legal_actions: List[int]) -> np.ndarray:

        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.strategy_networks[player_idx](obs_tensor).cpu().numpy()[0]

        # Masked softmax: set illegal actions to -inf, then softmax
        strategy = np.full(self.action_size, -1e9, dtype=np.float32)
        for a in legal_actions:
            strategy[a] = logits[a]

        # Numerically stable softmax over legal actions
        strategy -= strategy.max()
        exp_s = np.exp(strategy)
        strategy = exp_s / exp_s.sum()

        # Zero out any residual mass on illegal actions (from exp(-1e9) ≈ 0)
        legal_mask = np.zeros(self.action_size, dtype=np.float32)
        legal_mask[legal_actions] = 1.0
        strategy = strategy * legal_mask
        s = strategy.sum()
        if s > 0:
            strategy /= s
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
            states, actions, regrets, _, _, weights = batch

            self.advantage_optimizers[player_idx].zero_grad()

            with torch.cuda.amp.autocast(enabled=self._amp_enabled):
                predicted = self.advantage_networks[player_idx](states)
                pred_act = predicted.gather(1, actions.unsqueeze(1)).squeeze(1)
                # Weighted MSE loss (importance sampling)
                loss = (weights * (pred_act - regrets) ** 2).mean()

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
            # Use sqrt-weighted sampling for strategy network — gives more
            # weight to older iterations, preventing collapse to current policy
            batch = self.replay_buffer.sample_player_tensors_sqrt(
                batch_size, player_idx, self.device)
            if batch is None:
                continue
            states, _, _, target_strategies, legal_masks, weights = batch

            self.strategy_optimizers[player_idx].zero_grad()

            with torch.cuda.amp.autocast(enabled=self._amp_enabled):
                logits = self.strategy_networks[player_idx](states)

                # Mask illegal actions to -inf before log_softmax
                logits = logits + (1.0 - legal_masks) * (-1e9)
                log_probs = torch.log_softmax(logits, dim=-1)

                # Renormalize targets to legal actions only (safety)
                masked_targets = target_strategies * legal_masks
                target_sums = masked_targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                masked_targets = masked_targets / target_sums

                # KL divergence only on legal actions
                # KL = sum(target * (log(target) - log_pred))
                # Only compute where legal_masks == 1
                kl_per_action = masked_targets * (
                    torch.log(masked_targets + 1e-8) - log_probs
                )
                # Zero out illegal actions, apply importance weights
                kl_per_action = kl_per_action * legal_masks
                loss = (weights * kl_per_action.sum(dim=-1)).mean()

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
