"""
Action-Copy CQL — learns to copy good players' checked-to actions.

Reward is static, based on action similarity (not game outcome):
- Exact match with player's action -> full reward (+1.0 for raises, 0.0 for check)
- Similar raise size -> partial reward (50% adjacent, 25% far)
- Opposite direction (check vs raise) -> negative reward

Each decision point is expanded: one training sample per legal action,
each with similarity-based reward. CQL's conservative regularizer
then pushes the model to copy the data distribution.

Usage:
  # Build data first (from good players)
  python3 scripts/train_cql_actions_copy.py --build-data data/cql_copy.npz \
      --good-players data/good_players.txt --no-cache

  # Train
  python3 scripts/train_cql_actions_copy.py --training-data data/cql_copy.npz \
      --epochs 100 --output models/cql_copy_ct.pt
"""
import sys
import os
import warnings
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import copy
import time
import json
import hashlib
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.training.deep_cfr import AdvantageNetwork, StrategyNetwork
from src.hand_history import ObservationBuilder, DecisionPoint
from src.hand_history.parser import parse_file
from validate_spots import run_validation, validation_summary_line


# ──────────────────────────────────────────────────────────────────────────────
# Action-Copy Reward Matrix
# ──────────────────────────────────────────────────────────────────────────────
# COPY_REWARD[player_action, training_action] -> reward
#
# Philosophy: reward = how "close" the training action is to what the
# good player actually did.  Teaches the model to mimic the player's
# decision-making pattern without connecting to game outcomes.
#
# Actions (checked_to, no fold):
#   1 = check, 2 = raise_half, 3 = raise_pot, 4 = all_in
#
# Row = what player did,  Column = what we train for:
#                 fold  check  r_half  r_pot  all_in
# player check    --    0.0    -0.3   -0.5   -1.0   (pot-controlled)
# player r_half   --   -0.5    1.0     0.5    0.25  (small bet)
# player r_pot    --   -0.5    0.5     1.0    0.5   (big bet)
# player all_in   --   -0.75   0.25    0.5    1.0   (shove)

COPY_REWARD = np.zeros((5, 5), dtype=np.float32)
COPY_REWARD[1] = [0.0,  0.0,  -0.3, -0.5, -1.0]    # player checked
COPY_REWARD[2] = [0.0, -0.5,   1.0,  0.5,  0.25]   # player raise_half
COPY_REWARD[3] = [0.0, -0.5,   0.5,  1.0,  0.5]    # player raise_pot
COPY_REWARD[4] = [0.0, -0.75,  0.25, 0.5,  1.0]    # player all_in


# ──────────────────────────────────────────────────────────────────────────────
# Q-Network
# ──────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """Q-network: maps observation -> Q-value for each action."""

    def __init__(self, obs_size: int, action_size: int,
                 hidden_layers: list, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_size = obs_size
        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(dropout))
            if i + 1 < len(hidden_layers) and hidden_layers[i + 1] == h:
                from src.training.deep_cfr import ResidualBlock
                layers.append(ResidualBlock(h, dropout))
            prev_size = h
        layers.append(nn.Linear(prev_size, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ──────────────────────────────────────────────────────────────────────────────
# CQL-compatible wrapper for validation (mimics DeepCFR interface)
# ──────────────────────────────────────────────────────────────────────────────

class CQLModelWrapper:
    """Wraps a QNetwork to mimic the DeepCFR interface for run_validation."""

    def __init__(self, q_net, obs_size, action_size, num_players, device,
                 iteration=0, temperature=1.0, exclude_fold=True):
        self.action_size = action_size
        self.num_players = num_players
        self.obs_size = obs_size
        self.device = torch.device(device)
        self.iteration = iteration
        self.temperature = temperature
        self.exclude_fold = exclude_fold

        self.q_net = q_net
        self.strategy_networks = [q_net] * num_players
        self.advantage_networks = [q_net] * num_players

    def get_strategy(self, observation: np.ndarray, player_idx: int,
                     legal_actions: list) -> np.ndarray:
        if len(observation) < self.obs_size:
            padded = np.zeros(self.obs_size, dtype=np.float32)
            padded[:len(observation)] = observation
            if len(observation) >= 19 and self.obs_size > len(observation):
                pot_odds = observation[10]
                equity = observation[18]
                evo_idx = self.obs_size - 1
                padded[evo_idx] = np.clip(equity - (1.0 - pot_odds), -1.0, 1.0)
            observation = padded
        obs_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t).cpu().numpy()[0]

        if self.action_size == 6:
            return self._strategy_6to5(q_values, observation, legal_actions)

        masked = np.full(self.action_size, -1e9, dtype=np.float32)
        effective_legal = [a for a in legal_actions if not (self.exclude_fold and a == 0)] or [1]
        for a in effective_legal:
            masked[a] = q_values[a] / self.temperature
        masked -= masked.max()
        exp_s = np.exp(masked)
        strategy = exp_s / exp_s.sum()

        legal_mask = np.zeros(self.action_size, dtype=np.float32)
        legal_mask[legal_actions] = 1.0
        strategy *= legal_mask
        s = strategy.sum()
        if s > 0:
            strategy /= s
        else:
            strategy[legal_actions] = 1.0 / len(legal_actions)
        return strategy

    def _strategy_6to5(self, q_values: np.ndarray, observation: np.ndarray,
                       legal_actions: list) -> np.ndarray:
        call_norm = float(observation[31]) if len(observation) > 31 else 0.0
        facing_bet = call_norm > 0.01
        mapped_q = np.full(5, -1e9, dtype=np.float32)
        mapped_q[0] = q_values[0]
        mapped_q[1] = q_values[2] if facing_bet else q_values[1]
        mapped_q[2] = q_values[3]
        mapped_q[3] = q_values[4]
        mapped_q[4] = q_values[5]

        masked = np.full(5, -1e9, dtype=np.float32)
        for a in legal_actions:
            masked[a] = mapped_q[a] / self.temperature
        masked -= masked.max()
        exp_s = np.exp(masked)
        strategy = exp_s / exp_s.sum()

        legal_mask = np.zeros(5, dtype=np.float32)
        legal_mask[legal_actions] = 1.0
        strategy *= legal_mask
        s = strategy.sum()
        if s > 0:
            strategy /= s
        else:
            for a in legal_actions:
                strategy[a] = 1.0 / len(legal_actions)
        return strategy


# ──────────────────────────────────────────────────────────────────────────────
# Engine factory
# ──────────────────────────────────────────────────────────────────────────────

def _make_engine(num_players):
    from src.engine import EnhancedPokerEngine
    return EnhancedPokerEngine(
        num_players=num_players,
        add_position_features=True,
        add_pot_odds_features=True,
        add_hand_strength_features=True,
        add_stack_features=True,
        add_equity_features=True,
        add_street_features=True,
        add_game_state_features=True,
        add_betting_history_features=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class _ActionCopyDataset(Dataset):
    """Zero-copy dataset that indexes into shared numpy arrays."""

    def __init__(self, states, actions, rewards, next_states, dones,
                 legal_masks, next_legal_masks, weights, is_primary, indices):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.legal_masks = legal_masks
        self.next_legal_masks = next_legal_masks
        self.weights = weights
        self.is_primary = is_primary
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (torch.from_numpy(self.states[i]),
                torch.tensor(self.actions[i], dtype=torch.long),
                torch.tensor(self.rewards[i], dtype=torch.float32),
                torch.from_numpy(self.next_states[i]),
                torch.tensor(self.dones[i], dtype=torch.float32),
                torch.from_numpy(self.legal_masks[i]),
                torch.from_numpy(self.next_legal_masks[i]),
                torch.tensor(self.weights[i], dtype=torch.float32),
                torch.tensor(self.is_primary[i], dtype=torch.float32))


# ──────────────────────────────────────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────────────────────────────────────

def _prefilter_files(hh_files, players, hh_dir):
    """Use ripgrep to quickly find files containing target player names."""
    import subprocess
    import shutil
    import re

    if not players:
        return hh_files

    rg_path = shutil.which('rg')
    if not rg_path:
        return hh_files

    matched_files = set()
    try:
        result = subprocess.run(
            [rg_path, '-l', '--no-messages', '-f', '-', str(hh_dir)],
            input='\n'.join(re.escape(p) for p in players),
            capture_output=True, text=True, timeout=1200
        )
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line:
                matched_files.add(os.path.abspath(line))
    except Exception:
        return hh_files

    print(f'  Pre-filter: {len(matched_files)}/{len(hh_files)} files contain target players')
    return [f for f in hh_files if os.path.abspath(str(f)) in matched_files]


# Module-level worker config
_PARSE_CFG = {}


def _parse_worker_init(num_players, good_players, use_behavioral_stats=False):
    _PARSE_CFG['num_players'] = num_players
    _PARSE_CFG['good_players'] = good_players
    _PARSE_CFG['builder'] = ObservationBuilder(num_players=num_players)
    _PARSE_CFG['behavioral_stats'] = use_behavioral_stats


def _parse_single_file(file_path):
    """Parse one HH file — returns (hand_id, player_name, dps, 0.0) per player.

    Only collects postflop decision points from good players with known cards
    (showdown hands). net_dollars=0 since action-copy rewards don't use outcomes.
    """
    num_players = _PARSE_CFG['num_players']
    good_players = _PARSE_CFG['good_players']
    builder = _PARSE_CFG['builder']
    use_bh = _PARSE_CFG['behavioral_stats']

    file_results = []
    file_total = 0
    file_used = 0

    try:
        hands = parse_file(str(file_path))
    except Exception:
        return file_results, file_total, file_used

    for hand in hands:
        if abs(hand.big_blind_dollars - 0.05) > 0.001:
            continue
        if hand.num_active_players != num_players:
            continue

        file_total += 1
        if not hand.has_showdown:
            continue

        known_players = set(hand.shown_hands.keys())
        target = (known_players & good_players) if good_players else known_players
        if not target:
            continue

        file_used += 1

        dps = builder.build_decision_points(hand, allow_unknown_cards=False,
                                            source_file=str(file_path),
                                            behavioral_stats=use_bh)
        if not dps:
            continue

        player_dps = defaultdict(list)
        for dp in dps:
            if dp.street == 'preflop':
                continue  # checked_to is postflop only
            if dp.player_name in target:
                player_dps[dp.player_name].append(dp)

        for player_name, pdps in player_dps.items():
            file_results.append((hand.hand_id, player_name, pdps, 0.0))

    return file_results, file_total, file_used


def collect_transitions(hh_dir, num_players, max_hands, good_players,
                        no_cache, use_behavioral_stats=False, workers=0):
    """Parse HH files and build raw transitions (s, a, 0, s', done, legal)."""
    hh_path = Path(hh_dir)
    all_hh_files = sorted(hh_path.rglob('*.txt'))
    print(f'  Found {len(all_hh_files)} .txt files in {hh_dir}')

    hh_files = _prefilter_files(all_hh_files, good_players or set(), hh_dir)
    if not hh_files:
        return []

    # Cache
    player_key = '|'.join(sorted(good_players)) if good_players else 'all'
    bh_key = 'bh' if use_behavioral_stats else ''
    cache_key = (f'copy|{hh_dir}|{len(all_hh_files)}|flt{len(hh_files)}'
                 f'|np{num_players}|max{max_hands}|g{player_key}|{bh_key}')
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    cache_dir = Path(__file__).parent.parent / 'data' / '.hh_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    npz_cache = cache_dir / f'copy_{cache_hash}.npz'
    if npz_cache.exists() and not no_cache:
        print(f'  Loading cached data from {npz_cache}...')
        data = np.load(str(npz_cache))
        n = len(data['actions'])
        print(f'  Loaded {n} transitions')
        return {
            'states': data['states'], 'actions': data['actions'],
            'rewards': data['rewards'], 'next_states': data['next_states'],
            'dones': data['dones'], 'legal_masks': data['legal_masks'],
            'next_legal_masks': data['next_legal_masks'],
            'streets': data['streets'], 'source_files': [],
            'obs_size': data['states'].shape[1],
        }

    # Parallel parsing
    import multiprocessing as mp
    parse_workers = workers if workers > 0 else min(120, max(1, mp.cpu_count() - 1))

    all_results = []
    total_hands = 0
    hands_used = 0
    t0 = time.time()

    print(f'  Parsing {len(hh_files)} files with {parse_workers} workers...')

    with mp.Pool(processes=parse_workers, initializer=_parse_worker_init,
                 initargs=(num_players, good_players,
                           use_behavioral_stats)) as pool:
        results = pool.imap_unordered(_parse_single_file, hh_files, chunksize=16)
        files_done = 0
        for file_results, file_total, file_used in results:
            all_results.extend(file_results)
            total_hands += file_total
            hands_used += file_used
            files_done += 1

            if files_done % 500 == 0:
                elapsed = time.time() - t0
                print(f'  [{files_done}/{len(hh_files)}] {total_hands} hands, '
                      f'{len(all_results)} player-hands ({elapsed:.0f}s)')

            if max_hands > 0 and hands_used >= max_hands:
                pool.terminate()
                print(f'  Reached {max_hands} hands limit')
                break

    elapsed = time.time() - t0
    print(f'  Parsed {files_done} files in {elapsed:.1f}s')
    print(f'  Player-hand sequences: {len(all_results)}')

    # Build flat transition arrays
    print(f'  Building transitions...')
    t_build = time.time()
    total_trans = sum(len(dps) for _, _, dps, _ in all_results if dps)

    obs_size = (all_results[0][2][0].observation.shape[0]
                if all_results and all_results[0][2] else 41)
    action_size = 5

    states = np.zeros((total_trans, obs_size), dtype=np.float32)
    actions_arr = np.zeros(total_trans, dtype=np.int64)
    rewards_arr = np.zeros(total_trans, dtype=np.float32)
    next_states = np.zeros((total_trans, obs_size), dtype=np.float32)
    dones_arr = np.zeros(total_trans, dtype=np.float32)
    legal_masks = np.zeros((total_trans, action_size), dtype=np.float32)
    next_legal_masks = np.zeros((total_trans, action_size), dtype=np.float32)
    street_arr = np.zeros(total_trans, dtype=np.int8)

    street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    idx = 0

    for hand_id, player_name, dps, _ in all_results:
        n = len(dps)
        if n == 0:
            continue

        for i, dp in enumerate(dps):
            is_terminal = (i + 1 >= n) or (dp.actual_action == 0)

            states[idx] = dp.observation
            actions_arr[idx] = dp.actual_action
            rewards_arr[idx] = 0.0  # replaced by action-copy rewards later
            dones_arr[idx] = 1.0 if is_terminal else 0.0

            if not is_terminal and i + 1 < n:
                next_states[idx] = dps[i + 1].observation
                for a in dps[i + 1].legal_actions:
                    next_legal_masks[idx, a] = 1.0
            else:
                next_states[idx] = dp.observation
                for a in dp.legal_actions:
                    next_legal_masks[idx, a] = 1.0

            for a in dp.legal_actions:
                legal_masks[idx, a] = 1.0

            street_arr[idx] = street_map.get(dp.street, 1)
            idx += 1

    # Trim
    states = states[:idx]
    actions_arr = actions_arr[:idx]
    rewards_arr = rewards_arr[:idx]
    next_states = next_states[:idx]
    dones_arr = dones_arr[:idx]
    legal_masks = legal_masks[:idx]
    next_legal_masks = next_legal_masks[:idx]
    street_arr = street_arr[:idx]

    print(f'  Total raw transitions: {idx} ({time.time() - t_build:.0f}s)')

    # Cache
    try:
        np.savez_compressed(
            str(npz_cache),
            states=states, actions=actions_arr, rewards=rewards_arr,
            next_states=next_states, dones=dones_arr,
            legal_masks=legal_masks, next_legal_masks=next_legal_masks,
            streets=street_arr,
        )
        mb = npz_cache.stat().st_size / 1e6
        print(f'  Cached to {npz_cache} ({mb:.0f} MB)')
    except Exception as e:
        print(f'  Cache save failed: {e}')

    return {
        'states': states, 'actions': actions_arr, 'rewards': rewards_arr,
        'next_states': next_states, 'dones': dones_arr,
        'legal_masks': legal_masks, 'next_legal_masks': next_legal_masks,
        'streets': street_arr, 'source_files': [],
        'obs_size': obs_size,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Build Action-Copy Dataset
# ──────────────────────────────────────────────────────────────────────────────

def build_action_copy_dataset(trans_data):
    """Filter checked_to spots and expand with action-similarity rewards.

    For each decision point, creates one training sample per legal action.
    The player's actual action gets full reward; similar actions get partial
    credit; dissimilar actions get negative reward.

    Counterfactual actions (not the player's actual) are marked terminal
    (done=1) so bootstrapping only uses real next-states.
    """
    states = trans_data['states'].copy()
    actions = trans_data['actions'].copy()
    next_states = trans_data['next_states'].copy()
    dones = trans_data['dones'].copy()
    legal_masks = trans_data['legal_masks'].copy()
    next_legal_masks = trans_data['next_legal_masks'].copy()
    streets = trans_data['streets'].copy()
    n_raw = len(states)

    # ── 1. Filter to checked_to spots (pot_odds <= 0.001) ──
    if states.shape[1] > 10:
        pot_odds = states[:, 10]
        ct_mask = pot_odds <= 0.001
    else:
        ct_mask = np.ones(n_raw, dtype=bool)

    states = states[ct_mask]
    actions = actions[ct_mask]
    next_states = next_states[ct_mask]
    dones = dones[ct_mask]
    legal_masks = legal_masks[ct_mask]
    next_legal_masks = next_legal_masks[ct_mask]
    streets = streets[ct_mask]

    n_ct = len(states)
    print(f'  Checked-to filter: {n_raw:,} -> {n_ct:,} '
          f'({100*n_ct/max(1,n_raw):.1f}%)')

    # ── 2. Remap fold -> check (folding when checked to makes no sense) ──
    n_folds = (actions == 0).sum()
    actions[actions == 0] = 1
    legal_masks[:, 0] = 0.0
    next_legal_masks[:, 0] = 0.0
    if n_folds > 0:
        print(f'  Remapped {n_folds:,} fold -> check')

    # ── 3. Add equity_vs_odds feature ──
    obs_size = states.shape[1]
    if obs_size >= 19:
        evo_s = np.clip(
            states[:, 18] - (1.0 - states[:, 10]), -1.0, 1.0
        ).reshape(-1, 1).astype(np.float32)
        evo_ns = np.clip(
            next_states[:, 18] - (1.0 - next_states[:, 10]), -1.0, 1.0
        ).reshape(-1, 1).astype(np.float32)
        states = np.hstack([states, evo_s])
        next_states = np.hstack([next_states, evo_ns])
        print(f'  Added equity_vs_odds -> obs_dim={states.shape[1]}')

    # ── 4. Expand: one sample per legal action per decision ──
    # Vectorized: iterate over target actions, not over samples
    expanded = []

    for a in range(1, 5):  # check=1, raise_half=2, raise_pot=3, all_in=4
        legal_for_a = legal_masks[:, a] > 0
        n_a = int(legal_for_a.sum())
        if n_a == 0:
            continue

        player_acts = actions[legal_for_a]
        # Vectorized reward lookup from COPY_REWARD matrix
        rewards_a = COPY_REWARD[player_acts, a]

        # Counterfactual actions (not player's actual) -> mark terminal
        is_actual = (player_acts == a)
        cf_dones = dones[legal_for_a].copy()
        cf_dones[~is_actual] = 1.0

        expanded.append({
            'states': states[legal_for_a],
            'actions': np.full(n_a, a, dtype=np.int64),
            'rewards': rewards_a,
            'next_states': next_states[legal_for_a],
            'dones': cf_dones,
            'legal_masks': legal_masks[legal_for_a],
            'next_legal_masks': next_legal_masks[legal_for_a],
            'streets': streets[legal_for_a],
            'is_primary': is_actual.astype(np.float32),
        })

    # Concatenate
    all_states = np.concatenate([e['states'] for e in expanded])
    all_actions = np.concatenate([e['actions'] for e in expanded])
    all_rewards = np.concatenate([e['rewards'] for e in expanded])
    all_next_states = np.concatenate([e['next_states'] for e in expanded])
    all_dones = np.concatenate([e['dones'] for e in expanded])
    all_legal = np.concatenate([e['legal_masks'] for e in expanded])
    all_next_legal = np.concatenate([e['next_legal_masks'] for e in expanded])
    all_streets = np.concatenate([e['streets'] for e in expanded])
    all_is_primary = np.concatenate([e['is_primary'] for e in expanded])

    n_expanded = len(all_states)
    n_primary = int(all_is_primary.sum())
    print(f'  Expansion: {n_ct:,} decisions -> {n_expanded:,} samples '
          f'({n_primary:,} primary, {n_expanded - n_primary:,} counterfactual)')

    # ── 5. Street weights ──
    street_w = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    weights = street_w[all_streets]
    weights /= weights.mean()

    # ── 6. Stats ──
    action_names = {1: 'check', 2: 'raise_half', 3: 'raise_pot', 4: 'all_in'}
    print(f'  Reward stats: mean={all_rewards.mean():.3f}, '
          f'std={all_rewards.std():.3f}, '
          f'min={all_rewards.min():.2f}, max={all_rewards.max():.2f}')

    # Primary action distribution (what good players actually did)
    primary_mask = all_is_primary > 0.5
    primary_acts = all_actions[primary_mask]
    print(f'  Player action distribution (primary only):')
    for a in range(1, 5):
        c = int((primary_acts == a).sum())
        if c > 0 or a <= 2:
            print(f'    {action_names.get(a, str(a)):<12} {c:>8}  '
                  f'({c/max(1,len(primary_acts))*100:.1f}%)')

    return (all_states, all_actions, all_rewards, all_next_states, all_dones,
            all_legal, all_next_legal, weights, all_is_primary)


# ──────────────────────────────────────────────────────────────────────────────
# CQL Training
# ──────────────────────────────────────────────────────────────────────────────

def cql_train_epoch(q_net, q_target, dataloader, optimizer, device, scaler,
                    alpha_cql=1.0, gamma=0.0):
    """One epoch of CQL training. Returns avg loss components."""
    q_net.train()
    total_td = 0.0
    total_cql = 0.0
    total_loss = 0.0
    total_q = 0.0
    n = 0

    for batch in dataloader:
        obs, acts, rews, next_obs, done, legal, next_legal, wts, _is_primary = [
            b.to(device, non_blocking=True) for b in batch]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            q_values = q_net(obs)
            q_taken = q_values.gather(1, acts.unsqueeze(1)).squeeze(1)

            # TD target: gamma=0 -> target = reward (no bootstrapping)
            if gamma > 0:
                with torch.no_grad():
                    next_q_online = q_net(next_obs)
                    next_q_masked = next_q_online + (1.0 - next_legal) * (-1e9)
                    best_a = next_q_masked.argmax(dim=1, keepdim=True)
                    next_q_target = q_target(next_obs)
                    v_next = next_q_target.gather(1, best_a).squeeze(1)
                    target = rews + gamma * v_next * (1.0 - done)
            else:
                target = rews

            td_loss = (wts * F.smooth_l1_loss(q_taken, target, reduction='none')).mean()

            # CQL conservative penalty
            q_legal = q_values + (1.0 - legal) * (-1e9)
            logsumexp_q = torch.logsumexp(q_legal, dim=1)
            cql_loss = (logsumexp_q - q_taken).mean()

            loss = td_loss + alpha_cql * cql_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()

        bs = obs.size(0)
        total_td += td_loss.item() * bs
        total_cql += cql_loss.item() * bs
        total_loss += loss.item() * bs
        total_q += q_taken.mean().item() * bs
        n += bs

    return {
        'td_loss': total_td / n,
        'cql_loss': total_cql / n,
        'total_loss': total_loss / n,
        'q_mean': total_q / n,
    }


@torch.no_grad()
def cql_eval_epoch(q_net, q_target, dataloader, device,
                   alpha_cql=1.0, gamma=0.0):
    """Evaluate CQL. Accuracy = % of primary samples where model picks player's action."""
    q_net.eval()
    total_td = 0.0
    total_cql = 0.0
    total_loss = 0.0
    total_q = 0.0
    correct = 0
    n_primary = 0
    n = 0

    for batch in dataloader:
        obs, acts, rews, next_obs, done, legal, next_legal, wts, is_primary = [
            b.to(device, non_blocking=True) for b in batch]

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            q_values = q_net(obs)
            q_taken = q_values.gather(1, acts.unsqueeze(1)).squeeze(1)

            if gamma > 0:
                next_q_online = q_net(next_obs)
                next_q_masked = next_q_online + (1.0 - next_legal) * (-1e9)
                best_a = next_q_masked.argmax(dim=1, keepdim=True)
                next_q_target = q_target(next_obs)
                v_next = next_q_target.gather(1, best_a).squeeze(1)
                target = rews + gamma * v_next * (1.0 - done)
            else:
                target = rews

            td_loss = (wts * F.smooth_l1_loss(q_taken, target, reduction='none')).mean()
            q_legal = q_values + (1.0 - legal) * (-1e9)
            logsumexp_q = torch.logsumexp(q_legal, dim=1)
            cql_loss = (logsumexp_q - q_taken).mean()
            loss = td_loss + alpha_cql * cql_loss

        # Accuracy on primary samples only (model picks = player's action?)
        primary = is_primary > 0.5
        if primary.any():
            preds = (q_values[primary] + (1.0 - legal[primary]) * (-1e9)).argmax(dim=-1)
            correct += (preds == acts[primary]).sum().item()
            n_primary += primary.sum().item()

        bs = obs.size(0)
        total_td += td_loss.item() * bs
        total_cql += cql_loss.item() * bs
        total_loss += loss.item() * bs
        total_q += q_taken.mean().item() * bs
        n += bs

    return {
        'td_loss': total_td / n,
        'cql_loss': total_cql / n,
        'total_loss': total_loss / n,
        'q_mean': total_q / n,
        'accuracy': correct / max(1, n_primary),
    }


def soft_update(target_net, online_net, tau=0.005):
    """Polyak averaging: target <- tau * online + (1 - tau) * target."""
    for tp, op in zip(target_net.parameters(), online_net.parameters()):
        tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Action-Copy CQL: learn to copy good players when checked to')

    # Data
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--hh-dir', type=str, default='data/poker_archive/')
    parser.add_argument('--good-players', type=str, default='',
                        help='Path to file with good player names (one per line)')
    parser.add_argument('--max-hands', type=int, default=0)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parsing workers (0=auto)')
    parser.add_argument('--behavioral-stats', action='store_true',
                        help='Append 8-dim behavioral stats to observations')
    parser.add_argument('--training-data', type=str, default='',
                        help='Pre-built data (.npz). Skips parsing.')
    parser.add_argument('--build-data', type=str, default='',
                        help='Build data and save to this path, then exit.')

    # CQL hyperparameters
    parser.add_argument('--alpha-cql', type=float, default=0.4,
                        help='CQL conservative regularizer strength')
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='Discount factor (0=immediate reward, no bootstrapping)')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Target network soft update rate (only used if gamma>0)')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=65536)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--hidden-layers', type=str, default='512,512,256')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Softmax temperature for policy extraction')

    # Validation
    parser.add_argument('--val-interval', type=int, default=10)
    parser.add_argument('--val-games', type=int, default=500)

    # Output
    parser.add_argument('--output', type=str, default='models/cql_copy_ct.pt')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config) as _f:
            _cfg = json.load(_f)
        _data = _cfg.get('data', {})
        _train = _cfg.get('train', {})
        if not args.hh_dir or args.hh_dir == 'data/poker_archive/':
            args.hh_dir = _data.get('hh_dir', args.hh_dir)
        if not args.good_players:
            args.good_players = _data.get('good_players', '')
        if not args.behavioral_stats:
            args.behavioral_stats = _train.get('behavioral_stats', False)
        if args.hidden_layers == '512,512,256':
            args.hidden_layers = ','.join(
                str(x) for x in _train.get('hidden_layers', [512, 512, 256]))
        if args.workers == 0:
            args.workers = _data.get('workers', 0)
        for key in ['epochs', 'lr', 'batch_size', 'dropout',
                     'alpha_cql', 'gamma', 'tau', 'temperature']:
            if key in _train:
                cli_default = parser.get_default(key.replace('_', '-') if '-' in key else key)
                if getattr(args, key.replace('-', '_')) == cli_default:
                    setattr(args, key.replace('-', '_'), _train[key])
        if _train.get('output') and args.output == 'models/cql_copy_ct.pt':
            args.output = _train['output']

    # Load player list
    def _load_player_list(path):
        if not path:
            return set()
        p = Path(path)
        if not p.exists():
            print(f'Player file not found: {p}')
            return set()
        return set(line.strip() for line in p.read_text().splitlines()
                   if line.strip() and not line.strip().startswith('#'))

    good_players = _load_player_list(args.good_players)
    if good_players:
        print(f'Good players ({len(good_players)}): loaded')
    else:
        print(f'No --good-players specified, using ALL showdown players')

    use_bh = bool(args.behavioral_stats)
    if use_bh:
        from src.hand_history import BEHAVIORAL_STATS_DIM
        print(f'Behavioral stats: ENABLED (+{BEHAVIORAL_STATS_DIM} dims)')

    num_players = 6
    action_size = 5
    hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # ── Header ───────────────────────────────────────────────────────────
    print('=' * 60)
    print('ACTION-COPY CQL  —  CHECKED-TO SPOTS')
    print('=' * 60)
    print(f'  device     : {device}')
    if device.type == 'cuda':
        print(f'  GPU        : {torch.cuda.get_device_name(0)}')
    print(f'  players    : {num_players}')
    print(f'  epochs     : {args.epochs}')
    print(f'  batch      : {args.batch_size}')
    print(f'  lr         : {args.lr}')
    print(f'  layers     : {hidden_layers}')
    print(f'  alpha_cql  : {args.alpha_cql}')
    print(f'  gamma      : {args.gamma}')
    print(f'  temperature: {args.temperature}')
    print(f'  behavioral : {use_bh}')
    print(f'  mode       : {"immediate reward" if args.gamma == 0 else "bootstrapped (gamma=" + str(args.gamma) + ")"}')
    print('=' * 60)

    # ── Collect data ─────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  COLLECTING TRAINING DATA')
    print(f'{"="*60}')

    if args.training_data:
        print(f'  Loading pre-built data from {args.training_data}...')
        t0 = time.time()
        data = np.load(args.training_data)
        trans_data = {
            'states': data['states'], 'actions': data['actions'],
            'rewards': data['rewards'], 'next_states': data['next_states'],
            'dones': data['dones'], 'legal_masks': data['legal_masks'],
            'next_legal_masks': data['next_legal_masks'],
            'streets': data['streets'], 'source_files': [],
            'obs_size': data['states'].shape[1],
        }
        print(f'  Loaded {len(trans_data["actions"])} transitions ({time.time()-t0:.0f}s)')
    else:
        trans_data = collect_transitions(
            args.hh_dir, num_players, args.max_hands,
            good_players, args.no_cache,
            use_behavioral_stats=use_bh, workers=args.workers)

        if args.build_data:
            out = Path(args.build_data)
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f'  Saving data to {out}...')
            np.savez_compressed(str(out),
                states=trans_data['states'], actions=trans_data['actions'],
                rewards=trans_data['rewards'], next_states=trans_data['next_states'],
                dones=trans_data['dones'], legal_masks=trans_data['legal_masks'],
                next_legal_masks=trans_data['next_legal_masks'],
                streets=trans_data['streets'])
            mb = out.stat().st_size / 1e6
            print(f'  Saved ({mb:.0f} MB). Use --training-data {out} to skip parsing.')
            if not args.epochs:
                return

    n_trans = len(trans_data['actions'])
    if n_trans < 100:
        print(f'Not enough data: {n_trans} transitions.')
        sys.exit(1)

    # ── Build action-copy dataset ────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  BUILDING ACTION-COPY DATASET')
    print(f'{"="*60}')

    arrays = build_action_copy_dataset(trans_data)
    (states, actions, rewards, next_states, dones,
     legal_masks, next_legal_masks, weights, is_primary) = arrays
    obs_size = states.shape[1]

    print(f'  Observation size: {obs_size}')
    print(f'  Total samples: {len(states):,}')

    if len(states) < 100:
        print(f'Not enough checked-to data.')
        sys.exit(1)

    # ── Train/val split ──────────────────────────────────────────────────
    n = len(states)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_n = int(n * args.val_split)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    train_dataset = _ActionCopyDataset(
        states, actions, rewards, next_states, dones,
        legal_masks, next_legal_masks, weights, is_primary, train_idx)
    val_dataset = _ActionCopyDataset(
        states, actions, rewards, next_states, dones,
        legal_masks, next_legal_masks, weights, is_primary, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=24, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=24, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4)

    print(f'  Train: {len(train_idx):,}, Val: {len(val_idx):,}')

    # ── Model ────────────────────────────────────────────────────────────
    q_net = QNetwork(obs_size, action_size, hidden_layers, args.dropout).to(device)
    q_target = QNetwork(obs_size, action_size, hidden_layers, args.dropout).to(device)
    q_target.load_state_dict(q_net.state_dict())
    q_target.eval()

    total_params = sum(p.numel() for p in q_net.parameters())
    print(f'  Q-network params: {total_params:,}')

    optimizer = torch.optim.AdamW(q_net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    start_epoch = 0
    best_val_acc = 0.0
    _val_acc_history = []
    best_epoch = 0

    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device)
        q_net.load_state_dict(resume_ckpt['model_state_dict'])
        if 'target_state_dict' in resume_ckpt:
            q_target.load_state_dict(resume_ckpt['target_state_dict'])
        else:
            q_target.load_state_dict(resume_ckpt['model_state_dict'])
        start_epoch = resume_ckpt.get('epoch', 0)
        print(f'  Resumed from {args.resume} (epoch {start_epoch})')
        for _ in range(start_epoch):
            scheduler.step()

    # Validation engine + preflop ranges
    engine = _make_engine(num_players)
    from src.preflop_ranges import PreflopRanges
    preflop_ranges = PreflopRanges()
    print(f'  Preflop ranges loaded (GTO locked)')

    # ── Training loop ────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  TRAINING ACTION-COPY CQL')
    print(f'{"="*60}')

    patience = 50
    patience_counter = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _ALPHA_MIN = 0.33
    _ALPHA_MAX = 0.43
    _alpha_rng = np.random.RandomState(42)
    print(f'  Alpha range: [{_ALPHA_MIN:.2f}, {_ALPHA_MAX:.2f}] (random per epoch)')

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        current_alpha = round(_alpha_rng.uniform(_ALPHA_MIN, _ALPHA_MAX), 3)

        train_metrics = cql_train_epoch(
            q_net, q_target, train_loader, optimizer, device, scaler,
            current_alpha, args.gamma)

        # Soft update target (only meaningful if gamma > 0)
        if args.gamma > 0:
            soft_update(q_target, q_net, args.tau)

        val_metrics = cql_eval_epoch(
            q_net, q_target, val_loader, device, current_alpha, args.gamma)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        # Track best (smoothed val_acc)
        improved = ''
        val_acc = val_metrics['accuracy']
        _val_acc_history.append(val_acc)
        _smooth_window = min(5, len(_val_acc_history))
        smoothed_acc = sum(_val_acc_history[-_smooth_window:]) / _smooth_window
        if smoothed_acc > best_val_acc:
            best_val_acc = smoothed_acc
            best_epoch = epoch
            patience_counter = 0
            improved = ' *best*'

            save_dict = {
                'model_state_dict': q_net.state_dict(),
                'target_state_dict': q_target.state_dict(),
                'obs_size': obs_size,
                'action_size': action_size,
                'hidden_layers': hidden_layers,
                'dropout': args.dropout,
                'num_players': num_players,
                'epoch': epoch,
                'alpha_cql': current_alpha,
                'gamma': args.gamma,
                'temperature': args.temperature,
                'model_type': 'cql_action_copy',
                'behavioral_stats': use_bh,
            }
            torch.save(save_dict, str(output_path))
        else:
            patience_counter += 1

        print(f'  Epoch {epoch:>3}/{args.epochs}  '
              f'td={train_metrics["td_loss"]:.4f}  '
              f'cql={train_metrics["cql_loss"]:.4f}  '
              f'Q={train_metrics["q_mean"]:.3f}  '
              f'val_td={val_metrics["td_loss"]:.4f}  '
              f'val_acc={val_acc:.1%}({smoothed_acc:.1%})  '
              f'a={current_alpha:.3f}  lr={lr:.6f}  {elapsed:.1f}s{improved}')

        # Spot validation
        if args.val_interval > 0 and epoch % args.val_interval == 0:
            wrapper = CQLModelWrapper(
                q_net, obs_size, action_size, num_players, device,
                iteration=epoch, temperature=args.temperature,
                exclude_fold=True)
            val_results = run_validation(wrapper, engine, num_games=args.val_games,
                                        verbose=False, preflop_ranges=preflop_ranges)
            print(f'  {validation_summary_line(val_results)}')

        if patience_counter >= patience:
            print(f'\n  Early stopping at epoch {epoch}')
            break

    # ── Final validation ─────────────────────────────────────────────────
    best_ckpt = torch.load(str(output_path), map_location=device)
    q_net.load_state_dict(best_ckpt['model_state_dict'])
    wrapper = CQLModelWrapper(
        q_net, obs_size, action_size, num_players, device,
        iteration=best_epoch, temperature=args.temperature,
        exclude_fold=True)
    final_results = run_validation(wrapper, engine, num_games=args.val_games,
                                   verbose=False, preflop_ranges=preflop_ranges)

    print(f'\n{"="*60}')
    print(f'  ACTION-COPY CQL TRAINING COMPLETE')
    print(f'{"="*60}')
    print(f'  Best epoch:     {best_epoch}')
    print(f'  Best val acc:   {best_val_acc:.1%}')
    print(f'  Model saved to: {output_path}')
    summary = final_results.get('_summary', {})
    print(f'  Validation:     {summary.get("passed",0)}/{summary.get("total",0)} spots passed')
    print(f'  {validation_summary_line(final_results)}')
    print(f'{"="*60}')


if __name__ == '__main__':
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    main()
