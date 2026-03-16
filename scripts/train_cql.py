"""
Conservative Q-Learning (CQL) from hand histories.

Offline RL approach: learns Q-values (expected reward per action) directly from
logged hand history data. Unlike behavioral cloning, CQL naturally learns when
to fold (negative Q-value actions) without needing synthetic fold labels.

The conservative regularizer penalizes Q-values on out-of-distribution actions,
keeping the policy close to the behavior policy while still improving.

Usage:
  # Train on all showdown players
  python3 scripts/train_cql.py --epochs 100 --output models/cql_v1.pt --no-cache

  # Train on good players only
  python3 scripts/train_cql.py --good-players data/good_players.txt --output models/cql_v1.pt --no-cache
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
# Q-Network (reuses AdvantageNetwork architecture)
# ──────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """Q-network: maps observation → Q-value for each action."""

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
                 iteration=0, temperature=1.0):
        self.action_size = action_size
        self.num_players = num_players
        self.device = torch.device(device)
        self.iteration = iteration
        self.temperature = temperature

        # Wrap Q-net as a StrategyNetwork-like object for validation
        self.q_net = q_net
        self.strategy_networks = [q_net] * num_players
        self.advantage_networks = [q_net] * num_players

    def get_strategy(self, observation: np.ndarray, player_idx: int,
                     legal_actions: list) -> np.ndarray:
        obs_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t).cpu().numpy()[0]

        # Softmax over legal Q-values
        masked = np.full(self.action_size, -1e9, dtype=np.float32)
        for a in legal_actions:
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
# Offline Replay Dataset — stores (s, a, r, s', done, legal, legal_next)
# ──────────────────────────────────────────────────────────────────────────────

class OfflineReplayDataset(Dataset):
    """Dataset of (state, action, reward, next_state, done, legal_mask, next_legal_mask)."""

    def __init__(self, states, actions, rewards, next_states, dones,
                 legal_masks, next_legal_masks, weights):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)
        self.rewards = torch.FloatTensor(rewards)
        self.next_states = torch.FloatTensor(next_states)
        self.dones = torch.FloatTensor(dones)
        self.legal_masks = torch.FloatTensor(legal_masks)
        self.next_legal_masks = torch.FloatTensor(next_legal_masks)
        self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx],
                self.legal_masks[idx], self.next_legal_masks[idx],
                self.weights[idx])


# ──────────────────────────────────────────────────────────────────────────────
# Data collection (reuses parse infrastructure from train_supervised)
# ──────────────────────────────────────────────────────────────────────────────

def _prefilter_files(hh_files, good_players, hh_dir):
    """Grep pre-filter — same as train_supervised."""
    import subprocess
    import shutil
    import re

    if not good_players:
        return hh_files

    escaped = [re.escape(name) for name in good_players]
    BATCH = 50
    pattern_batches = [escaped[i:i+BATCH] for i in range(0, len(escaped), BATCH)]
    matched_files = set()
    use_rg = shutil.which('rg') is not None

    for batch in pattern_batches:
        pattern = '|'.join(batch)
        try:
            if use_rg:
                cmd = ['rg', '-l', '--no-ignore', '--glob', '*.txt',
                       '-e', pattern, hh_dir]
            else:
                cmd = ['grep', '-rl', '-E', pattern, '--include=*.txt', hh_dir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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


def _parse_worker_init(num_players, good_players, postflop_only):
    _PARSE_CFG['num_players'] = num_players
    _PARSE_CFG['good_players'] = good_players
    _PARSE_CFG['postflop_only'] = postflop_only
    _PARSE_CFG['builder'] = ObservationBuilder(num_players=num_players)


def _parse_single_file(file_path):
    """Parse one HH file — returns list of (hand_id, player_name, dps_for_player).

    For CQL we collect TWO types of data:
    1. Known-cards players (showdown): full obs with real equity, all their actions
    2. Fold-only transitions from ALL players: we parse with allow_unknown_cards=True
       but only keep the final fold action. Fold reward = -invested (known exactly).
       The unknown-card obs has approximate equity (~0.45) but CQL relies on reward
       signal, not features, so this is acceptable.
    """
    num_players = _PARSE_CFG['num_players']
    good_players = _PARSE_CFG['good_players']
    postflop_only = _PARSE_CFG['postflop_only']
    builder = _PARSE_CFG['builder']

    file_results = []  # list of (hand_id, player_name, sorted_dps, net_dollars)
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
        if good_players:
            target_known = known_players & good_players
        else:
            target_known = known_players

        if not target_known:
            continue

        # Compute player net profit
        player_winnings = {}
        for w in hand.result.winners:
            name = w['name']
            player_winnings[name] = player_winnings.get(name, 0) + w['amount_dollars']

        player_invested = {}
        for action in hand.actions:
            if action.is_blind or action.action_type in ('bet', 'raise', 'call'):
                name = action.player_name
                player_invested[name] = player_invested.get(name, 0) + action.amount_dollars

        file_used += 1

        # ── 1) Known-cards players: full decision sequences ──────────────
        dps = builder.build_decision_points(hand, allow_unknown_cards=False,
                                            source_file=str(file_path))
        if dps:
            player_dps = defaultdict(list)
            for dp in dps:
                if postflop_only and dp.street == 'preflop':
                    continue
                if dp.player_name in target_known:
                    player_dps[dp.player_name].append(dp)

            for player_name, pdps in player_dps.items():
                net = player_winnings.get(player_name, 0.0) - player_invested.get(player_name, 0.0)
                file_results.append((hand.hand_id, player_name, pdps, net))

        # ── 2) Fold transitions from ALL players (unknown cards OK) ──────
        # Parse again with allow_unknown_cards=True to get fold actions
        # from players whose cards we don't know
        dps_all = builder.build_decision_points(hand, allow_unknown_cards=True,
                                                source_file=str(file_path))
        if dps_all:
            # Find players who folded (not in showdown known_players)
            fold_players = set()
            for dp in dps_all:
                if dp.actual_action == 0:  # fold
                    fold_players.add(dp.player_name)

            for fold_player in fold_players:
                # Skip players we already have full data for
                if fold_player in target_known:
                    continue

                # Get only the LAST decision point (the fold itself)
                fold_dps = [dp for dp in dps_all
                            if dp.player_name == fold_player and dp.actual_action == 0]
                if not fold_dps:
                    continue
                if postflop_only and all(dp.street == 'preflop' for dp in fold_dps):
                    continue

                # Fold reward = -amount invested (they lose what they put in)
                fold_net = -player_invested.get(fold_player, 0.0)
                # Only keep the fold action itself (terminal transition)
                last_fold = fold_dps[-1]
                if postflop_only and last_fold.street == 'preflop':
                    continue
                file_results.append((hand.hand_id, fold_player, [last_fold], fold_net))

    return file_results, file_total, file_used


def collect_transitions(hh_dir, num_players, max_hands, good_players,
                        no_cache, postflop_only):
    """Parse HH files and build offline RL transitions (s, a, r, s', done)."""
    hh_path = Path(hh_dir)
    all_hh_files = sorted(hh_path.rglob('*.txt'))
    print(f'  Found {len(all_hh_files)} .txt files in {hh_dir}')

    hh_files = _prefilter_files(all_hh_files, good_players, hh_dir)
    if not hh_files:
        return []

    # Cache
    player_key = '|'.join(sorted(good_players)) if good_players else 'all'
    cache_key = f'cql1|{hh_dir}|{len(all_hh_files)}|flt{len(hh_files)}|np{num_players}|max{max_hands}|g{player_key}|pf{postflop_only}'
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    cache_dir = Path(__file__).parent.parent / 'data' / '.hh_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'cql_{cache_hash}.pkl'

    if cache_file.exists() and not no_cache:
        print(f'  Loading cached CQL data from {cache_file.name}...')
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f'  Loaded {len(data["transitions"])} transitions from {data["hands_used"]} hands')
        return data['transitions']

    # Parallel parsing
    import multiprocessing as mp
    parse_workers = min(120, max(1, mp.cpu_count() - 1))

    all_results = []
    total_hands = 0
    hands_used = 0
    t0 = time.time()

    print(f'  Parsing {len(hh_files)} files with {parse_workers} workers...')

    with mp.Pool(processes=parse_workers, initializer=_parse_worker_init,
                 initargs=(num_players, good_players, postflop_only)) as pool:
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

    # Convert to transitions: (s, a, r, s', done, legal, legal_next)
    transitions = []
    bb = 0.05

    for hand_id, player_name, dps, net_dollars in all_results:
        n = len(dps)
        if n == 0:
            continue

        # Reward: distribute hand outcome across actions with discounting
        # Last action gets full reward, earlier actions get discounted
        gamma = 0.99
        for i, dp in enumerate(dps):
            # Discounted reward: later actions get more of the outcome
            discount = gamma ** (n - 1 - i)
            reward = (net_dollars / bb) * discount  # reward in BB units

            # Next state
            if i + 1 < n:
                next_obs = dps[i + 1].observation
                next_legal = dps[i + 1].legal_actions
                done = 0.0
            else:
                # Terminal — use same obs as dummy (won't be used due to done=1)
                next_obs = dp.observation
                next_legal = dp.legal_actions
                done = 1.0

            transitions.append({
                'obs': dp.observation,
                'action': dp.actual_action,
                'reward': reward,
                'next_obs': next_obs,
                'done': done,
                'legal': dp.legal_actions,
                'next_legal': next_legal,
                'street': dp.street,
                'source_file': dp.source_file,
            })

    print(f'  Total transitions: {len(transitions)}')

    # Cache
    data = {'transitions': transitions, 'hands_used': hands_used}
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        mb = cache_file.stat().st_size / 1e6
        print(f'  Cached to {cache_file.name} ({mb:.0f} MB)')
    except Exception as e:
        print(f'  Cache save failed: {e}')

    return transitions


def build_cql_dataset(transitions, action_size=5):
    """Convert transitions to numpy arrays for the Dataset."""
    n = len(transitions)
    obs_size = transitions[0]['obs'].shape[0]

    states = np.zeros((n, obs_size), dtype=np.float32)
    actions = np.zeros(n, dtype=np.int64)
    rewards = np.zeros(n, dtype=np.float32)
    next_states = np.zeros((n, obs_size), dtype=np.float32)
    dones = np.zeros(n, dtype=np.float32)
    legal_masks = np.zeros((n, action_size), dtype=np.float32)
    next_legal_masks = np.zeros((n, action_size), dtype=np.float32)
    weights = np.ones(n, dtype=np.float32)

    street_weights = {'preflop': 1.0, 'flop': 2.0, 'turn': 3.0, 'river': 4.0}

    for i, t in enumerate(transitions):
        states[i] = t['obs']
        actions[i] = t['action']
        rewards[i] = t['reward']
        next_states[i] = t['next_obs']
        dones[i] = t['done']
        for a in t['legal']:
            legal_masks[i, a] = 1.0
        for a in t['next_legal']:
            next_legal_masks[i, a] = 1.0
        weights[i] = street_weights.get(t['street'], 1.0)

    # Normalize rewards to have reasonable scale (mean~0, std~1)
    r_std = rewards.std()
    if r_std > 0:
        rewards /= r_std
    # Normalize weights
    w_mean = weights.mean()
    if w_mean > 0:
        weights /= w_mean

    return states, actions, rewards, next_states, dones, legal_masks, next_legal_masks, weights


# ──────────────────────────────────────────────────────────────────────────────
# CQL Training
# ──────────────────────────────────────────────────────────────────────────────

def cql_train_epoch(q_net, q_target, dataloader, optimizer, device, scaler,
                    alpha_cql=1.0, gamma=0.99):
    """One epoch of CQL training. Returns avg loss components."""
    q_net.train()
    total_td_loss = 0.0
    total_cql_loss = 0.0
    total_loss = 0.0
    total_q_mean = 0.0
    total_samples = 0

    for batch in dataloader:
        obs, acts, rews, next_obs, done, legal, next_legal, wts = [
            b.to(device, non_blocking=True) for b in batch]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            # Current Q-values
            q_values = q_net(obs)  # (B, 5)
            q_taken = q_values.gather(1, acts.unsqueeze(1)).squeeze(1)  # (B,)

            # Target Q-values (from target network)
            with torch.no_grad():
                next_q = q_target(next_obs)  # (B, 5)
                # Mask illegal next actions
                next_q_masked = next_q + (1.0 - next_legal) * (-1e9)
                # Use max for next-state value (standard Q-learning)
                v_next = next_q_masked.max(dim=1).values
                # Bellman target
                target = rews + gamma * v_next * (1.0 - done)

            # TD loss (weighted)
            td_loss = (wts * F.smooth_l1_loss(q_taken, target, reduction='none')).mean()

            # Conservative regularizer:
            # Penalize Q-values for being high on all actions (logsumexp)
            # minus Q-value of the action actually taken (from data)
            # This pushes down Q on unseen actions, keeps Q on seen actions
            q_legal = q_values + (1.0 - legal) * (-1e9)
            logsumexp_q = torch.logsumexp(q_legal, dim=1)  # (B,)
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
        total_td_loss += td_loss.item() * bs
        total_cql_loss += cql_loss.item() * bs
        total_loss += loss.item() * bs
        total_q_mean += q_taken.mean().item() * bs
        total_samples += bs

    return {
        'td_loss': total_td_loss / total_samples,
        'cql_loss': total_cql_loss / total_samples,
        'total_loss': total_loss / total_samples,
        'q_mean': total_q_mean / total_samples,
    }


@torch.no_grad()
def cql_eval_epoch(q_net, q_target, dataloader, device, alpha_cql=1.0, gamma=0.99):
    """Evaluate CQL on validation set."""
    q_net.eval()
    total_td_loss = 0.0
    total_cql_loss = 0.0
    total_loss = 0.0
    total_q_mean = 0.0
    correct = 0
    total_samples = 0

    for batch in dataloader:
        obs, acts, rews, next_obs, done, legal, next_legal, wts = [
            b.to(device, non_blocking=True) for b in batch]

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            q_values = q_net(obs)
            q_taken = q_values.gather(1, acts.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q = q_target(next_obs)
                next_q_masked = next_q + (1.0 - next_legal) * (-1e9)
                v_next = next_q_masked.max(dim=1).values
                target = rews + gamma * v_next * (1.0 - done)

            td_loss = (wts * F.smooth_l1_loss(q_taken, target, reduction='none')).mean()

            q_legal = q_values + (1.0 - legal) * (-1e9)
            logsumexp_q = torch.logsumexp(q_legal, dim=1)
            cql_loss = (logsumexp_q - q_taken).mean()

            loss = td_loss + alpha_cql * cql_loss

        # "Accuracy" = model's best action matches actual action
        q_masked = q_values + (1.0 - legal) * (-1e9)
        preds = q_masked.argmax(dim=-1)
        correct += (preds == acts).sum().item()

        bs = obs.size(0)
        total_td_loss += td_loss.item() * bs
        total_cql_loss += cql_loss.item() * bs
        total_loss += loss.item() * bs
        total_q_mean += q_taken.mean().item() * bs
        total_samples += bs

    return {
        'td_loss': total_td_loss / total_samples,
        'cql_loss': total_cql_loss / total_samples,
        'total_loss': total_loss / total_samples,
        'q_mean': total_q_mean / total_samples,
        'accuracy': correct / total_samples,
    }


def soft_update(target_net, online_net, tau=0.005):
    """Polyak averaging: target ← τ * online + (1 - τ) * target."""
    for tp, op in zip(target_net.parameters(), online_net.parameters()):
        tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='CQL (Conservative Q-Learning) from hand histories')

    # Data
    parser.add_argument('--hh-dir', type=str, default='data/poker_archive/')
    parser.add_argument('--good-players', type=str, default='')
    parser.add_argument('--max-hands', type=int, default=0)
    parser.add_argument('--no-cache', action='store_true')

    # CQL hyperparameters
    parser.add_argument('--alpha-cql', type=float, default=0.1,
                        help='CQL conservative regularizer strength (lower=more deviation from data)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Target network soft update rate')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=65536)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--hidden-layers', type=str, default='512,512,256')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Softmax temperature for policy extraction (lower=greedier)')

    # Validation
    parser.add_argument('--val-interval', type=int, default=10)
    parser.add_argument('--val-games', type=int, default=500)

    # Output
    parser.add_argument('--output', type=str, default='models/cql_model.pt')
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    postflop_only = True

    # Load player list
    good_players = set()
    if args.good_players:
        p = Path(args.good_players)
        if not p.exists():
            print(f'Good players file not found: {p}')
            sys.exit(1)
        good_players = set(
            line.strip() for line in p.read_text().splitlines()
            if line.strip() and not line.strip().startswith('#')
        )
        print(f'Good players ({len(good_players)}): loaded from {p.name}')

    num_players = 6
    action_size = 5
    hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # ── Header ───────────────────────────────────────────────────────────
    print('=' * 60)
    print('CQL  —  CONSERVATIVE Q-LEARNING')
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
    print(f'  tau        : {args.tau}')
    print(f'  temperature: {args.temperature}')
    print('=' * 60)

    # ── Collect data ─────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  COLLECTING TRAINING DATA')
    print(f'{"="*60}')

    transitions = collect_transitions(
        args.hh_dir, num_players, args.max_hands,
        good_players, args.no_cache, postflop_only)

    if len(transitions) < 100:
        print(f'Not enough data: {len(transitions)} transitions.')
        sys.exit(1)

    # ── Build dataset ────────────────────────────────────────────────────
    print(f'Building CQL dataset...')
    obs_size = transitions[0]['obs'].shape[0]
    arrays = build_cql_dataset(transitions, action_size)
    states, actions, rewards, next_states, dones, legal_masks, next_legal_masks, weights = arrays

    # Stats
    print(f'  Observation size: {obs_size}')
    print(f'  Total transitions: {len(states)}')
    print(f'  Reward stats: mean={rewards.mean():.3f}, std={rewards.std():.3f}, '
          f'min={rewards.min():.1f}, max={rewards.max():.1f}')

    action_counts = defaultdict(int)
    street_counts = defaultdict(int)
    for t in transitions:
        action_counts[t['action']] += 1
        street_counts[t['street']] += 1

    action_names = {0: 'fold', 1: 'call', 2: 'raise_half', 3: 'raise_pot', 4: 'all_in'}
    print(f'  Action distribution:')
    for a in range(action_size):
        c = action_counts.get(a, 0)
        print(f'    {action_names[a]:<12} {c:>8}  ({c/len(transitions)*100:.1f}%)')

    # ── File-based train/val split ───────────────────────────────────────
    file_to_indices = defaultdict(list)
    for i, t in enumerate(transitions):
        file_to_indices[t.get('source_file', f'unknown_{i}')].append(i)

    unique_files = sorted(file_to_indices.keys())
    rng = np.random.RandomState(42)
    rng.shuffle(unique_files)

    val_target = int(len(states) * args.val_split)
    val_indices, train_indices = [], []
    val_count = 0
    for f in unique_files:
        idxs = file_to_indices[f]
        if val_count < val_target:
            val_indices.extend(idxs)
            val_count += len(idxs)
        else:
            train_indices.extend(idxs)

    train_idx = np.array(train_indices)
    val_idx = np.array(val_indices)

    train_dataset = OfflineReplayDataset(
        states[train_idx], actions[train_idx], rewards[train_idx],
        next_states[train_idx], dones[train_idx],
        legal_masks[train_idx], next_legal_masks[train_idx], weights[train_idx])
    val_dataset = OfflineReplayDataset(
        states[val_idx], actions[val_idx], rewards[val_idx],
        next_states[val_idx], dones[val_idx],
        legal_masks[val_idx], next_legal_masks[val_idx], weights[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=8, pin_memory=True,
                            persistent_workers=True)

    print(f'  Train: {len(train_idx)}, Val: {len(val_idx)}')

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

    # Validation engine + preflop ranges
    engine = _make_engine(num_players)
    from src.preflop_ranges import PreflopRanges
    preflop_ranges = PreflopRanges()
    print(f'  Preflop ranges loaded (GTO locked)')

    # ── Training loop ────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  TRAINING CQL')
    print(f'{"="*60}')

    best_val_loss = float('inf')
    best_epoch = 0
    patience = 30
    patience_counter = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = cql_train_epoch(
            q_net, q_target, train_loader, optimizer, device, scaler,
            args.alpha_cql, args.gamma)

        # Soft update target network
        soft_update(q_target, q_net, args.tau)

        val_metrics = cql_eval_epoch(
            q_net, q_target, val_loader, device, args.alpha_cql, args.gamma)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        # Track best
        improved = ''
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
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
                'alpha_cql': args.alpha_cql,
                'gamma': args.gamma,
                'temperature': args.temperature,
                'model_type': 'cql',
            }
            torch.save(save_dict, str(output_path))
        else:
            patience_counter += 1

        print(f'  Epoch {epoch:>3}/{args.epochs}  '
              f'td={train_metrics["td_loss"]:.4f}  '
              f'cql={train_metrics["cql_loss"]:.4f}  '
              f'Q={train_metrics["q_mean"]:.3f}  '
              f'val_td={val_metrics["td_loss"]:.4f}  '
              f'val_acc={val_metrics["accuracy"]:.1%}  '
              f'lr={lr:.6f}  {elapsed:.1f}s{improved}')

        # Spot validation
        if args.val_interval > 0 and epoch % args.val_interval == 0:
            wrapper = CQLModelWrapper(
                q_net, obs_size, action_size, num_players, device,
                iteration=epoch, temperature=args.temperature)
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
        iteration=best_epoch, temperature=args.temperature)
    final_results = run_validation(wrapper, engine, num_games=args.val_games,
                                   verbose=False, preflop_ranges=preflop_ranges)

    print(f'\n{"="*60}')
    print(f'  CQL TRAINING COMPLETE')
    print(f'{"="*60}')
    print(f'  Best epoch:     {best_epoch}')
    print(f'  Best val loss:  {best_val_loss:.4f}')
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
