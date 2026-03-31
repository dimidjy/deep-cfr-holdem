"""
Implicit Q-Learning (IQL) from hand histories.

Key difference from CQL:
- Learns V(s) separately using expectile regression
- Never queries Q-values for unseen actions (no overestimation)
- No conservative penalty needed (no alpha to tune)
- More stable with imbalanced action distributions (70% call data)

The IQL objective:
  V(s) ← expectile regression on Q(s, a_data)  (only data actions)
  Q(s,a) ← r + gamma * V(s')                    (uses V, not max Q)
  Policy ← exp(Q - V) weighted regression        (advantage-weighted)

Usage:
  python3 scripts/train_iql.py --config configs/cql_config.json --output models/iql_6p_v1.pt --no-cache
"""
import sys
import os
import warnings
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import copy
import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from collections import defaultdict

# Reuse everything from CQL: QNetwork, data collection, dataset, wrapper
from train_cql import (
    QNetwork, CQLModelWrapper, _IndexedReplayDataset,
    collect_transitions, build_cql_dataset, soft_update,
    _make_engine,
)
from validate_spots import run_validation, validation_summary_line


# ──────────────────────────────────────────────────────────────────────────────
# V-Network (state value function)
# ──────────────────────────────────────────────────────────────────────────────

class VNetwork(nn.Module):
    """V-network: maps observation → scalar state value."""

    def __init__(self, obs_size: int, hidden_layers: list, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_size = obs_size
        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(dropout))
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# IQL Loss Functions
# ──────────────────────────────────────────────────────────────────────────────

def expectile_loss(pred, target, expectile=0.7):
    """Asymmetric L2 loss for expectile regression.

    When expectile > 0.5, penalizes under-predictions more than over-predictions.
    This makes V(s) approximate the upper expectile of Q(s, a_data),
    effectively learning the value of "good" actions without querying all actions.
    """
    diff = target - pred
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


def iql_train_epoch(q_net, q_target, v_net, dataloader, q_optimizer, v_optimizer,
                    device, scaler, gamma=0.99, expectile=0.7, beta=3.0):
    """One epoch of IQL training.

    Three steps:
    1. Update V using expectile regression on Q_target(s, a_data)
    2. Update Q using Bellman with V(s') instead of max Q(s')
    3. (Policy extraction happens at inference via advantage weighting)
    """
    q_net.train()
    v_net.train()
    total_v_loss = total_q_loss = total_loss = total_q_mean = 0.0
    total_samples = 0

    for batch in dataloader:
        obs, acts, rews, next_obs, done, legal, next_legal, wts = [
            b.to(device, non_blocking=True) for b in batch]

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            # ── Step 1: Update V(s) ──
            # V should approximate the expectile of Q(s, a) for actions in the data
            with torch.no_grad():
                q_target_values = q_target(obs)
                q_taken_target = q_target_values.gather(1, acts.unsqueeze(1)).squeeze(1)

            v_pred = v_net(obs)
            v_loss = expectile_loss(v_pred, q_taken_target, expectile)

            # ── Step 2: Update Q(s,a) ──
            # Bellman target uses V(s') instead of max Q(s')
            # This is the key IQL difference — no max over unseen actions
            q_values = q_net(obs)
            q_taken = q_values.gather(1, acts.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                v_next = v_net(next_obs)
                target = rews + gamma * v_next * (1.0 - done)

            q_loss = (wts * F.smooth_l1_loss(q_taken, target, reduction='none')).mean()

            loss = q_loss + v_loss

        # Update both networks
        q_optimizer.zero_grad(set_to_none=True)
        v_optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(q_optimizer)
            scaler.unscale_(v_optimizer)
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
            scaler.step(q_optimizer)
            scaler.step(v_optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
            q_optimizer.step()
            v_optimizer.step()

        bs = obs.size(0)
        total_v_loss += v_loss.item() * bs
        total_q_loss += q_loss.item() * bs
        total_loss += loss.item() * bs
        total_q_mean += q_taken.mean().item() * bs
        total_samples += bs

    return {
        'v_loss': total_v_loss / total_samples,
        'q_loss': total_q_loss / total_samples,
        'total_loss': total_loss / total_samples,
        'q_mean': total_q_mean / total_samples,
    }


@torch.no_grad()
def iql_eval_epoch(q_net, q_target, v_net, dataloader, device, gamma=0.99, expectile=0.7):
    """Evaluate IQL on validation set."""
    q_net.eval()
    v_net.eval()
    total_v_loss = total_q_loss = total_loss = total_q_mean = 0.0
    correct = 0
    total_samples = 0

    for batch in dataloader:
        obs, acts, rews, next_obs, done, legal, next_legal, wts = [
            b.to(device, non_blocking=True) for b in batch]

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            q_target_values = q_target(obs)
            q_taken_target = q_target_values.gather(1, acts.unsqueeze(1)).squeeze(1)

            v_pred = v_net(obs)
            v_loss = expectile_loss(v_pred, q_taken_target, expectile)

            q_values = q_net(obs)
            q_taken = q_values.gather(1, acts.unsqueeze(1)).squeeze(1)

            v_next = v_net(next_obs)
            target = rews + gamma * v_next * (1.0 - done)
            q_loss = (wts * F.smooth_l1_loss(q_taken, target, reduction='none')).mean()

        q_masked = q_values + (1.0 - legal) * (-1e9)
        correct += (q_masked.argmax(dim=-1) == acts).sum().item()

        bs = obs.size(0)
        total_v_loss += v_loss.item() * bs
        total_q_loss += q_loss.item() * bs
        total_loss += (q_loss.item() + v_loss.item()) * bs
        total_q_mean += q_taken.mean().item() * bs
        total_samples += bs

    return {
        'v_loss': total_v_loss / total_samples,
        'q_loss': total_q_loss / total_samples,
        'total_loss': total_loss / total_samples,
        'q_mean': total_q_mean / total_samples,
        'accuracy': correct / total_samples,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='IQL (Implicit Q-Learning) from hand histories')

    # Data
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--hh-dir', type=str, default='data/poker_archive/')
    parser.add_argument('--good-players', type=str, default='')
    parser.add_argument('--bad-players', type=str, default='')
    parser.add_argument('--max-hands', type=int, default=0)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--behavioral-stats', action='store_true')
    parser.add_argument('--training-data', type=str, default='',
                        help='Pre-built data (.npz from CQL --build-data). Skips parsing.')
    parser.add_argument('--build-data', type=str, default='',
                        help='Save parsed transitions to .npz file.')

    # IQL hyperparameters
    parser.add_argument('--expectile', type=float, default=0.7,
                        help='Expectile for V regression (0.5=median, 0.9=optimistic)')
    parser.add_argument('--beta', type=float, default=3.0,
                        help='Inverse temperature for advantage-weighted policy')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=65536)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--hidden-layers', type=str, default='512,512,256')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--val-interval', type=int, default=10)
    parser.add_argument('--val-games', type=int, default=500)

    # Output
    parser.add_argument('--output', type=str, default='models/iql_model.pt')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        _data = cfg.get('data', {})
        _train = cfg.get('train', {})
        if args.hh_dir == 'data/poker_archive/':
            args.hh_dir = _data.get('hh_dir', args.hh_dir)
        if not args.workers:
            args.workers = _data.get('workers', 0)
        if not args.behavioral_stats:
            args.behavioral_stats = _train.get('behavioral_stats', False)
        if args.hidden_layers == '512,512,256':
            args.hidden_layers = ','.join(str(x) for x in _train.get('hidden_layers', [512, 512, 256]))
        if _train.get('output') and args.output == 'models/iql_model.pt':
            args.output = _train['output'].replace('cql', 'iql')

    postflop_only = True
    num_players = 6
    action_size = 5
    hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    use_bh = bool(args.behavioral_stats)

    # ── Header ──
    print('=' * 60)
    print('IQL  —  IMPLICIT Q-LEARNING')
    print('=' * 60)
    print(f'  device    : {device}')
    if device.type == 'cuda':
        print(f'  GPU       : {torch.cuda.get_device_name(0)}')
    print(f'  epochs    : {args.epochs}')
    print(f'  batch     : {args.batch_size}')
    print(f'  lr        : {args.lr}')
    print(f'  layers    : {hidden_layers}')
    print(f'  expectile : {args.expectile}')
    print(f'  beta      : {args.beta}')
    print(f'  gamma     : {args.gamma}')
    print(f'  behavioral: {use_bh}')
    print('=' * 60)

    # ── Load player lists ──
    def _load_player_list(path):
        if not path or not Path(path).exists():
            return set()
        names = set()
        for line in open(path):
            line = line.strip()
            if line and not line.startswith('#'):
                names.add(line)
        return names

    good_players = _load_player_list(args.good_players)
    bad_players = _load_player_list(args.bad_players)
    if good_players:
        print(f'Good players ({len(good_players)}): loaded')
    if bad_players:
        print(f'Bad players ({len(bad_players)}): loaded')

    # ── Collect data (reuses CQL infrastructure) ──
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
            good_players, bad_players, args.no_cache, postflop_only,
            use_behavioral_stats=use_bh, workers=args.workers)

        # Save data if --build-data specified
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
    print(f'  Transitions: {n_trans}')

    # ── Build dataset ──
    print(f'Building IQL dataset...')
    arrays = build_cql_dataset(trans_data)
    states, actions, rewards, next_states, dones, legal_masks, next_legal_masks, weights = arrays
    obs_size = states.shape[1]

    print(f'  Observation size: {obs_size}')
    print(f'  Total transitions: {len(states)}')

    # Split
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(states))
    val_n = int(len(states) * args.val_split)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    train_dataset = _IndexedReplayDataset(
        states, actions, rewards, next_states, dones,
        legal_masks, next_legal_masks, weights, train_idx)
    val_dataset = _IndexedReplayDataset(
        states, actions, rewards, next_states, dones,
        legal_masks, next_legal_masks, weights, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=24, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=24, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4)

    print(f'  Train: {len(train_idx)}, Val: {len(val_idx)}')

    # ── Models ──
    q_net = QNetwork(obs_size, action_size, hidden_layers, args.dropout).to(device)
    q_target = QNetwork(obs_size, action_size, hidden_layers, args.dropout).to(device)
    q_target.load_state_dict(q_net.state_dict())
    q_target.eval()
    v_net = VNetwork(obs_size, hidden_layers, args.dropout).to(device)

    q_params = sum(p.numel() for p in q_net.parameters())
    v_params = sum(p.numel() for p in v_net.parameters())
    print(f'  Q-network params: {q_params:,}')
    print(f'  V-network params: {v_params:,}')

    q_optimizer = torch.optim.AdamW(q_net.parameters(), lr=args.lr, weight_decay=1e-4)
    v_optimizer = torch.optim.AdamW(v_net.parameters(), lr=args.lr, weight_decay=1e-4)
    q_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    v_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        v_optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        q_net.load_state_dict(ckpt['model_state_dict'])
        if 'target_state_dict' in ckpt:
            q_target.load_state_dict(ckpt['target_state_dict'])
        if 'v_state_dict' in ckpt:
            v_net.load_state_dict(ckpt['v_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f'  Resumed from {args.resume} (epoch {start_epoch})')
        for _ in range(start_epoch):
            q_scheduler.step()
            v_scheduler.step()

    # Validation
    engine = _make_engine(num_players)
    from src.preflop_ranges import PreflopRanges
    preflop_ranges = PreflopRanges()
    print(f'  Preflop ranges loaded')

    # ── Training ──
    print(f'\n{"="*60}')
    print(f'  TRAINING IQL')
    print(f'{"="*60}')

    best_val_loss = float('inf')
    best_epoch = 0
    patience = 30
    patience_counter = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        train_metrics = iql_train_epoch(
            q_net, q_target, v_net, train_loader, q_optimizer, v_optimizer,
            device, scaler, args.gamma, args.expectile, args.beta)

        soft_update(q_target, q_net, args.tau)

        val_metrics = iql_eval_epoch(
            q_net, q_target, v_net, val_loader, device, args.gamma, args.expectile)

        q_scheduler.step()
        v_scheduler.step()
        lr = q_optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        improved = ''
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_epoch = epoch
            patience_counter = 0
            improved = ' *best*'

            torch.save({
                'model_state_dict': q_net.state_dict(),
                'target_state_dict': q_target.state_dict(),
                'v_state_dict': v_net.state_dict(),
                'obs_size': obs_size,
                'action_size': action_size,
                'hidden_layers': hidden_layers,
                'dropout': args.dropout,
                'num_players': num_players,
                'epoch': epoch,
                'expectile': args.expectile,
                'beta': args.beta,
                'gamma': args.gamma,
                'temperature': args.temperature,
                'model_type': 'iql',
                'behavioral_stats': use_bh,
            }, str(output_path))
        else:
            patience_counter += 1

        print(f'  Epoch {epoch:>3}/{args.epochs}  '
              f'q={train_metrics["q_loss"]:.4f}  v={train_metrics["v_loss"]:.4f}  '
              f'Q={train_metrics["q_mean"]:.3f}  '
              f'val_q={val_metrics["q_loss"]:.4f}  val_acc={val_metrics["accuracy"]:.1%}  '
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

    # Final
    best_ckpt = torch.load(str(output_path), map_location=device)
    q_net.load_state_dict(best_ckpt['model_state_dict'])
    wrapper = CQLModelWrapper(
        q_net, obs_size, action_size, num_players, device,
        iteration=best_epoch, temperature=args.temperature)
    final_results = run_validation(wrapper, engine, num_games=args.val_games,
                                    verbose=False, preflop_ranges=preflop_ranges)

    print(f'\n{"="*60}')
    print(f'  IQL TRAINING COMPLETE')
    print(f'{"="*60}')
    print(f'  Best epoch:     {best_epoch}')
    print(f'  Best val loss:  {best_val_loss:.4f}')
    print(f'  Model saved to: {output_path}')
    print(f'  {validation_summary_line(final_results)}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
