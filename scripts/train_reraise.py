"""
Re-Raise Classifier: binary classifier to decide "should I raise when facing a bet?"

The CQL facing_bet model always outputs fold/call but never raise, because CQL's
conservative regularizer suppresses underrepresented actions. This classifier
runs AFTER the facing_bet model: if facing_bet says "call" and this says P(raise) > threshold,
the agent upgrades to a raise.

Training data: all facing_bet postflop spots from hand histories.
Labels: 1 = player raised/all-in, 0 = player called/folded.
Good-player raises get 3x weight.

Usage:
  # Build data first (shared with CQL training)
  python3 scripts/train_reraise.py --training-data data/cql_all_bh.npz --output models/reraise_v1.pt

  # Or parse from scratch
  python3 scripts/train_reraise.py --hh-dir data/poker_archive/ --max-hands 200000 \
      --good-players data/good_players.txt --behavioral-stats --output models/reraise_v1.pt
"""
import sys
import os
import warnings
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class ReRaiseClassifier(nn.Module):
    """Binary classifier: P(raise | observation, facing_bet)."""

    def __init__(self, obs_size: int, hidden_layers: list, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = obs_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,)


class _ReRaiseDataset(Dataset):
    def __init__(self, states, labels, weights, indices):
        self.states = states
        self.labels = labels
        self.weights = weights
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (torch.from_numpy(self.states[i]),
                torch.tensor(self.labels[i], dtype=torch.float32),
                torch.tensor(self.weights[i], dtype=torch.float32))


def build_reraise_dataset(trans_data, good_player_boost=3.0):
    """Filter to facing_bet spots, create binary raise labels."""
    states = trans_data['states']
    actions = trans_data['actions']
    rewards = trans_data['rewards']

    # Filter: facing_bet only (pot_odds > 0 means there's a bet to face)
    pot_odds = states[:, 10]
    fb_mask = pot_odds > 0.001
    states = states[fb_mask]
    actions = actions[fb_mask]
    rewards = rewards[fb_mask]
    n = len(states)
    print(f'  Facing-bet spots: {n}')

    # Append equity_vs_odds feature
    if states.shape[1] >= 19:
        evo = np.clip(states[:, 18] - (1.0 - states[:, 10]), -1.0, 1.0).reshape(-1, 1)
        states = np.hstack([states, evo.astype(np.float32)])
        print(f'  Added equity_vs_odds → obs_dim={states.shape[1]}')

    # Labels: 1 = raised (action >= 2), 0 = called/folded
    labels = (actions >= 2).astype(np.float32)
    n_raises = labels.sum()
    print(f'  Raises: {int(n_raises)}/{n} ({100*n_raises/n:.1f}%)')

    # Weights: good players' raises get extra weight
    # We approximate "good player" by positive rewards (winners)
    weights = np.ones(n, dtype=np.float32)
    # Boost winning raises (reward > 0 means player won the hand)
    winning_raise = (labels == 1) & (rewards > 0)
    weights[winning_raise] *= good_player_boost
    # Also boost strong-hand raises regardless of outcome
    if states.shape[1] > 19:
        strong_raise = (labels == 1) & (states[:, 19] >= 0.25)  # two pair+
        weights[strong_raise] *= 2.0

    # Balance: oversample raises to ~40% (from typical ~5%)
    raise_idx = np.where(labels == 1)[0]
    call_idx = np.where(labels == 0)[0]
    target_raise_pct = 0.40
    target_raises = int(target_raise_pct / (1.0 - target_raise_pct) * len(call_idx))
    extra = max(0, target_raises - len(raise_idx))

    if extra > 0 and len(raise_idx) > 0:
        # Weight oversample by hand strength
        made_hand = states[raise_idx, 19] if states.shape[1] > 19 else np.zeros(len(raise_idx))
        has_draw = np.zeros(len(raise_idx), dtype=bool)
        if states.shape[1] > 21:
            has_draw = (states[raise_idx, 20] > 0.5) | (states[raise_idx, 21] > 0.5)

        os_weights = np.ones(len(raise_idx), dtype=np.float64)
        for i in range(len(raise_idx)):
            mh = made_hand[i]
            if mh >= 0.375:
                os_weights[i] = 4.0   # trips+
            elif mh >= 0.25:
                os_weights[i] = 3.0   # two pair
            elif mh >= 0.01:
                os_weights[i] = 2.0   # one pair
            elif has_draw[i]:
                os_weights[i] = 2.5   # draws (semi-bluff)
            else:
                os_weights[i] = 1.5   # air (bluff)
        os_weights /= os_weights.sum()

        rng = np.random.RandomState(42)
        os_idx = rng.choice(raise_idx, size=extra, replace=True, p=os_weights)
        states = np.concatenate([states, states[os_idx]])
        labels = np.concatenate([labels, labels[os_idx]])
        weights = np.concatenate([weights, weights[os_idx]])
        print(f'  Oversampled raises: {int(n_raises)} → {int(labels.sum())} '
              f'({100*labels.sum()/len(labels):.1f}%)')

    # Normalize weights
    weights /= weights.mean()

    return states, labels, weights


def train_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        obs, lbl, wts = [b.to(device, non_blocking=True) for b in batch]
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(obs)
            loss = F.binary_cross_entropy_with_logits(logits, lbl, weight=wts)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = obs.size(0)
        total_loss += loss.item() * bs
        preds = (logits > 0).float()
        correct += (preds == lbl).sum().item()
        total += bs

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    tp = fp = fn = tn = 0

    for batch in loader:
        obs, lbl, wts = [b.to(device, non_blocking=True) for b in batch]

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(obs)
            loss = F.binary_cross_entropy_with_logits(logits, lbl, weight=wts)

        bs = obs.size(0)
        total_loss += loss.item() * bs
        preds = (logits > 0).float()
        correct += (preds == lbl).sum().item()
        total += bs

        tp += ((preds == 1) & (lbl == 1)).sum().item()
        fp += ((preds == 1) & (lbl == 0)).sum().item()
        fn += ((preds == 0) & (lbl == 1)).sum().item()
        tn += ((preds == 0) & (lbl == 0)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'raise_rate': (tp + fp) / total if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Train re-raise binary classifier')
    parser.add_argument('--training-data', type=str, required=True,
                        help='Pre-built CQL data (.npz from train_cql.py --build-data)')
    parser.add_argument('--output', type=str, default='models/reraise_v1.pt')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=32768)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-layers', type=str, default='256,128,64')
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--good-player-boost', type=float, default=3.0,
                        help='Weight multiplier for winning raises')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    hidden_layers = [int(x) for x in args.hidden_layers.split(',')]

    print('=' * 60)
    print('RE-RAISE CLASSIFIER TRAINING')
    print('=' * 60)
    print(f'  device: {device}')
    print(f'  layers: {hidden_layers}')
    print(f'  lr:     {args.lr}')
    print(f'  epochs: {args.epochs}')
    print('=' * 60)

    # Load data
    print(f'\nLoading data from {args.training_data}...')
    data = np.load(args.training_data)
    trans_data = {
        'states': data['states'],
        'actions': data['actions'],
        'rewards': data['rewards'],
    }
    print(f'  Total transitions: {len(trans_data["actions"])}')

    # Build dataset
    states, labels, weights = build_reraise_dataset(
        trans_data, good_player_boost=args.good_player_boost)
    obs_size = states.shape[1]
    print(f'  Final dataset: {len(states)} samples, obs_size={obs_size}')

    # Split
    n = len(states)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_n = int(n * args.val_split)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    train_ds = _ReRaiseDataset(states, labels, weights, train_idx)
    val_ds = _ReRaiseDataset(states, labels, weights, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)

    print(f'  Train: {len(train_idx)}, Val: {len(val_idx)}')

    # Model
    model = ReRaiseClassifier(obs_size, hidden_layers, args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Params: {total_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    best_epoch = 0
    patience = 30
    patience_counter = 0

    print(f'\n{"="*60}')
    print(f'  TRAINING')
    print(f'{"="*60}')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scaler)
        val = eval_epoch(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        improved = ''
        if val['f1'] > best_f1:
            best_f1 = val['f1']
            best_epoch = epoch
            patience_counter = 0
            improved = ' *best*'

            torch.save({
                'model_state_dict': model.state_dict(),
                'obs_size': obs_size,
                'hidden_layers': hidden_layers,
                'dropout': args.dropout,
                'epoch': epoch,
                'model_type': 'reraise_classifier',
                'f1': val['f1'],
            }, str(output_path))
        else:
            patience_counter += 1

        print(f'  Epoch {epoch:>3}/{args.epochs}  '
              f'loss={train_loss:.4f}  acc={train_acc:.1%}  '
              f'val_loss={val["loss"]:.4f}  val_acc={val["accuracy"]:.1%}  '
              f'P={val["precision"]:.1%} R={val["recall"]:.1%} F1={val["f1"]:.1%}  '
              f'raise%={val["raise_rate"]:.1%}  '
              f'lr={lr:.6f}  {elapsed:.1f}s{improved}')

        if patience_counter >= patience:
            print(f'\n  Early stopping at epoch {epoch}')
            break

    # Load best and show final stats
    best_ckpt = torch.load(str(output_path), map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])
    final = eval_epoch(model, val_loader, device)

    print(f'\n{"="*60}')
    print(f'  TRAINING COMPLETE')
    print(f'{"="*60}')
    print(f'  Best epoch:     {best_epoch}')
    print(f'  Best F1:        {best_f1:.1%}')
    print(f'  Precision:      {final["precision"]:.1%}')
    print(f'  Recall:         {final["recall"]:.1%}')
    print(f'  Val raise rate: {final["raise_rate"]:.1%}')
    print(f'  Model saved to: {output_path}')
    print(f'{"="*60}')


if __name__ == '__main__':
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    main()
