"""
Supervised learning (behavioral cloning) from winning player hand histories.

Trains a StrategyNetwork to predict actions of profitable players.
Can filter by player names (good players to learn from, bad players to avoid).

Usage:
  # Train on good players only (one name per line in .txt file)
  python3 scripts/train_supervised.py --good-players data/good_players.txt

  # Train on all showdown players except bad ones
  python3 scripts/train_supervised.py --bad-players data/bad_players.txt

  # Both lists
  python3 scripts/train_supervised.py --good-players data/good_players.txt --bad-players data/bad_players.txt

  # All data, no filter
  python3 scripts/train_supervised.py --max-hands 100000

  # Resume from a previous checkpoint
  python3 scripts/train_supervised.py --good-players data/good_players.txt --resume models/supervised_best.pt
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import hashlib
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.training.deep_cfr import StrategyNetwork
from src.hand_history import ObservationBuilder, DecisionPoint
from src.hand_history.parser import parse_file


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PokerActionDataset(Dataset):
    """Dataset of (observation, action, legal_mask, weight) tuples."""

    def __init__(self, observations, actions, legal_masks, weights):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.LongTensor(actions)
        self.legal_masks = torch.FloatTensor(legal_masks)
        self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (self.observations[idx], self.actions[idx],
                self.legal_masks[idx], self.weights[idx])


# ──────────────────────────────────────────────────────────────────────────────
# Data collection from hand histories
# ──────────────────────────────────────────────────────────────────────────────

def collect_decision_points(hh_dir: str, num_players: int,
                            max_hands: int,
                            good_players: set, bad_players: set,
                            no_cache: bool,
                            postflop_only: bool) -> list:
    """Parse HH files and collect decision points from target players.

    Filtering logic:
    - If good_players is set: only include decisions from those players
    - If bad_players is set: exclude decisions from those players
    - If neither: include all showdown players
    """
    hh_path = Path(hh_dir)
    hh_files = sorted(hh_path.rglob('*.txt'))
    print(f'  Found {len(hh_files)} .txt files in {hh_dir}')

    if not hh_files:
        return []

    # Cache key
    player_key = '|'.join(sorted(good_players)) if good_players else 'all'
    bad_key = '|'.join(sorted(bad_players)) if bad_players else 'none'
    cache_key = f'sv|{hh_dir}|{len(hh_files)}|np{num_players}|max{max_hands}|g{player_key}|b{bad_key}|pf{postflop_only}'
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    cache_dir = Path(__file__).parent.parent / 'data' / '.hh_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'supervised_{cache_hash}.pkl'

    if cache_file.exists() and not no_cache:
        print(f'  Loading cached data from {cache_file.name}...')
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f'  Loaded {len(data["decision_points"])} decision points '
              f'from {data["hands_used"]} hands')
        return data['decision_points']

    builder = ObservationBuilder(num_players=num_players)
    all_dps = []
    total_hands = 0
    hands_used = 0
    files_parsed = 0
    t0 = time.time()

    # Track player stats for reporting
    player_hands = defaultdict(int)
    player_dps = defaultdict(int)

    for fi, f in enumerate(hh_files):
        try:
            hands = parse_file(str(f))
        except Exception:
            continue

        files_parsed += 1

        for hand in hands:
            # Filter: NL5, 6-player, showdown
            if abs(hand.big_blind_dollars - 0.05) > 0.001:
                continue
            if hand.num_active_players != num_players:
                continue
            if not hand.has_showdown:
                continue

            total_hands += 1

            # Determine which players to learn from in this hand
            # For showdown players (cards known): learn all actions
            # For non-showdown players: learn fold actions (with neutral card features)
            known_players = set(hand.shown_hands.keys())

            if good_players:
                target_known = known_players & good_players
            elif bad_players:
                target_known = known_players - bad_players
            else:
                target_known = known_players

            # All active player names (for fold learning)
            all_active = set(p.name for p in hand.active_players)
            if bad_players:
                all_active -= bad_players

            # Build decision points (allow_unknown_cards=True to get fold examples)
            dps = builder.build_decision_points(hand, allow_unknown_cards=True)
            if not dps:
                continue

            hands_used += 1
            for dp in dps:
                if postflop_only and dp.street == 'preflop':
                    continue

                has_cards = dp.hole_cards is not None
                is_target = dp.player_name in target_known

                # Include if: known target player (all actions),
                # OR unknown player with fold action (fold examples)
                if is_target:
                    all_dps.append(dp)
                    player_hands[dp.player_name] += 1
                    player_dps[dp.player_name] += 1
                elif not has_cards and dp.actual_action == 0 and dp.player_name in all_active:
                    # Fold from non-showdown player — valuable training signal
                    all_dps.append(dp)
                    player_dps[dp.player_name] += 1

            # Early stop
            if max_hands > 0 and hands_used >= max_hands:
                print(f'  Reached {max_hands} hands limit')
                break

        # Progress
        if files_parsed % 500 == 0:
            elapsed = time.time() - t0
            print(f'  [{files_parsed}/{len(hh_files)}] '
                  f'{total_hands} hands, {len(all_dps)} dps  '
                  f'({elapsed:.0f}s)')

        if max_hands > 0 and hands_used >= max_hands:
            break

    elapsed = time.time() - t0
    print(f'  Parsed {files_parsed} files in {elapsed:.1f}s')
    print(f'  Total showdown hands: {total_hands}')
    print(f'  Hands with target players: {hands_used}')
    print(f'  Decision points collected: {len(all_dps)}')

    # Show top players
    if player_dps:
        top = sorted(player_dps.items(), key=lambda x: -x[1])[:20]
        print(f'\n  Top contributing players:')
        for name, count in top:
            print(f'    {name:<25} {count:>6} decisions')

    # Cache
    data = {'decision_points': all_dps, 'hands_used': hands_used}
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        mb = cache_file.stat().st_size / 1e6
        print(f'  Cached to {cache_file.name} ({mb:.0f} MB)')
    except Exception as e:
        print(f'  Cache save failed: {e}')

    return all_dps


def balance_decision_points(decision_points: list, max_fold_ratio: float = 2.0):
    """Balance fold vs non-fold samples to prevent fold-dominated datasets.

    Args:
        max_fold_ratio: Max ratio of folds to non-folds. E.g. 2.0 means
                        at most 2 folds per non-fold sample.
    """
    folds = [dp for dp in decision_points if dp.actual_action == 0]
    non_folds = [dp for dp in decision_points if dp.actual_action != 0]

    if not non_folds:
        return decision_points

    max_folds = int(len(non_folds) * max_fold_ratio)
    if len(folds) <= max_folds:
        return decision_points

    # Prioritize folds with known cards, then sample from unknown-card folds
    known_folds = [dp for dp in folds if dp.hole_cards is not None]
    unknown_folds = [dp for dp in folds if dp.hole_cards is None]

    if len(known_folds) >= max_folds:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(known_folds), max_folds, replace=False)
        sampled_folds = [known_folds[i] for i in indices]
    else:
        remaining = max_folds - len(known_folds)
        rng = np.random.RandomState(42)
        if len(unknown_folds) > remaining:
            indices = rng.choice(len(unknown_folds), remaining, replace=False)
            sampled_unknown = [unknown_folds[i] for i in indices]
        else:
            sampled_unknown = unknown_folds
        sampled_folds = known_folds + sampled_unknown

    print(f'  Balanced folds: {len(folds)} -> {len(sampled_folds)} '
          f'(non-folds: {len(non_folds)}, ratio: {len(sampled_folds)/max(len(non_folds),1):.1f}x)')

    result = sampled_folds + non_folds
    np.random.RandomState(42).shuffle(result)
    return result


def build_dataset(decision_points: list, action_size: int = 5,
                  weight_by_street: bool = True):
    """Convert decision points to training tensors.

    Args:
        weight_by_street: Weight postflop decisions higher (they're rarer
                          but more important for learning).
    """
    n = len(decision_points)
    obs_size = decision_points[0].observation.shape[0]

    observations = np.zeros((n, obs_size), dtype=np.float32)
    actions = np.zeros(n, dtype=np.int64)
    legal_masks = np.zeros((n, action_size), dtype=np.float32)
    weights = np.ones(n, dtype=np.float32)

    # Street weights: postflop decisions are more valuable
    street_weights = {
        'preflop': 1.0,
        'flop': 2.0,
        'turn': 3.0,
        'river': 4.0,
    }

    for i, dp in enumerate(decision_points):
        observations[i] = dp.observation
        actions[i] = dp.actual_action
        for a in dp.legal_actions:
            legal_masks[i, a] = 1.0
        if weight_by_street:
            weights[i] = street_weights.get(dp.street, 1.0)
        # Lower weight for unknown-card fold samples (less reliable features)
        if dp.hole_cards is None:
            weights[i] *= 0.5

    return observations, actions, legal_masks, weights


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def masked_cross_entropy(logits, targets, legal_masks, weights):
    """Cross-entropy loss masked to legal actions only.

    The network outputs softmax probs, so we convert back to log-probs,
    mask illegal actions, and compute NLL.
    """
    # logits are already softmax probs from StrategyNetwork
    # Mask illegal actions to tiny prob before log
    masked_probs = logits * legal_masks + 1e-8
    masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
    log_probs = torch.log(masked_probs)

    # NLL loss per sample
    nll = F.nll_loss(log_probs, targets, reduction='none')

    # Apply weights
    weighted_loss = (nll * weights).mean()
    return weighted_loss


def train_epoch(model, dataloader, optimizer, device, scaler):
    """Train one epoch, return avg loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for obs, acts, masks, wts in dataloader:
        obs = obs.to(device)
        acts = acts.to(device)
        masks = masks.to(device)
        wts = wts.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            probs = model(obs)
            loss = masked_cross_entropy(probs, acts, masks, wts)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * obs.size(0)

        # Accuracy (top-1 among legal actions)
        masked_probs = probs * masks
        preds = masked_probs.argmax(dim=-1)
        correct += (preds == acts).sum().item()
        total += obs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, dataloader, device):
    """Evaluate on validation set, return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Per-street accuracy
    street_correct = defaultdict(int)
    street_total = defaultdict(int)

    for obs, acts, masks, wts in dataloader:
        obs = obs.to(device)
        acts = acts.to(device)
        masks = masks.to(device)
        wts = wts.to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            probs = model(obs)
            loss = masked_cross_entropy(probs, acts, masks, wts)

        total_loss += loss.item() * obs.size(0)

        masked_probs = probs * masks
        preds = masked_probs.argmax(dim=-1)
        correct += (preds == acts).sum().item()
        total += obs.size(0)

    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train supervised poker model from hand histories'
    )
    parser.add_argument('--hh-dir', type=str, default='data/poker_archive/',
                        help='Directory with PokerStars HH files')
    parser.add_argument('--good-players', type=str, default='',
                        help='Path to .txt file with winning player names (one per line)')
    parser.add_argument('--bad-players', type=str, default='',
                        help='Path to .txt file with losing player names (one per line)')
    parser.add_argument('--max-hands', type=int, default=0,
                        help='Max showdown hands to use (0=all)')
    parser.add_argument('--postflop-only', action='store_true',
                        help='Only train on postflop decisions (use GTO preflop)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force re-parse HH files')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden-layers', type=str, default='512,512,256',
                        help='Hidden layer sizes (comma-separated)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--weight-streets', action='store_true', default=True,
                        help='Weight postflop decisions higher')
    parser.add_argument('--no-weight-streets', action='store_true',
                        help='Equal weight for all streets')

    # Output
    parser.add_argument('--output', type=str, default='models/supervised_model.pt',
                        help='Output model path')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint')

    args = parser.parse_args()

    weight_streets = args.weight_streets and not args.no_weight_streets

    # Load player lists from files
    good_players = set()
    bad_players = set()
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
        if len(good_players) <= 10:
            for name in sorted(good_players):
                print(f'  - {name}')
    if args.bad_players:
        p = Path(args.bad_players)
        if not p.exists():
            print(f'Bad players file not found: {p}')
            sys.exit(1)
        bad_players = set(
            line.strip() for line in p.read_text().splitlines()
            if line.strip() and not line.strip().startswith('#')
        )
        print(f'Bad players ({len(bad_players)}): loaded from {p.name}')
        if len(bad_players) <= 10:
            for name in sorted(bad_players):
                print(f'  - {name}')

    num_players = 6
    action_size = 5
    hidden_layers = [int(x) for x in args.hidden_layers.split(',')]

    # ── Collect data ─────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'  COLLECTING TRAINING DATA')
    print(f'{"="*70}')

    decision_points = collect_decision_points(
        args.hh_dir, num_players, args.max_hands,
        good_players, bad_players, args.no_cache,
        args.postflop_only,
    )

    if len(decision_points) < 100:
        print(f'Not enough data: {len(decision_points)} decision points. Need at least 100.')
        sys.exit(1)

    # ── Balance and build dataset ────────────────────────────────────────
    print(f'\nBalancing dataset...')
    decision_points = balance_decision_points(decision_points, max_fold_ratio=2.0)

    print(f'Building dataset...')
    obs_size = decision_points[0].observation.shape[0]
    observations, actions, legal_masks, weights = build_dataset(
        decision_points, action_size, weight_streets)

    # Street distribution
    street_counts = defaultdict(int)
    action_counts = defaultdict(int)
    for dp in decision_points:
        street_counts[dp.street] += 1
        action_counts[dp.actual_action] += 1

    print(f'  Observation size: {obs_size}')
    print(f'  Total samples: {len(observations)}')
    print(f'  Street distribution:')
    for s in ['preflop', 'flop', 'turn', 'river']:
        c = street_counts.get(s, 0)
        print(f'    {s:<10} {c:>8}  ({c/len(observations)*100:.1f}%)')
    print(f'  Action distribution:')
    action_names = {0: 'fold', 1: 'call', 2: 'raise_half', 3: 'raise_pot', 4: 'all_in'}
    for a in range(action_size):
        c = action_counts.get(a, 0)
        print(f'    {action_names[a]:<12} {c:>8}  ({c/len(observations)*100:.1f}%)')

    # ── Train/val split ──────────────────────────────────────────────────
    n = len(observations)
    indices = np.random.permutation(n)
    val_n = int(n * args.val_split)
    train_n = n - val_n

    train_idx = indices[:train_n]
    val_idx = indices[train_n:]

    train_dataset = PokerActionDataset(
        observations[train_idx], actions[train_idx],
        legal_masks[train_idx], weights[train_idx])
    val_dataset = PokerActionDataset(
        observations[val_idx], actions[val_idx],
        legal_masks[val_idx], weights[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    print(f'\n  Train samples: {train_n}')
    print(f'  Val samples:   {val_n}')

    # ── Model ────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device: {device}')

    model = StrategyNetwork(obs_size, action_size, hidden_layers, args.dropout)
    model = model.to(device)

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            ckpt = torch.load(str(resume_path), map_location=device)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                print(f'  Resumed from {resume_path.name}')
            else:
                print(f'  Warning: no model_state_dict in checkpoint, starting fresh')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {total_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # ── Training loop ────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'  TRAINING  (epochs={args.epochs}, batch={args.batch_size}, lr={args.lr})')
    print(f'{"="*70}')

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    patience = 10
    patience_counter = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, scaler)
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        # Track best
        improved = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            improved = ' *best*'

            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'obs_size': obs_size,
                'action_size': action_size,
                'hidden_layers': hidden_layers,
                'dropout': args.dropout,
                'num_players': num_players,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_samples': train_n,
                'good_players': list(good_players),
                'bad_players': list(bad_players),
                'postflop_only': args.postflop_only,
            }, str(output_path))
        else:
            patience_counter += 1

        print(f'  Epoch {epoch:>3}/{args.epochs}  '
              f'train_loss={train_loss:.4f}  train_acc={train_acc:.1%}  '
              f'val_loss={val_loss:.4f}  val_acc={val_acc:.1%}  '
              f'lr={lr:.6f}  {elapsed:.1f}s{improved}')

        # Early stopping
        if patience_counter >= patience:
            print(f'\n  Early stopping at epoch {epoch} '
                  f'(no improvement for {patience} epochs)')
            break

    # ── Summary ──────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'  TRAINING COMPLETE')
    print(f'{"="*70}')
    print(f'  Best epoch:     {best_epoch}')
    print(f'  Best val loss:  {best_val_loss:.4f}')
    print(f'  Best val acc:   {best_val_acc:.1%}')
    print(f'  Model saved to: {output_path}')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
