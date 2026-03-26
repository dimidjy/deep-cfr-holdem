"""Train the Range Estimator model from hand histories.

Uses showdown hands where we know the opponent's actual cards to train
a GRU that reads action sequences and predicts opponent hand category.

Labels use eval7 hand categories (high card, pair, two pair, trips,
straight, flush, full house, quads, straight flush) instead of
equity-vs-random buckets, giving unambiguous ground truth.

For each showdown hand:
  - Pick each player as the "target opponent"
  - Encode all actions up to the current street as a sequence
  - Label = hand category of the target opponent (from known cards + board)

Usage:
  python3 scripts/train_range_estimator.py --epochs 30 --output models/range_estimator.pt
  python3 scripts/train_range_estimator.py --epochs 50 --no-cache
"""
import sys
import os
import warnings
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import time
import hashlib
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.training.range_estimator import (
    RangeEstimator, encode_action, hand_category_from_cards,
    ACTION_STEP_DIM, MAX_ACTIONS, NUM_HAND_CATEGORIES, CATEGORY_NAMES,
    _EVAL7_AVAILABLE,
)
from src.hand_history.parser import parse_file
from src.hand_history.models import ParsedHand

if not _EVAL7_AVAILABLE:
    print('FATAL: eval7 is required for hand category labels. Install with: pip install eval7')
    sys.exit(1)

# Quick sanity check
_test_cat = hand_category_from_cards(['As', 'Kd'], ['Ah', '9c', '3d'])
assert _test_cat is not None, 'eval7 hand_category_from_cards failed sanity check'


# ──────────────────────────────────────────────────────────────────────────────
# Extract training samples from a hand
# ──────────────────────────────────────────────────────────────────────────────

def extract_samples_from_hand(hand: ParsedHand, num_players: int = 6):
    """Extract (action_sequence, target_position, hand_category) from a showdown hand.

    For each street transition, creates a sample for each showdown opponent:
      - input = all actions so far as an encoded sequence
      - target_position = relative position of the opponent we're estimating
      - label = hand category of that opponent (eval7 on their cards + board)

    Preflop samples are skipped (need 5+ cards for hand category).
    """
    active = hand.active_players
    if len(active) != num_players:
        return []

    # Build position map and collect starting stacks
    btn_idx = 0
    name_to_idx = {}
    name_to_stack = {}
    for i, p in enumerate(active):
        name_to_idx[p.name] = i
        name_to_stack[p.name] = p.stack_chips
        if p.seat == hand.button_seat:
            btn_idx = i

    # Average starting stack for normalization
    avg_stack = max(1, sum(name_to_stack.values()) / len(name_to_stack))

    # Community cards at each street
    cc = hand.community_cards_at_street

    # Track state
    current_street = 'preflop'
    current_community = []  # board cards visible on current street
    pot_chips = 0
    player_bets = defaultdict(int)  # current street bets
    player_spent = defaultdict(int)  # total spent across all streets
    encoded_actions = []

    samples = []
    opponent_street_samples = defaultdict(set)

    for action in hand.actions:
        if action.player_name not in name_to_idx:
            continue

        pidx = name_to_idx[action.player_name]
        rel_pos = (pidx - btn_idx) % num_players

        # Handle street transition
        if action.street != current_street:
            # Create samples at end of completed street
            # Skip preflop — need 5+ cards for hand category
            if current_street != 'preflop':
                community = cc.get(current_street, [])
                if len(community) >= 3:  # flop+ has enough cards
                    for opp_name, opp_cards in hand.shown_hands.items():
                        if opp_name not in name_to_idx:
                            continue
                        if current_street in opponent_street_samples[opp_name]:
                            continue
                        if not encoded_actions:
                            continue

                        category = hand_category_from_cards(opp_cards, community)
                        if category is None:
                            continue

                        opp_idx = name_to_idx[opp_name]
                        opp_rel_pos = (opp_idx - btn_idx) % num_players

                        seq = list(encoded_actions)
                        samples.append((seq, opp_rel_pos, category, len(seq)))
                        opponent_street_samples[opp_name].add(current_street)

            # Transition
            pot_chips += sum(player_bets.values())
            player_bets.clear()
            current_street = action.street
            current_community = cc.get(current_street, [])

        # Skip blinds for action encoding (forced, not informative)
        if action.is_blind:
            player_bets[action.player_name] += action.amount_chips
            player_spent[action.player_name] += action.amount_chips
            continue

        # Compute sizing as fraction of pot
        total_pot = pot_chips + sum(player_bets.values())
        if action.action_type in ('bet', 'raise') and total_pot > 0:
            sizing = action.amount_chips / max(total_pot, 1)
        else:
            sizing = 0.0

        # Pot ratio: current pot / average starting stack
        pot_ratio = total_pot / avg_stack

        # Stack ratio: actor's remaining stack / their starting stack
        actor_start = name_to_stack.get(action.player_name, avg_stack)
        actor_spent = player_spent.get(action.player_name, 0)
        actor_remaining = max(0, actor_start - actor_spent)
        stack_ratio = actor_remaining / max(1, actor_start)

        # Map action type
        if action.is_all_in:
            atype = 'all_in'
        elif action.action_type in ('bet', 'raise', 'call', 'check', 'fold'):
            atype = action.action_type
        else:
            atype = 'check'

        encoded = encode_action(
            action_type=atype,
            rel_position=rel_pos,
            sizing=sizing,
            street=current_street,
            pot_ratio=pot_ratio,
            stack_ratio=stack_ratio,
            num_players=num_players,
            community_cards=current_community,
        )
        encoded_actions.append(encoded)

        # Update bets and spending
        if action.action_type in ('bet', 'call'):
            player_bets[action.player_name] += action.amount_chips
            player_spent[action.player_name] += action.amount_chips
        elif action.action_type == 'raise':
            already = player_bets.get(action.player_name, 0)
            delta = action.amount_chips - already
            if delta > 0:
                player_spent[action.player_name] += delta
                player_bets[action.player_name] = action.amount_chips

    # Final street samples (skip if preflop-only)
    if current_street != 'preflop':
        community = cc.get(current_street, [])
        if len(community) >= 3:
            for opp_name, opp_cards in hand.shown_hands.items():
                if opp_name not in name_to_idx:
                    continue
                if current_street in opponent_street_samples[opp_name]:
                    continue
                if not encoded_actions:
                    continue

                category = hand_category_from_cards(opp_cards, community)
                if category is None:
                    continue

                opp_idx = name_to_idx[opp_name]
                opp_rel_pos = (opp_idx - btn_idx) % num_players

                seq = list(encoded_actions)
                samples.append((seq, opp_rel_pos, category, len(seq)))
                opponent_street_samples[opp_name].add(current_street)

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Parallel worker
# ──────────────────────────────────────────────────────────────────────────────

_WORKER_CFG = {}


def _worker_init(num_players):
    _WORKER_CFG['num_players'] = num_players


def _parse_single_file(file_path):
    """Parse one HH file and extract range estimation samples."""
    num_players = _WORKER_CFG['num_players']

    all_samples = []
    total_hands = 0
    hands_used = 0

    try:
        hands = parse_file(str(file_path))
    except Exception:
        return all_samples, total_hands, hands_used

    for hand in hands:
        if abs(hand.big_blind_dollars - 0.05) > 0.001:
            continue
        if hand.num_active_players != num_players:
            continue
        total_hands += 1

        if not hand.has_showdown:
            continue

        samples = extract_samples_from_hand(hand, num_players)
        if samples:
            all_samples.extend(samples)
            hands_used += 1

    return all_samples, total_hands, hands_used


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class RangeDataset(Dataset):
    """Index-based dataset — references shared numpy arrays, no copies."""

    def __init__(self, indices, sequences, lengths, positions, targets):
        self.indices = indices          # np array of indices into the shared arrays
        self.sequences = sequences      # shared (N_total, MAX_ACTIONS, ACTION_STEP_DIM) numpy
        self.lengths = lengths          # shared (N_total,) numpy
        self.positions = positions      # shared (N_total,) numpy
        self.targets = targets          # shared (N_total,) numpy

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (torch.from_numpy(self.sequences[i].copy()),
                self.lengths[i],
                self.positions[i],
                self.targets[i])


# ──────────────────────────────────────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────────────────────────────────────

def collect_samples(hh_dir, num_players, max_hands, no_cache, workers):
    """Parse HH files and collect range estimation samples."""
    hh_path = Path(hh_dir)
    all_files = sorted(hh_path.rglob('*.txt'))
    print(f'  Found {len(all_files)} .txt files in {hh_dir}')

    if not all_files:
        return [], [], [], []

    # Cache (v4 = merged 7 categories + board texture encoding)
    cache_key = f'range_v4|{hh_dir}|{len(all_files)}|np{num_players}|max{max_hands}'
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    cache_dir = Path(__file__).parent.parent / 'data' / '.hh_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'range_{cache_hash}.pkl'

    if cache_file.exists() and not no_cache:
        print(f'  Loading cached data from {cache_file.name}...')
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f'  Loaded {len(data["targets"])} samples from {data["hands_used"]} hands')
        return data['sequences'], data['lengths'], data['positions'], data['targets']

    # Parallel parsing
    import multiprocessing as mp
    parse_workers = workers if workers > 0 else min(120, max(1, mp.cpu_count() - 1))

    all_samples = []
    total_hands = 0
    hands_used = 0
    t0 = time.time()

    print(f'  Parsing {len(all_files)} files with {parse_workers} workers...')

    with mp.Pool(processes=parse_workers, initializer=_worker_init,
                 initargs=(num_players,)) as pool:
        results = pool.imap_unordered(_parse_single_file, all_files, chunksize=16)
        files_done = 0
        for file_samples, file_total, file_used in results:
            all_samples.extend(file_samples)
            total_hands += file_total
            hands_used += file_used
            files_done += 1

            if files_done % 500 == 0:
                elapsed = time.time() - t0
                print(f'  [{files_done}/{len(all_files)}] '
                      f'{total_hands} hands, {len(all_samples)} samples  '
                      f'({elapsed:.0f}s)')

            if max_hands > 0 and hands_used >= max_hands:
                pool.terminate()
                print(f'  Reached {max_hands} hands limit')
                break

    elapsed = time.time() - t0
    print(f'  Parsed {files_done} files in {elapsed:.1f}s')
    print(f'  Total hands: {total_hands}, showdown hands used: {hands_used}')
    print(f'  Total samples: {len(all_samples)}')

    if not all_samples:
        return [], [], [], []

    # Build padded arrays in chunks to avoid double memory peak
    n = len(all_samples)
    print(f'  Building padded arrays ({n} samples, {n * MAX_ACTIONS * ACTION_STEP_DIM * 4 / 1e9:.1f} GB)...')

    # Allocate target arrays first (small)
    lengths = np.zeros(n, dtype=np.int64)
    positions = np.zeros(n, dtype=np.int64)
    targets = np.zeros(n, dtype=np.int64)

    # Fill small arrays and free sequence data from tuples incrementally
    # First pass: extract metadata only
    for i, (seq, opp_pos, category, seq_len) in enumerate(all_samples):
        lengths[i] = min(seq_len, MAX_ACTIONS)
        positions[i] = opp_pos
        targets[i] = category

    # Allocate the big sequence array
    sequences = np.zeros((n, MAX_ACTIONS, ACTION_STEP_DIM), dtype=np.float32)

    # Second pass: fill sequences in chunks and free raw data
    CHUNK = 500_000
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        for i in range(start, end):
            seq, _, _, seq_len = all_samples[i]
            actual_len = min(seq_len, MAX_ACTIONS)
            for j in range(actual_len):
                sequences[i, j] = seq[j]
            all_samples[i] = None  # free raw Python lists immediately
        if start % (CHUNK * 4) == 0 and start > 0:
            print(f'    [{start}/{n}] converted...')

    del all_samples
    import gc; gc.collect()
    print(f'  Padded arrays built.')

    # Cache
    data = {'sequences': sequences, 'lengths': lengths, 'positions': positions,
            'targets': targets, 'hands_used': hands_used}
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        mb = cache_file.stat().st_size / 1e6
        print(f'  Cached to {cache_file.name} ({mb:.0f} MB)')
    except Exception as e:
        print(f'  Cache save failed: {e}')

    return sequences, lengths, positions, targets


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, device, scaler, class_weights=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for seqs, lens, pos, tgts in dataloader:
        seqs = seqs.to(device, non_blocking=True)
        lens = lens.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(seqs, lens, pos)
            loss = F.cross_entropy(logits, tgts, weight=class_weights)

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

        total_loss += loss.item() * seqs.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == tgts).sum().item()
        total += seqs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, dataloader, device, class_weights=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    cat_correct = np.zeros(NUM_HAND_CATEGORIES)
    cat_total = np.zeros(NUM_HAND_CATEGORIES)

    for seqs, lens, pos, tgts in dataloader:
        seqs = seqs.to(device, non_blocking=True)
        lens = lens.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(seqs, lens, pos)
            loss = F.cross_entropy(logits, tgts, weight=class_weights)

        total_loss += loss.item() * seqs.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == tgts).sum().item()
        total += seqs.size(0)

        for b in range(NUM_HAND_CATEGORIES):
            mask = tgts == b
            cat_total[b] += mask.sum().item()
            cat_correct[b] += ((preds == tgts) & mask).sum().item()

    cat_acc = {}
    for b in range(NUM_HAND_CATEGORIES):
        cat_acc[b] = cat_correct[b] / cat_total[b] if cat_total[b] > 0 else 0.0

    return total_loss / total, correct / total, cat_acc


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train Range Estimator -- predict opponent hand category from actions')
    parser.add_argument('--hh-dir', type=str, default='data/poker_archive/',
                        help='Directory with PokerStars HH files')
    parser.add_argument('--num-players', type=int, default=6)
    parser.add_argument('--max-hands', type=int, default=0,
                        help='Max showdown hands (0=all)')
    parser.add_argument('--no-cache', action='store_true')

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='GRU hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='GRU layers')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val-split', type=float, default=0.1)

    parser.add_argument('--output', type=str, default='models/range_estimator.pt')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--workers', type=int, default=0)

    args = parser.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    print('=' * 60)
    print('RANGE ESTIMATOR TRAINING (v2 — hand categories)')
    print('=' * 60)
    print(f'  device      : {device}')
    if device.type == 'cuda':
        print(f'  GPU         : {torch.cuda.get_device_name(0)}')
    print(f'  players     : {args.num_players}')
    print(f'  epochs      : {args.epochs}')
    print(f'  batch       : {args.batch_size}')
    print(f'  lr          : {args.lr}')
    print(f'  hidden_dim  : {args.hidden_dim}')
    print(f'  gru_layers  : {args.num_layers}')
    print(f'  dropout     : {args.dropout}')
    print(f'  input_dim   : {ACTION_STEP_DIM} per action step')
    print(f'  max_actions : {MAX_ACTIONS}')
    print(f'  categories  : {NUM_HAND_CATEGORIES} ({", ".join(CATEGORY_NAMES)})')
    print('=' * 60)

    # ── Collect data ─────────────────────────────────────────────────────
    print(f'\nCollecting training data...')
    sequences, lengths, positions, targets = collect_samples(
        args.hh_dir, args.num_players, args.max_hands,
        args.no_cache, args.workers)

    if len(targets) < 100:
        print(f'Not enough data: {len(targets)} samples')
        sys.exit(1)

    n = len(targets)
    print(f'\n  Total samples: {n}')

    # Category distribution
    print(f'  Category distribution:')
    for b in range(NUM_HAND_CATEGORIES):
        c = int((targets == b).sum())
        pct = c / n * 100 if n > 0 else 0
        print(f'    {CATEGORY_NAMES[b]:<12} {c:>8}  ({pct:.1f}%)')

    # Train/val split (random)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_n = int(n * args.val_split)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    train_dataset = RangeDataset(train_idx, sequences, lengths, positions, targets)
    val_dataset = RangeDataset(val_idx, sequences, lengths, positions, targets)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True,
                            persistent_workers=True)

    print(f'  Train: {len(train_idx)}, Val: {val_n}')

    # ── Class weights (sqrt inverse frequency) ─────────────────────────
    train_targets = targets[train_idx]
    class_counts = np.bincount(train_targets.astype(int), minlength=NUM_HAND_CATEGORIES).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    # sqrt(inverse frequency) — balances classes without extreme ratios
    raw_weights = 1.0 / np.sqrt(class_counts)
    raw_weights /= raw_weights.mean()  # normalize so mean weight = 1
    class_weights = torch.FloatTensor(raw_weights).to(device)
    print(f'  Class weights: {", ".join(f"{CATEGORY_NAMES[i][:4]}={raw_weights[i]:.2f}" for i in range(NUM_HAND_CATEGORIES))}')

    # ── Model ────────────────────────────────────────────────────────────
    model = RangeEstimator(
        input_dim=ACTION_STEP_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=NUM_HAND_CATEGORIES,
        dropout=args.dropout,
        num_positions=args.num_players,
    )
    model = model.to(device)

    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            ckpt = torch.load(str(resume_path), map_location=device)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                start_epoch = ckpt.get('epoch', 0)
                print(f'  Resumed from {resume_path.name} (epoch {start_epoch})')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {total_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # ── Training loop ────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'TRAINING')
    print(f'{"="*60}')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    patience = 15
    patience_counter = 0

    total_epochs = start_epoch + args.epochs

    for epoch in range(start_epoch + 1, total_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, scaler, class_weights)
        val_loss, val_acc, cat_acc = eval_epoch(model, val_loader, device, class_weights)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        improved = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            improved = ' *best*'

            save_dict = {
                'model_state_dict': model.state_dict(),
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'input_dim': ACTION_STEP_DIM,
                'output_dim': NUM_HAND_CATEGORIES,
                'dropout': args.dropout,
                'num_players': args.num_players,
                'num_positions': args.num_players,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }
            torch.save(save_dict, str(output_path))
        else:
            patience_counter += 1

        # Per-category accuracy (show top categories only to fit line)
        cat_str = '  '.join(
            f'{CATEGORY_NAMES[b][:4]}={cat_acc[b]:.0%}'
            for b in range(NUM_HAND_CATEGORIES))

        print(f'  Epoch {epoch:>3}/{total_epochs}  '
              f'train={train_loss:.4f}/{train_acc:.1%}  '
              f'val={val_loss:.4f}/{val_acc:.1%}  '
              f'lr={lr:.6f}  {elapsed:.1f}s{improved}')
        print(f'    categories: {cat_str}')

        if patience_counter >= patience:
            print(f'\n  Early stopping at epoch {epoch}')
            break

    # ── Summary ──────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'TRAINING COMPLETE')
    print(f'{"="*60}')
    print(f'  Best epoch:    {best_epoch}')
    print(f'  Best val loss: {best_val_loss:.4f}')
    print(f'  Best val acc:  {best_val_acc:.1%}')
    print(f'  Model saved:   {output_path}')
    print(f'{"="*60}')


if __name__ == '__main__':
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    main()
