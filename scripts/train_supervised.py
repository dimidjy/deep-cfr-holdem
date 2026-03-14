"""
Supervised learning (behavioral cloning) from winning player hand histories.

Trains a StrategyNetwork to predict postflop actions of profitable players.
Preflop play is locked to GTO ranges (same as train_gpu.py), model only learns postflop.

Usage:
  # Train on good players (one name per line in .txt file)
  python3 scripts/train_supervised.py --good-players data/good_players.txt

  # All showdown players (no filter)
  python3 scripts/train_supervised.py --max-hands 100000

  # Include preflop decisions too (not recommended)
  python3 scripts/train_supervised.py --no-preflop-ranges --all-streets

  # Resume from a previous checkpoint
  python3 scripts/train_supervised.py --good-players data/good_players.txt --resume models/supervised_best.pt
"""
import sys
import os
import warnings
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
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
from torch.utils.tensorboard import SummaryWriter

from src.training.deep_cfr import StrategyNetwork
from src.hand_history import ObservationBuilder, DecisionPoint
from src.hand_history.parser import parse_file
from validate_spots import run_validation, validation_summary_line


# ──────────────────────────────────────────────────────────────────────────────
# DeepCFR-compatible wrapper for validation
# ──────────────────────────────────────────────────────────────────────────────

class SupervisedModelWrapper:
    """Wraps a StrategyNetwork to mimic the DeepCFR interface for run_validation."""

    def __init__(self, model, obs_size, action_size, num_players, device, iteration=0):
        self.action_size = action_size
        self.num_players = num_players
        self.device = torch.device(device)
        self.iteration = iteration

        # run_validation reads .strategy_networks[player_idx] and .advantage_networks
        self.strategy_networks = [model] * num_players
        self.advantage_networks = [model] * num_players

    def get_strategy(self, observation: np.ndarray, player_idx: int,
                     legal_actions: list) -> np.ndarray:
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.strategy_networks[player_idx](obs_tensor).cpu().numpy()[0]

        # Masked softmax: set illegal actions to -inf, then softmax
        masked = np.full(self.action_size, -1e9, dtype=np.float32)
        for a in legal_actions:
            masked[a] = logits[a]
        masked -= masked.max()
        exp_s = np.exp(masked)
        strategy = exp_s / exp_s.sum()

        # Zero residual mass on illegal actions
        legal_mask = np.zeros(self.action_size, dtype=np.float32)
        legal_mask[legal_actions] = 1.0
        strategy *= legal_mask
        s = strategy.sum()
        if s > 0:
            strategy /= s
        else:
            strategy = legal_mask / legal_mask.sum()
        return strategy


# ──────────────────────────────────────────────────────────────────────────────
# Engine factory (matches train_gpu.py)
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
# Fast file pre-filter using grep
# ──────────────────────────────────────────────────────────────────────────────

def _prefilter_files(hh_files: list, good_players: set, hh_dir: str) -> list:
    """Use grep to quickly find files containing target player names.

    With 100GB of data, parsing every file in Python is ~100x slower than
    a simple grep pre-filter. Player names appear as 'Seat N: PlayerName'
    in PokerStars HH format, so plain-text grep works perfectly.
    """
    import subprocess
    import shutil

    if not good_players:
        return hh_files

    # Escape regex special chars in player names (e.g. *Player1989*, ~player~337)
    import re
    escaped = [re.escape(name) for name in good_players]

    # Process in batches of patterns (very long regexes can fail)
    BATCH = 50
    pattern_batches = [escaped[i:i+BATCH] for i in range(0, len(escaped), BATCH)]

    matched_files = set()
    use_rg = shutil.which('rg') is not None

    t0 = time.time()
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
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f'  Warning: grep pre-filter failed ({e}), falling back to full scan')
            return hh_files

    elapsed = time.time() - t0
    tool = 'rg' if use_rg else 'grep'
    print(f'  Pre-filter ({tool}): {len(matched_files)}/{len(hh_files)} files '
          f'contain target players ({elapsed:.1f}s)')

    return [f for f in hh_files if os.path.abspath(str(f)) in matched_files]


# ──────────────────────────────────────────────────────────────────────────────
# Parallel parse worker (module-level for pickling)
# ──────────────────────────────────────────────────────────────────────────────

_PARSE_CFG = {}  # set by _parse_worker_init

def _parse_worker_init(num_players, good_players, postflop_only):
    _PARSE_CFG['num_players'] = num_players
    _PARSE_CFG['good_players'] = good_players
    _PARSE_CFG['postflop_only'] = postflop_only
    _PARSE_CFG['builder'] = ObservationBuilder(num_players=num_players)


def _parse_single_file(file_path):
    """Parse one HH file and extract decision points (worker function)."""
    num_players = _PARSE_CFG['num_players']
    good_players = _PARSE_CFG['good_players']
    postflop_only = _PARSE_CFG['postflop_only']
    builder = _PARSE_CFG['builder']

    file_dps = []
    file_total = 0
    file_used = 0
    player_counts = defaultdict(int)

    try:
        hands = parse_file(str(file_path))
    except Exception:
        return file_dps, file_total, file_used, dict(player_counts)

    for hand in hands:
        if abs(hand.big_blind_dollars - 0.05) > 0.001:
            continue
        if hand.num_active_players != num_players:
            continue

        file_total += 1

        if not hand.has_showdown:
            continue

        known_players = set(hand.shown_hands.keys())

        # Only learn from good players (or all showdown players if no list)
        if good_players:
            target_known = known_players & good_players
        else:
            target_known = known_players

        if not target_known:
            continue

        dps = builder.build_decision_points(hand, allow_unknown_cards=False)
        if not dps:
            continue

        file_used += 1
        for dp in dps:
            if postflop_only and dp.street == 'preflop':
                continue

            if dp.player_name in target_known:
                file_dps.append(dp)
                player_counts[dp.player_name] += 1

    return file_dps, file_total, file_used, dict(player_counts)


# ──────────────────────────────────────────────────────────────────────────────
# Data collection from hand histories
# ──────────────────────────────────────────────────────────────────────────────

def collect_decision_points(hh_dir: str, num_players: int,
                            max_hands: int,
                            good_players: set,
                            no_cache: bool,
                            postflop_only: bool) -> list:
    """Parse HH files and collect decision points from target players."""
    hh_path = Path(hh_dir)
    all_hh_files = sorted(hh_path.rglob('*.txt'))
    print(f'  Found {len(all_hh_files)} .txt files in {hh_dir}')

    # Fast pre-filter: grep for player names to skip irrelevant files
    hh_files = _prefilter_files(all_hh_files, good_players, hh_dir)

    if not hh_files:
        return []

    # Cache key
    player_key = '|'.join(sorted(good_players)) if good_players else 'all'
    cache_key = f'sv2|{hh_dir}|{len(all_hh_files)}|flt{len(hh_files)}|np{num_players}|max{max_hands}|g{player_key}|pf{postflop_only}'
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

    # ── Parallel file parsing ────────────────────────────────────────────
    import multiprocessing as mp

    parse_workers = min(120, max(1, mp.cpu_count() - 1))

    all_dps = []
    total_hands = 0
    hands_used = 0
    player_dps = defaultdict(int)
    t0 = time.time()

    print(f'  Parsing {len(hh_files)} files with {parse_workers} workers...')

    with mp.Pool(processes=parse_workers, initializer=_parse_worker_init,
                 initargs=(num_players, good_players, postflop_only)) as parse_pool:
        results = parse_pool.imap_unordered(_parse_single_file, hh_files, chunksize=16)
        files_done = 0
        for file_dps, file_total, file_used, file_player_counts in results:
            all_dps.extend(file_dps)
            total_hands += file_total
            hands_used += file_used
            for name, count in file_player_counts.items():
                player_dps[name] += count
            files_done += 1

            if files_done % 500 == 0:
                elapsed = time.time() - t0
                print(f'  [{files_done}/{len(hh_files)}] '
                      f'{total_hands} hands, {len(all_dps)} dps  '
                      f'({elapsed:.0f}s)')

            if max_hands > 0 and hands_used >= max_hands:
                parse_pool.terminate()
                print(f'  Reached {max_hands} hands limit')
                break

    elapsed = time.time() - t0
    print(f'  Parsed {files_done} files in {elapsed:.1f}s ({parse_workers} workers)')
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


def build_dataset(decision_points: list, action_size: int = 5,
                  weight_by_street: bool = True):
    """Convert decision points to training tensors."""
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

    return observations, actions, legal_masks, weights


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def masked_cross_entropy(logits, targets, legal_masks, weights):
    """Cross-entropy loss masked to legal actions only.

    StrategyNetwork outputs raw logits. We mask illegal actions to -inf
    so softmax (inside cross_entropy) gives them zero probability.
    """
    # Use -1e4 (not -1e9) to stay safe with float16 AMP
    masked_logits = logits + (1.0 - legal_masks) * (-1e4)
    nll = F.cross_entropy(masked_logits, targets, reduction='none')
    weighted_loss = (nll * weights).mean()
    return weighted_loss


def train_epoch(model, dataloader, optimizer, device, scaler):
    """Train one epoch, return avg loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for obs, acts, masks, wts in dataloader:
        obs = obs.to(device, non_blocking=True)
        acts = acts.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        wts = wts.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(obs)
            loss = masked_cross_entropy(logits, acts, masks, wts)

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
        masked_logits = logits + (1.0 - masks) * (-1e4)
        preds = masked_logits.argmax(dim=-1)
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

    for obs, acts, masks, wts in dataloader:
        obs = obs.to(device, non_blocking=True)
        acts = acts.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        wts = wts.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(obs)
            loss = masked_cross_entropy(logits, acts, masks, wts)

        total_loss += loss.item() * obs.size(0)

        masked_logits = logits + (1.0 - masks) * (-1e4)
        preds = masked_logits.argmax(dim=-1)
        correct += (preds == acts).sum().item()
        total += obs.size(0)

    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Supervised poker model — behavioral cloning from hand histories'
    )
    parser.add_argument('--hh-dir', type=str, default='data/poker_archive/',
                        help='Directory with PokerStars HH files')
    parser.add_argument('--good-players', type=str, default='',
                        help='Path to .txt file with winning player names (one per line)')
    parser.add_argument('--max-hands', type=int, default=0,
                        help='Max showdown hands to use (0=all)')
    parser.add_argument('--all-streets', action='store_true',
                        help='Train on all streets including preflop (default: postflop only)')
    parser.add_argument('--preflop-ranges', action='store_true', default=True,
                        help='Use GTO preflop ranges during validation (default: on)')
    parser.add_argument('--no-preflop-ranges', action='store_true',
                        help='Disable GTO preflop ranges during validation')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force re-parse HH files')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32768,
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

    # Validation
    parser.add_argument('--val-interval', type=int, default=5,
                        help='Run spot validation every N epochs (0=disabled)')
    parser.add_argument('--val-games', type=int, default=500,
                        help='Number of games for spot validation')

    # Output
    parser.add_argument('--output', type=str, default='models/supervised_model.pt',
                        help='Output model path')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='cuda | cpu (auto-detected if omitted)')

    args = parser.parse_args()

    if args.no_preflop_ranges:
        args.preflop_ranges = False

    postflop_only = not args.all_streets
    weight_streets = args.weight_streets and not args.no_weight_streets

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
        if len(good_players) <= 10:
            for name in sorted(good_players):
                print(f'  - {name}')

    num_players = 6
    action_size = 5
    hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # ── Header ───────────────────────────────────────────────────────────
    print('=' * 60)
    print('SUPERVISED LEARNING  —  BEHAVIORAL CLONING')
    print('=' * 60)
    print(f'  device   : {device}')
    if device.type == 'cuda':
        print(f'  GPU      : {torch.cuda.get_device_name(0)}')
    print(f'  engine   : fast (Rust)')
    print(f'  players  : {num_players}')
    print(f'  epochs   : {args.epochs}')
    print(f'  batch    : {args.batch_size}')
    print(f'  lr       : {args.lr}')
    print(f'  layers   : {hidden_layers}')
    print(f'  dropout  : {args.dropout}')
    print(f'  preflop  : {"GTO ranges (locked)" if args.preflop_ranges else "neural net (learned)"}')
    print(f'  training : {"postflop only" if postflop_only else "all streets"}')
    print(f'  val_int  : {args.val_interval} epochs')
    print('=' * 60)

    # ── Collect data ─────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'  COLLECTING TRAINING DATA')
    print(f'{"="*70}')

    decision_points = collect_decision_points(
        args.hh_dir, num_players, args.max_hands,
        good_players, args.no_cache,
        postflop_only,
    )

    if len(decision_points) < 100:
        print(f'Not enough data: {len(decision_points)} decision points. Need at least 100.')
        sys.exit(1)

    # ── Build dataset ────────────────────────────────────────────────────
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
                              shuffle=True, num_workers=12, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=12, pin_memory=True,
                            persistent_workers=True)

    print(f'\n  Train samples: {train_n}')
    print(f'  Val samples:   {val_n}')

    # ── Model ────────────────────────────────────────────────────────────
    print(f'\n  obs_size={obs_size}  act_size={action_size}\n')

    model = StrategyNetwork(obs_size, action_size, hidden_layers, args.dropout)
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
            else:
                print(f'  Warning: no model_state_dict in checkpoint, starting fresh')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {total_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.epochs)

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # ── Validation engine + preflop ranges ───────────────────────────────
    engine = _make_engine(num_players)

    preflop_ranges = None
    if args.preflop_ranges:
        from src.preflop_ranges import PreflopRanges
        preflop_ranges = PreflopRanges()
        print(f'  Preflop ranges loaded (GTO locked)')

    # ── TensorBoard ──────────────────────────────────────────────────────
    logs_dir = Path('data/logs') / 'tensorboard_supervised'
    logs_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logs_dir))
    print(f'TensorBoard logs -> {logs_dir}  (run: tensorboard --logdir {logs_dir})\n')

    # ── Training loop ────────────────────────────────────────────────────
    print(f'{"="*70}')
    print(f'  TRAINING  (epochs={args.epochs}, batch={args.batch_size}, lr={args.lr})')
    print(f'{"="*70}')

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    patience = 10
    patience_counter = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_epochs = start_epoch + args.epochs

    for epoch in range(start_epoch + 1, total_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, scaler)
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        # ── TensorBoard: per-epoch metrics ───────────────────────────────
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('acc/train', train_acc, epoch)
        writer.add_scalar('acc/val', val_acc, epoch)
        writer.add_scalar('lr', lr, epoch)

        # Track best
        improved = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            improved = ' *best*'

            # Save best model
            save_dict = {
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
                'postflop_only': postflop_only,
                'preflop_ranges': args.preflop_ranges,
            }
            torch.save(save_dict, str(output_path))

            # Save metadata JSON alongside checkpoint
            meta = {
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_samples': train_n,
                'val_samples': val_n,
                'obs_size': obs_size,
                'action_size': action_size,
                'hidden_layers': hidden_layers,
                'dropout': args.dropout,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'postflop_only': postflop_only,
                'preflop_ranges': args.preflop_ranges,
            }
            meta_path = str(output_path).replace('.pt', '_metadata.json')
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
        else:
            patience_counter += 1

        print(f'  Epoch {epoch:>3}/{total_epochs}  '
              f'train_loss={train_loss:.4f}  train_acc={train_acc:.1%}  '
              f'val_loss={val_loss:.4f}  val_acc={val_acc:.1%}  '
              f'lr={lr:.6f}  {elapsed:.1f}s{improved}')

        # ── Spot validation ──────────────────────────────────────────────
        if args.val_interval > 0 and epoch % args.val_interval == 0:
            wrapper = SupervisedModelWrapper(
                model, obs_size, action_size, num_players, device, iteration=epoch)
            val_results = run_validation(wrapper, engine, num_games=args.val_games,
                                        verbose=False, preflop_ranges=preflop_ranges)
            print(f'  {validation_summary_line(val_results)}')

            # TensorBoard: validation metrics
            for spot_key, spot_data in val_results.items():
                if spot_key == '_summary' or not isinstance(spot_data, dict):
                    continue
                if spot_data.get('n', 0) == 0:
                    continue
                writer.add_scalar(f'val/{spot_key}/fold', spot_data['fold_pct'], epoch)
                writer.add_scalar(f'val/{spot_key}/call', spot_data['call_pct'], epoch)
                writer.add_scalar(f'val/{spot_key}/aggr', spot_data['aggr_pct'], epoch)
            summary = val_results.get('_summary', {})
            if summary.get('total', 0) > 0:
                writer.add_scalar('val/pass_rate',
                                  summary['passed'] / summary['total'], epoch)

        # Early stopping
        if patience_counter >= patience:
            print(f'\n  Early stopping at epoch {epoch} '
                  f'(no improvement for {patience} epochs)')
            break

    writer.close()

    # ── Final validation on best model ───────────────────────────────────
    best_ckpt = torch.load(str(output_path), map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])
    wrapper = SupervisedModelWrapper(
        model, obs_size, action_size, num_players, device, iteration=best_epoch)
    final_results = run_validation(wrapper, engine, num_games=args.val_games,
                                   verbose=False, preflop_ranges=preflop_ranges)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'  TRAINING COMPLETE')
    print(f'{"="*70}')
    print(f'  Best epoch:     {best_epoch}')
    print(f'  Best val loss:  {best_val_loss:.4f}')
    print(f'  Best val acc:   {best_val_acc:.1%}')
    print(f'  Model saved to: {output_path}')
    summary = final_results.get('_summary', {})
    print(f'  Validation:     {summary.get("passed",0)}/{summary.get("total",0)} spots passed')
    print(f'  {validation_summary_line(final_results)}')
    print(f'{"="*70}')

    return final_results


if __name__ == '__main__':
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    main()
