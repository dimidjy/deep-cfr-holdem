"""
Supervised learning (behavioral cloning) from winning player hand histories.

Trains a StrategyNetwork to predict postflop actions of profitable players.
Preflop play is locked to GTO ranges, model only learns postflop.

Features:
  - Counterfactual folds: synthetic fold labels from losers' last postflop actions
  - Reward weighting: winners boosted, losers' non-fold actions reduced
  - Label smoothing with action cost matrix
  - Pot-size weighting (bigger pots matter more)
  - Cosine annealing LR scheduler
  - File-based train/val split (no session leakage)

Usage:
  # Train on all showdown players
  python3 scripts/train_supervised.py --epochs 50 --output models/supervised_v3.pt --no-cache

  # Train on good players only
  python3 scripts/train_supervised.py --good-players data/good_players.txt --output models/supervised_v3.pt --no-cache

  # Resume from a previous checkpoint
  python3 scripts/train_supervised.py --resume models/supervised_v3.pt --output models/supervised_v3.pt
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

        dps = builder.build_decision_points(hand, allow_unknown_cards=False,
                                               source_file=str(file_path))
        if not dps:
            continue

        # Compute player net profit from hand result
        # net = winnings - total_invested (stack_before - stack_after is equivalent)
        player_winnings = {}
        for w in hand.result.winners:
            name = w['name']
            player_winnings[name] = player_winnings.get(name, 0) + w['amount_dollars']

        player_invested = {}
        for action in hand.actions:
            if action.is_blind or action.action_type in ('bet', 'raise', 'call'):
                name = action.player_name
                player_invested[name] = player_invested.get(name, 0) + action.amount_dollars

        player_net = {}
        for p in hand.active_players:
            won = player_winnings.get(p.name, 0.0)
            invested = player_invested.get(p.name, 0.0)
            player_net[p.name] = won - invested

        # Set net profit on each dp
        for dp in dps:
            dp.player_net_dollars = player_net.get(dp.player_name, 0.0)

        # Mark last action per player
        last_dp_per_player = {}
        for dp in dps:
            last_dp_per_player[dp.player_name] = dp
        for dp in last_dp_per_player.values():
            dp.is_last_action = True

        file_used += 1
        for dp in dps:
            if postflop_only and dp.street == 'preflop':
                continue

            if dp.player_name in target_known:
                # Good player's own actions (call, raise, fold, etc.)
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


def apply_counterfactual_folds(decision_points: list) -> list:
    """Generate synthetic fold labels from losing players' last actions.

    For each showdown loser whose last postflop action was call/raise,
    create a copy of that decision point with action=fold.
    This teaches the model "you should have folded here" using real
    observations with known outcomes.

    Only applies to postflop actions (preflop is handled by GTO ranges).
    Only creates folds when the player actually lost money.
    """
    FOLD = 0
    synthetic_folds = []
    # Equity feature is at index 18 in the 41-dim observation
    # (Position8 + PotOdds3 + HandStrength5 + Stack2 = 18)
    EQUITY_IDX = 18

    for dp in decision_points:
        # Only relabel last actions of losing players
        if not dp.is_last_action:
            continue
        if dp.player_net_dollars >= 0:
            continue  # winner or break-even — don't relabel
        if dp.actual_action == FOLD:
            continue  # already a fold
        if dp.street == 'preflop':
            continue  # preflop handled by GTO ranges
        if FOLD not in dp.legal_actions:
            continue  # fold wasn't legal

        # Only create fold labels when equity was below ~70%
        # This catches: weak/mid pairs, busted draws, ace-high
        # Skips: overpair+, two pair+, strong made hands that got unlucky
        # Data shows: p25=0.73, median=0.85 for losing showdown last-actions
        equity = dp.observation[EQUITY_IDX] if len(dp.observation) > EQUITY_IDX else 0.5
        if equity > 0.7:
            continue

        # Create synthetic fold — copy of same observation but action=fold
        import copy
        fold_dp = copy.copy(dp)
        fold_dp.actual_action = FOLD
        synthetic_folds.append(fold_dp)

    print(f'  Counterfactual folds: {len(synthetic_folds)} synthetic fold labels '
          f'from losing players\' last actions')

    return decision_points + synthetic_folds


def apply_reward_weighting(decision_points: list) -> list:
    """Set sample weights based on hand outcomes.

    - Winners' actions get high weight (learn to do what winners do)
    - Losers' actions get low weight (de-emphasize losing play)
    - Counterfactual folds (losers relabeled as fold) keep high weight

    The weight is stored in player_net_dollars and used later in build_dataset.
    This doesn't modify dp weights directly — build_dataset reads player_net_dollars.
    """
    winners = sum(1 for dp in decision_points if dp.player_net_dollars > 0)
    losers = sum(1 for dp in decision_points if dp.player_net_dollars < 0)
    neutral = len(decision_points) - winners - losers

    print(f'  Reward weighting: {winners} winner dps, {losers} loser dps, '
          f'{neutral} neutral')

    return decision_points


def build_dataset(decision_points: list, action_size: int = 5,
                  weight_by_street: bool = True,
                  weight_by_pot: bool = True,
                  reward_weight: bool = False):
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

    FOLD = 0
    bb = 0.05  # big blind in dollars

    for i, dp in enumerate(decision_points):
        observations[i] = dp.observation
        actions[i] = dp.actual_action
        for a in dp.legal_actions:
            legal_masks[i, a] = 1.0

        w = 1.0
        if weight_by_street:
            w *= street_weights.get(dp.street, 1.0)
        if weight_by_pot and dp.pot_dollars > 0:
            # log(1 + pot/$0.05) so small pots ~1x, $1 pot ~3x, $5 pot ~4.6x
            w *= np.log1p(dp.pot_dollars / 0.05)

        if reward_weight and dp.player_net_dollars != 0:
            net_bb = dp.player_net_dollars / bb
            if dp.actual_action == FOLD:
                # Counterfactual fold: weight by how much was lost (saved by folding)
                # net is negative for the original hand, fold "saves" that loss
                w *= min(abs(net_bb) / 10.0, 3.0)  # cap at 3x
            elif dp.player_net_dollars > 0:
                # Winner: boost weight (learn from winning play)
                w *= min(1.0 + net_bb / 20.0, 3.0)  # +1BB→1.05x, +20BB→2x, cap 3x
            else:
                # Loser (non-fold action): reduce weight
                w *= max(0.2, 1.0 / (1.0 + abs(net_bb) / 10.0))  # -10BB→0.5x, floor 0.2x

        weights[i] = w

    # Normalize weights to mean=1 so loss scale doesn't change
    mean_w = weights.mean()
    if mean_w > 0:
        weights /= mean_w

    return observations, actions, legal_masks, weights


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

# Action cost matrix: confusing similar actions is less bad than opposite ones.
# Rows = target action, Cols = predicted action
# 0=Fold, 1=Call, 2=RaiseHalf, 3=RaisePot, 4=AllIn
# Soft labels: when target is "call", distribute some mass to nearby actions.
_ACTION_SIMILARITY = torch.tensor([
    # fold   call   r_half r_pot  all_in
    [1.00,  0.00,  0.00,  0.00,  0.00],  # fold — unique, no bleed
    [0.00,  1.00,  0.15,  0.05,  0.00],  # call — some bleed to small raise
    [0.00,  0.15,  1.00,  0.20,  0.05],  # raise_half — bleed to call/raise_pot
    [0.00,  0.05,  0.20,  1.00,  0.15],  # raise_pot — bleed to raise_half/allin
    [0.00,  0.00,  0.05,  0.15,  1.00],  # all_in — bleed to raise_pot
], dtype=torch.float32)

# Normalize each row to sum to 1
_ACTION_SIMILARITY = _ACTION_SIMILARITY / _ACTION_SIMILARITY.sum(dim=1, keepdim=True)


def masked_cross_entropy(logits, targets, legal_masks, weights,
                         label_smoothing=0.1, use_action_costs=True):
    """Cross-entropy loss with label smoothing, action cost matrix, and legal masking.

    - Label smoothing: prevents overconfidence (poker is mixed-strategy)
    - Action cost matrix: confusing call/raise is penalized less than call/fold
    - Legal masking: illegal actions get zero probability
    """
    device = logits.device

    # Use -1e4 (not -1e9) to stay safe with float16 AMP
    masked_logits = logits + (1.0 - legal_masks) * (-1e4)

    if use_action_costs and label_smoothing > 0:
        # Build soft targets from action similarity matrix
        sim = _ACTION_SIMILARITY.to(device)
        # soft_targets[i] = sim[targets[i]]  — shape (batch, action_size)
        soft_targets = sim[targets]

        # Zero out illegal actions and renormalize
        soft_targets = soft_targets * legal_masks
        row_sums = soft_targets.sum(dim=1, keepdim=True).clamp(min=1e-8)
        soft_targets = soft_targets / row_sums

        # Blend with uniform over legal actions (label smoothing)
        n_legal = legal_masks.sum(dim=1, keepdim=True).clamp(min=1)
        uniform = legal_masks / n_legal
        blended = (1.0 - label_smoothing) * soft_targets + label_smoothing * uniform

        # KL divergence loss (equivalent to cross-entropy with soft targets)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        nll = -(blended * log_probs).sum(dim=-1)
    elif label_smoothing > 0:
        # Simple label smoothing without action costs
        n_legal = legal_masks.sum(dim=1, keepdim=True).clamp(min=1)
        uniform = legal_masks / n_legal
        one_hot = F.one_hot(targets, logits.size(-1)).float() * legal_masks
        blended = (1.0 - label_smoothing) * one_hot + label_smoothing * uniform

        log_probs = F.log_softmax(masked_logits, dim=-1)
        nll = -(blended * log_probs).sum(dim=-1)
    else:
        nll = F.cross_entropy(masked_logits, targets, reduction='none')

    weighted_loss = (nll * weights).mean()
    return weighted_loss


def train_epoch(model, dataloader, optimizer, device, scaler,
                label_smoothing=0.1, use_action_costs=True):
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
            loss = masked_cross_entropy(logits, acts, masks, wts,
                                        label_smoothing, use_action_costs)

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
def eval_epoch(model, dataloader, device,
               label_smoothing=0.1, use_action_costs=True):
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
            loss = masked_cross_entropy(logits, acts, masks, wts,
                                        label_smoothing, use_action_costs)

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
    parser.add_argument('--weight-pot', action='store_true', default=True,
                        help='Weight decisions by pot size (bigger pots matter more)')
    parser.add_argument('--no-weight-pot', action='store_true',
                        help='Disable pot-size weighting')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing factor (0=off, 0.1=default)')
    parser.add_argument('--no-action-costs', action='store_true',
                        help='Disable action cost matrix (treat all misclassifications equally)')
    parser.add_argument('--counterfactual-folds', action='store_true', default=True,
                        help='Generate synthetic fold labels from losers\' last actions (default: on)')
    parser.add_argument('--no-counterfactual-folds', action='store_true',
                        help='Disable counterfactual fold generation')
    parser.add_argument('--reward-weight', action='store_true', default=True,
                        help='Weight samples by hand outcome (winners up, losers down) (default: on)')
    parser.add_argument('--no-reward-weight', action='store_true',
                        help='Disable reward-based sample weighting')

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

    postflop_only = True
    weight_pot = args.weight_pot and not args.no_weight_pot
    use_action_costs = not args.no_action_costs
    use_counterfactual = args.counterfactual_folds and not args.no_counterfactual_folds
    use_reward_weight = args.reward_weight and not args.no_reward_weight

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
    print(f'  label_sm : {args.label_smoothing}')
    print(f'  act_costs: {use_action_costs}')
    print(f'  pot_wt   : {weight_pot}')
    print(f'  cf_folds : {use_counterfactual}')
    print(f'  reward_wt: {use_reward_weight}')
    print(f'  preflop  : GTO ranges (locked)')
    print(f'  training : postflop only')
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

    # ── Counterfactual fold labels ─────────────────────────────────────────
    if use_counterfactual:
        decision_points = apply_counterfactual_folds(decision_points)

    # ── Reward weighting info ─────────────────────────────────────────────
    if use_reward_weight:
        decision_points = apply_reward_weighting(decision_points)

    # ── Build dataset ────────────────────────────────────────────────────
    print(f'Building dataset...')
    obs_size = decision_points[0].observation.shape[0]
    observations, actions, legal_masks, weights = build_dataset(
        decision_points, action_size, True, weight_pot, use_reward_weight)

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

    # ── File-based train/val split (no data leakage between sessions) ───
    # Group decision points by source file, then split files into train/val
    file_to_indices = defaultdict(list)
    for i, dp in enumerate(decision_points):
        file_to_indices[dp.source_file or f'unknown_{i}'].append(i)

    unique_files = sorted(file_to_indices.keys())
    rng = np.random.RandomState(42)
    rng.shuffle(unique_files)

    val_target = int(len(observations) * args.val_split)
    val_indices = []
    train_indices = []
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
    train_n = len(train_idx)
    val_n = len(val_idx)

    n_val_files = sum(1 for f in unique_files
                      if any(i in set(val_indices) for i in file_to_indices[f][:1]))
    print(f'\n  File-based split: {len(unique_files)} files total, '
          f'~{n_val_files} val files')

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

    print(f'  Train samples: {train_n}')
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # ── Validation engine + preflop ranges ───────────────────────────────
    engine = _make_engine(num_players)

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
    patience = 20
    patience_counter = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_epochs = start_epoch + args.epochs

    for epoch in range(start_epoch + 1, total_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, scaler,
            args.label_smoothing, use_action_costs)
        val_loss, val_acc = eval_epoch(
            model, val_loader, device,
            args.label_smoothing, use_action_costs)

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
                'preflop_ranges': True,
                'label_smoothing': args.label_smoothing,
                'use_action_costs': use_action_costs,
                'counterfactual_folds': use_counterfactual,
                'reward_weight': use_reward_weight,
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
                'preflop_ranges': True,
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
