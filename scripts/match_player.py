"""
Compare model predictions against a specific player's actual actions.

For every decision point where the target player acted, queries the model
for its top action and checks if it matches. Reports overall match rate
and breakdowns by street, action type, and position.

Usage:
  python3 scripts/match_player.py --model models/supervised_v9.pt
  python3 scripts/match_player.py --model models/supervised_v9.pt --max-hands 100000
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import hashlib
import pickle
import re
import subprocess
import shutil
import warnings
import logging
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import torch

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

from src.training.deep_cfr import StrategyNetwork
from src.hand_history import ObservationBuilder, DecisionPoint
from src.hand_history.parser import parse_file

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TARGET_PLAYER = 'player_i'
ACTION_NAMES = {0: 'fold', 1: 'call', 2: 'raise_half', 3: 'raise_pot', 4: 'all_in'}
ACTION_SIZE = 5
NUM_PLAYERS = 6


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_path, device):
    """Load a supervised StrategyNetwork from checkpoint."""
    ckpt = torch.load(model_path, map_location=device)
    obs_size = ckpt['obs_size']
    action_size = ckpt['action_size']
    hidden_layers = ckpt['hidden_layers']
    dropout = ckpt.get('dropout', 0.1)

    model = StrategyNetwork(obs_size, action_size, hidden_layers, dropout)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    print(f'  Model loaded: {Path(model_path).name}')
    print(f'    obs_size={obs_size}, layers={hidden_layers}, '
          f'epoch={ckpt.get("epoch","?")}, val_acc={ckpt.get("val_acc",0):.1%}')
    return model, obs_size


def predict_action(model, observation, legal_actions, device):
    """Get model's top predicted action for a decision point."""
    obs_t = torch.FloatTensor(observation).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(obs_t).cpu().numpy()[0]

    # Mask illegal actions
    masked = np.full(ACTION_SIZE, -1e9, dtype=np.float32)
    for a in legal_actions:
        masked[a] = logits[a]

    return int(np.argmax(masked))


def predict_strategy(model, observation, legal_actions, device):
    """Get model's full strategy (probability distribution) for a decision point."""
    obs_t = torch.FloatTensor(observation).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(obs_t).cpu().numpy()[0]

    masked = np.full(ACTION_SIZE, -1e9, dtype=np.float32)
    for a in legal_actions:
        masked[a] = logits[a]
    masked -= masked.max()
    exp_s = np.exp(masked)
    strategy = exp_s / exp_s.sum()

    # Zero residual on illegal
    legal_mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    legal_mask[legal_actions] = 1.0
    strategy *= legal_mask
    s = strategy.sum()
    if s > 0:
        strategy /= s
    else:
        strategy = legal_mask / legal_mask.sum()
    return strategy


# ──────────────────────────────────────────────────────────────────────────────
# Grep pre-filter
# ──────────────────────────────────────────────────────────────────────────────

def prefilter_files(hh_files, player_name, hh_dir):
    """Use grep/rg to quickly find files containing the target player."""
    escaped = re.escape(player_name)
    use_rg = shutil.which('rg') is not None

    t0 = time.time()
    try:
        if use_rg:
            cmd = ['rg', '-l', '--no-ignore', '--glob', '*.txt',
                   '-e', escaped, hh_dir]
        else:
            cmd = ['grep', '-rl', '-E', escaped, '--include=*.txt', hh_dir]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        matched = set()
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line:
                matched.add(os.path.abspath(line))
    except Exception as e:
        print(f'  Warning: grep pre-filter failed ({e}), using all files')
        return hh_files

    elapsed = time.time() - t0
    tool = 'rg' if use_rg else 'grep'
    filtered = [f for f in hh_files if os.path.abspath(str(f)) in matched]
    print(f'  Pre-filter ({tool}): {len(filtered)}/{len(hh_files)} files '
          f'contain "{player_name}" ({elapsed:.1f}s)')
    return filtered


# ──────────────────────────────────────────────────────────────────────────────
# Parallel parsing
# ──────────────────────────────────────────────────────────────────────────────

_CFG = {}


def _worker_init(num_players, target_player):
    _CFG['num_players'] = num_players
    _CFG['target_player'] = target_player
    _CFG['builder'] = ObservationBuilder(num_players=num_players)


def _parse_one_file(file_path):
    """Parse one HH file, extract decision points for target player."""
    num_players = _CFG['num_players']
    target = _CFG['target_player']
    builder = _CFG['builder']

    dps = []
    total_hands = 0
    hands_with_player = 0

    try:
        hands = parse_file(str(file_path))
    except Exception:
        return dps, total_hands, hands_with_player

    for hand in hands:
        if abs(hand.big_blind_dollars - 0.05) > 0.001:
            continue
        if hand.num_active_players != num_players:
            continue

        total_hands += 1

        # Check if target player is in this hand
        player_names = [p.name for p in hand.active_players]
        if target not in player_names:
            continue

        # We need showdown + known cards for observation building
        if not hand.has_showdown:
            continue
        if target not in hand.shown_hands:
            continue

        hands_with_player += 1

        hand_dps = builder.build_decision_points(
            hand, allow_unknown_cards=False, source_file=str(file_path))
        if not hand_dps:
            continue

        for dp in hand_dps:
            if dp.player_name == target:
                dps.append(dp)

    return dps, total_hands, hands_with_player


def collect_decision_points(hh_dir, max_hands, no_cache):
    """Collect all decision points for the target player."""
    hh_path = Path(hh_dir)
    all_files = sorted(hh_path.rglob('*.txt'))
    print(f'  Found {len(all_files)} .txt files in {hh_dir}')

    hh_files = prefilter_files(all_files, TARGET_PLAYER, hh_dir)
    if not hh_files:
        return []

    # Cache
    cache_key = f'match|{hh_dir}|{len(all_files)}|flt{len(hh_files)}|{TARGET_PLAYER}|max{max_hands}'
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    cache_dir = Path(__file__).parent.parent / 'data' / '.hh_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'match_{TARGET_PLAYER}_{cache_hash}.pkl'

    if cache_file.exists() and not no_cache:
        print(f'  Loading cached data from {cache_file.name}...')
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f'  Loaded {len(data["dps"])} decision points '
              f'from {data["hands_with_player"]} hands')
        return data['dps']

    # Parallel parse
    workers = min(120, max(1, mp.cpu_count() - 1))
    all_dps = []
    total_hands = 0
    hands_with_player = 0
    t0 = time.time()

    print(f'  Parsing {len(hh_files)} files with {workers} workers...')

    with mp.Pool(processes=workers, initializer=_worker_init,
                 initargs=(NUM_PLAYERS, TARGET_PLAYER)) as pool:
        results = pool.imap_unordered(_parse_one_file, hh_files, chunksize=16)
        files_done = 0
        for file_dps, file_total, file_player in results:
            all_dps.extend(file_dps)
            total_hands += file_total
            hands_with_player += file_player
            files_done += 1

            if files_done % 500 == 0:
                elapsed = time.time() - t0
                print(f'  [{files_done}/{len(hh_files)}] '
                      f'{total_hands} hands, {len(all_dps)} dps  ({elapsed:.0f}s)')

            if max_hands > 0 and hands_with_player >= max_hands:
                pool.terminate()
                print(f'  Reached {max_hands} hands limit')
                break

    elapsed = time.time() - t0
    print(f'  Parsed {files_done} files in {elapsed:.1f}s')
    print(f'  Total 6-max NL5 hands: {total_hands}')
    print(f'  Hands with {TARGET_PLAYER} at showdown: {hands_with_player}')
    print(f'  Decision points: {len(all_dps)}')

    # Save cache
    data = {'dps': all_dps, 'hands_with_player': hands_with_player}
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        mb = cache_file.stat().st_size / 1e6
        print(f'  Cached to {cache_file.name} ({mb:.1f} MB)')
    except Exception as e:
        print(f'  Cache save failed: {e}')

    return all_dps


# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze(dps, model, device):
    """Compare model predictions vs player actions, print detailed report."""

    total = len(dps)
    matches = 0

    # Breakdown trackers
    by_street = defaultdict(lambda: {'total': 0, 'match': 0})
    by_action = defaultdict(lambda: {'total': 0, 'match': 0})
    by_position = defaultdict(lambda: {'total': 0, 'match': 0})
    # Confusion matrix: actual_action -> predicted_action -> count
    confusion = defaultdict(lambda: defaultdict(int))
    # Probability the model assigns to the player's actual action
    prob_of_actual = []

    for dp in dps:
        pred = predict_action(model, dp.observation, dp.legal_actions, device)
        strategy = predict_strategy(model, dp.observation, dp.legal_actions, device)
        actual = dp.actual_action

        matched = (pred == actual)
        if matched:
            matches += 1

        by_street[dp.street]['total'] += 1
        by_action[ACTION_NAMES[actual]]['total'] += 1
        by_position[dp.position_label or 'unknown']['total'] += 1
        confusion[actual][pred] += 1

        if matched:
            by_street[dp.street]['match'] += 1
            by_action[ACTION_NAMES[actual]]['match'] += 1
            by_position[dp.position_label or 'unknown']['match'] += 1

        prob_of_actual.append(strategy[actual])

    # ── Report ────────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'  MODEL vs {TARGET_PLAYER}  —  ACTION MATCH REPORT')
    print(f'{"="*70}')
    print(f'\n  Overall: {matches}/{total} = {matches/total:.1%} exact match')
    print(f'  Avg probability on actual action: {np.mean(prob_of_actual):.1%}')

    # By street
    print(f'\n  By street:')
    for street in ['preflop', 'flop', 'turn', 'river']:
        d = by_street.get(street)
        if d and d['total'] > 0:
            pct = d['match'] / d['total']
            print(f'    {street:<10} {d["match"]:>5}/{d["total"]:<5} = {pct:.1%}')

    # By action
    print(f'\n  By actual action:')
    for a in range(ACTION_SIZE):
        name = ACTION_NAMES[a]
        d = by_action.get(name)
        if d and d['total'] > 0:
            pct = d['match'] / d['total']
            print(f'    {name:<12} {d["match"]:>5}/{d["total"]:<5} = {pct:.1%}')

    # By position
    print(f'\n  By position:')
    pos_order = ['BTN', 'SB', 'BB', 'UTG', 'MP', 'CO']
    for pos in pos_order:
        d = by_position.get(pos)
        if d and d['total'] > 0:
            pct = d['match'] / d['total']
            print(f'    {pos:<6} {d["match"]:>5}/{d["total"]:<5} = {pct:.1%}')
    # Any positions not in standard list
    for pos, d in sorted(by_position.items()):
        if pos not in pos_order and d['total'] > 0:
            pct = d['match'] / d['total']
            print(f'    {pos:<6} {d["match"]:>5}/{d["total"]:<5} = {pct:.1%}')

    # Confusion matrix
    print(f'\n  Confusion matrix (rows=actual, cols=predicted):')
    header = '            ' + ''.join(f'{ACTION_NAMES[a]:>10}' for a in range(ACTION_SIZE))
    print(f'  {header}')
    for actual in range(ACTION_SIZE):
        row_total = sum(confusion[actual].values())
        if row_total == 0:
            continue
        cells = []
        for pred in range(ACTION_SIZE):
            cnt = confusion[actual][pred]
            if row_total > 0:
                cells.append(f'{cnt:>5}({cnt/row_total:>4.0%})')
            else:
                cells.append(f'{"":>10}')
        row = f'    {ACTION_NAMES[actual]:<10}' + ''.join(cells)
        print(row)

    print(f'\n{"="*70}')


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=f'Compare model predictions vs {TARGET_PLAYER}\'s actions')
    parser.add_argument('--model', type=str, default='models/supervised_v9.pt',
                        help='Model checkpoint path')
    parser.add_argument('--hh-dir', type=str, default='data/poker_archive/',
                        help='Hand history directory')
    parser.add_argument('--max-hands', type=int, default=0,
                        help='Max hands to process (0=all)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force re-parse')
    parser.add_argument('--device', type=str, default=None,
                        help='cuda | cpu')
    args = parser.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    print(f'{"="*70}')
    print(f'  PLAYER ACTION MATCH  —  {TARGET_PLAYER}')
    print(f'{"="*70}')
    print(f'  device: {device}')

    model, obs_size = load_model(args.model, device)

    print(f'\nCollecting {TARGET_PLAYER}\'s decision points...')
    dps = collect_decision_points(args.hh_dir, args.max_hands, args.no_cache)

    if not dps:
        print(f'No decision points found for {TARGET_PLAYER}')
        sys.exit(1)

    analyze(dps, model, device)


if __name__ == '__main__':
    main()
