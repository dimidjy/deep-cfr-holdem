"""
Build training dataset with spot-DB-validated fold labels.

Pipeline:
  A) Collect good player winning spots (call/raise with positive outcomes)
  B) Collect bad player losing spots, validate via spot DB similarity
     - Query K nearest neighbors for each candidate
     - If >60% of similar spots also lost → fold label
     - Fold reward = |call_loss_bb| (fold "saves" what call lost)
  C) Merge with configurable fold ratio, maintain hand ordering
  D) Save as .npz for direct loading in train_supervised.py

Usage:
  python3 scripts/build_training_data.py \
    --good-players data/good_players.txt \
    --bad-players data/bad_players.txt \
    --spot-db data/spot_db.npz \
    --fold-ratio 0.15 \
    --workers 47 \
    --output data/training_data.npz
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import multiprocessing as mp
import re
import time
import numpy as np
from collections import defaultdict
from pathlib import Path

from src.hand_history.parser import parse_file
from src.hand_history import ObservationBuilder, BEHAVIORAL_STATS_DIM


# ── Constants ─────────────────────────────────────────────────────────────────

FOLD = 0
ACTION_NAMES = {0: 'fold', 1: 'call', 2: 'raise_half', 3: 'raise_pot', 4: 'all_in'}
STREET_MAP = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
MADE_HAND_NAMES = {
    0.0: 'high_card', 0.125: 'pair', 0.25: 'two_pair', 0.375: 'trips/set',
    0.5: 'straight', 0.625: 'flush', 0.75: 'full_house', 0.875: 'quads', 1.0: 'str_flush',
}


# ── File pre-filter ───────────────────────────────────────────────────────────

def _prefilter_files(hh_files, players, hh_dir):
    """Use ripgrep to quickly find files containing target player names."""
    import subprocess
    import shutil

    if not players:
        return hh_files

    rg_path = shutil.which('rg')
    if not rg_path:
        return hh_files

    t0 = time.time()
    pattern = '|'.join(re.escape(p) for p in players)

    try:
        result = subprocess.run(
            [rg_path, '-l', '--no-messages', '-e', pattern, str(hh_dir)],
            capture_output=True, text=True, timeout=1200
        )
        matched = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()
        filtered = [f for f in hh_files if str(f) in matched]
        elapsed = time.time() - t0
        print(f'  Pre-filter (rg): {len(filtered)}/{len(hh_files)} files contain target players ({elapsed:.1f}s)')
        return filtered
    except Exception as e:
        print(f'  Pre-filter failed: {e}, using all files')
        return hh_files


# ── Worker for parsing ────────────────────────────────────────────────────────

_CFG = {}


def _worker_init(num_players, good_players, bad_players, use_behavioral_stats=False):
    _CFG['num_players'] = num_players
    _CFG['good_players'] = good_players
    _CFG['bad_players'] = bad_players
    _CFG['builder'] = ObservationBuilder(num_players=num_players)
    _CFG['behavioral_stats'] = use_behavioral_stats


def _worker(file_path):
    """Parse one HH file. Returns (good_spots, bad_candidates)."""
    num_players = _CFG['num_players']
    good_players = _CFG['good_players']
    bad_players = _CFG['bad_players']
    builder = _CFG['builder']
    use_bh = _CFG['behavioral_stats']

    good_spots = []   # (obs, action, net_bb, street_idx, legal_mask, hand_id)
    bad_cands = []     # same

    try:
        hands = parse_file(str(file_path))
    except Exception:
        return good_spots, bad_cands, 0, 0

    file_hands = 0
    file_used = 0

    for hand in hands:
        if abs(hand.big_blind_dollars - 0.05) > 0.001:
            continue
        if hand.num_active_players != num_players:
            continue
        if not hand.has_showdown:
            continue

        file_hands += 1
        known_players = set(hand.shown_hands.keys())

        has_good = bool(good_players and (known_players & good_players))
        has_bad = bool(bad_players and (known_players & bad_players))

        if not good_players:
            has_good = bool(known_players)

        if not has_good and not has_bad:
            continue

        dps = builder.build_decision_points(hand, allow_unknown_cards=False,
                                            source_file=str(file_path),
                                            behavioral_stats=use_bh)
        if not dps:
            continue

        # Compute player net profit
        player_winnings = {}
        for w in hand.result.winners:
            name = w['name']
            player_winnings[name] = player_winnings.get(name, 0) + w['amount_dollars']

        player_invested = defaultdict(float)
        round_committed = defaultdict(float)
        current_street = 'preflop'
        for action in hand.actions:
            if action.street != current_street:
                current_street = action.street
                round_committed.clear()
            name = action.player_name
            if action.is_blind or action.action_type in ('bet', 'call'):
                player_invested[name] += action.amount_dollars
                round_committed[name] += action.amount_dollars
            elif action.action_type == 'raise':
                to_amt = action.amount_dollars
                already = round_committed.get(name, 0)
                delta = to_amt - already
                if delta > 0:
                    player_invested[name] += delta
                round_committed[name] = to_amt

        # Fix uncalled bets
        total_won = sum(player_winnings.values())
        total_put = sum(player_invested.values())
        rake = hand.result.rake_dollars
        gap = total_won - total_put + rake
        if abs(gap) > 0.001:
            for action in reversed(hand.actions):
                if action.action_type in ('raise', 'bet') and not action.is_blind:
                    player_invested[action.player_name] += gap
                    break

        player_net = {}
        bb = hand.big_blind_dollars
        for p in hand.active_players:
            won = player_winnings.get(p.name, 0.0)
            invested = player_invested.get(p.name, 0.0)
            player_net[p.name] = (won - invested) / bb  # net in BB

        file_used += 1
        hand_id = hash(hand.hand_id) & 0xFFFFFFFF  # compact hand identifier

        for dp in dps:
            if dp.street == 'preflop':
                continue

            net_bb = player_net.get(dp.player_name, 0.0)
            street_idx = STREET_MAP.get(dp.street, 0)

            # Build legal mask
            action_size = 5
            legal_mask = np.zeros(action_size, dtype=np.float32)
            for a in dp.legal_actions:
                if a < action_size:
                    legal_mask[a] = 1.0

            obs = dp.observation.astype(np.float32)

            spot = (
                obs,
                dp.actual_action,
                np.float32(net_bb),
                np.int8(street_idx),
                legal_mask,
                np.uint32(hand_id),
            )

            # Good player winning spots
            if good_players and dp.player_name in (known_players & good_players):
                if net_bb >= 0:
                    good_spots.append(spot)
            elif not good_players and dp.player_name in known_players:
                if net_bb >= 0:
                    good_spots.append(spot)

            # Bad player losing spots (candidates for fold labels)
            if bad_players and dp.player_name in bad_players:
                if net_bb < 0 and dp.actual_action != FOLD:
                    if FOLD in dp.legal_actions:
                        bad_cands.append(spot)

    return good_spots, bad_cands, file_hands, file_used


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_config(config_path):
    """Load supervised config and return a namespace matching CLI args."""
    import json
    with open(config_path) as f:
        cfg = json.load(f)

    data = cfg.get('data', {})
    spot_db = cfg.get('spot_db', {})
    btd = cfg.get('build_training_data', {})

    class Args: pass
    args = Args()
    args.hh_dir = data.get('hh_dir', 'data/poker_archive/')
    args.good_players = data.get('good_players', '')
    args.bad_players = data.get('bad_players', '')
    args.num_players = data.get('num_players', 6)
    args.workers = data.get('workers', 0)
    args.spot_db = spot_db.get('output', '')
    args.fold_ratio = btd.get('fold_ratio', 0.15)
    args.min_lose_pct = btd.get('min_lose_pct', 60.0)
    args.max_made_hand = btd.get('max_made_hand', 0.25)
    args.spot_k = btd.get('spot_k', 100)
    args.behavioral_stats = btd.get('behavioral_stats', False)
    args.output = btd.get('output', 'data/training_data.npz')
    return args


def main():
    import re

    parser = argparse.ArgumentParser(
        description='Build training dataset with spot-DB-validated fold labels')
    parser.add_argument('--config', type=str, default='configs/supervised_config.json',
                        help='Path to supervised_config.json (default: configs/supervised_config.json)')
    # All params can still be overridden via CLI
    parser.add_argument('--hh-dir', type=str, default=None)
    parser.add_argument('--good-players', type=str, default=None)
    parser.add_argument('--bad-players', type=str, default=None)
    parser.add_argument('--spot-db', type=str, default=None)
    parser.add_argument('--num-players', type=int, default=None)
    parser.add_argument('--fold-ratio', type=float, default=None)
    parser.add_argument('--min-lose-pct', type=float, default=None)
    parser.add_argument('--max-made-hand', type=float, default=None)
    parser.add_argument('--spot-k', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--behavioral-stats', action='store_true', default=None,
                        help='Append 8-dim behavioral stats to observations')
    parser.add_argument('--output', type=str, default=None)
    cli_args = parser.parse_args()

    # Load config, then override with any CLI args
    args = _load_config(cli_args.config)
    for key in ['hh_dir', 'good_players', 'bad_players', 'spot_db', 'num_players',
                'fold_ratio', 'min_lose_pct', 'max_made_hand', 'spot_k', 'workers',
                'behavioral_stats', 'output']:
        cli_val = getattr(cli_args, key, None)
        if cli_val is not None:
            setattr(args, key, cli_val)

    # Load player lists
    def load_players(path):
        players = set()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    players.add(line)
        return players

    good_players = load_players(args.good_players)
    bad_players = load_players(args.bad_players)
    print(f'Good players: {len(good_players)}')
    print(f'Bad players: {len(bad_players)}')

    use_bh = bool(args.behavioral_stats)
    if use_bh:
        print(f'Behavioral stats: ENABLED (+{BEHAVIORAL_STATS_DIM} dims)')

    # Find HH files
    hh_path = Path(args.hh_dir)
    all_files = sorted(hh_path.rglob('*.txt'))
    print(f'Found {len(all_files)} HH files')

    # Pre-filter files
    all_target_players = good_players | bad_players
    filtered_files = _prefilter_files(all_files, all_target_players, args.hh_dir)

    # Parse with workers
    workers = args.workers if args.workers > 0 else min(120, max(1, mp.cpu_count() - 1))
    print(f'\n{"="*70}')
    print(f'  STEP 1: PARSING HAND HISTORIES')
    print(f'{"="*70}')
    print(f'  Parsing {len(filtered_files)} files with {workers} workers...')

    all_good = []
    all_bad_cands = []
    total_hands = 0
    total_used = 0
    done = 0
    t0 = time.time()

    with mp.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(args.num_players, good_players, bad_players, use_bh),
    ) as pool:
        for good_spots, bad_cands, n_hands, n_used in pool.imap_unordered(
            _worker, filtered_files, chunksize=32
        ):
            all_good.extend(good_spots)
            all_bad_cands.extend(bad_cands)
            total_hands += n_hands
            total_used += n_used
            done += 1

            if done % 2000 == 0:
                elapsed = time.time() - t0
                print(f'  [{done}/{len(filtered_files)}] {total_hands} hands, '
                      f'{len(all_good)} good, {len(all_bad_cands)} bad cands  ({elapsed:.0f}s)')

    elapsed = time.time() - t0
    print(f'\n  Parsing done in {elapsed:.0f}s')
    print(f'  Total hands: {total_hands}, used: {total_used}')
    print(f'  Good player spots: {len(all_good)}')
    print(f'  Bad player fold candidates: {len(all_bad_cands)}')

    if not all_good:
        print('ERROR: No good player spots found!')
        sys.exit(1)

    # ── STEP 2: Validate fold candidates via spot DB ─────────────────────

    print(f'\n{"="*70}')
    print(f'  STEP 2: VALIDATING FOLD CANDIDATES VIA SPOT DB')
    print(f'{"="*70}')

    from scipy.spatial import cKDTree as KDTree

    print(f'  Loading spot DB from {args.spot_db}...')
    db_data = np.load(args.spot_db)
    db_obs = db_data['observations']
    db_net_bb = db_data['net_bb']
    db_streets = db_data['streets']
    print(f'  Spot DB: {len(db_obs)} spots')

    # Build per-street KDTree
    trees = {}
    tree_indices = {}
    for s_idx in range(1, 4):  # flop, turn, river
        mask = db_streets == s_idx
        idx = np.where(mask)[0]
        if len(idx) > 0:
            trees[s_idx] = KDTree(db_obs[idx])
            tree_indices[s_idx] = idx
            s_name = {1: 'flop', 2: 'turn', 3: 'river'}[s_idx]
            print(f'    {s_name}: {len(idx)} spots in KDTree')

    # Validate each candidate
    validated_folds = []
    rejected = 0
    no_tree = 0

    # Group candidates by street for batch processing
    for s_idx in range(1, 4):
        s_name = {1: 'flop', 2: 'turn', 3: 'river'}[s_idx]
        street_cands = [c for c in all_bad_cands if c[3] == s_idx]

        if not street_cands or s_idx not in trees:
            no_tree += len(street_cands)
            continue

        tree = trees[s_idx]
        idx_map = tree_indices[s_idx]
        query_k = min(args.spot_k, len(idx_map))

        # Build observation matrix for batch query (use only base obs dims to match spot DB)
        db_dim = db_obs.shape[1]
        obs_matrix = np.array([c[0][:db_dim] for c in street_cands], dtype=np.float32)
        total_cands = len(street_cands)

        print(f'  Querying {s_name}: {total_cands} candidates, k={query_k}...')
        t1 = time.time()

        # Process in chunks for progress tracking
        chunk_size = 10000
        street_validated = 0
        street_rejected = 0

        for chunk_start in range(0, total_cands, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_cands)
            chunk_obs = obs_matrix[chunk_start:chunk_end]

            # KDTree query for this chunk (parallel across all cores)
            distances_chunk, local_indices_chunk = tree.query(chunk_obs, k=query_k, workers=-1)

            for j in range(len(chunk_obs)):
                cand = street_cands[chunk_start + j]
                nn_global = idx_map[local_indices_chunk[j]]
                nn_net_bb = db_net_bb[nn_global]

                # What % of similar spots lost money?
                lose_pct = (nn_net_bb < 0).mean() * 100

                if lose_pct >= args.min_lose_pct:
                    obs, action, net_bb, street, legal_mask, hand_id = cand[0], cand[1], cand[2], cand[3], cand[4], cand[5]
                    # Filter by made hand strength — don't create fold labels for strong hands
                    made_hand = float(obs[19]) if len(obs) > 19 else 0.0
                    if made_hand > args.max_made_hand:
                        street_rejected += 1
                        rejected += 1
                        continue
                    fold_reward = abs(float(net_bb))
                    validated_folds.append((
                        obs,
                        np.int64(FOLD),
                        np.float32(fold_reward),
                        street,
                        legal_mask,
                        hand_id,
                    ))
                    street_validated += 1
                else:
                    street_rejected += 1
                    rejected += 1

            elapsed_chunk = time.time() - t1
            done = chunk_end
            rate = done / elapsed_chunk if elapsed_chunk > 0 else 0
            eta = (total_cands - done) / rate if rate > 0 else 0
            print(f'    [{done}/{total_cands}] validated={street_validated} rejected={street_rejected}  '
                  f'({elapsed_chunk:.0f}s, ~{eta:.0f}s left)', flush=True)

        elapsed_s = time.time() - t1
        print(f'    {s_name} done in {elapsed_s:.0f}s: {street_validated} validated, {street_rejected} rejected')

    print(f'\n  Fold validation results:')
    print(f'    Candidates: {len(all_bad_cands)}')
    print(f'    Validated (>={args.min_lose_pct}% similar lose): {len(validated_folds)}')
    print(f'    Rejected: {rejected}')
    if no_tree:
        print(f'    Skipped (no tree): {no_tree}')

    # Show hand type breakdown of validated folds
    if validated_folds:
        fold_obs = np.array([f[0] for f in validated_folds])
        made_hand_cats = fold_obs[:, 19] if fold_obs.shape[1] > 19 else np.zeros(len(fold_obs))
        print(f'\n  Validated fold hand types:')
        for val, name in sorted(MADE_HAND_NAMES.items()):
            mask = np.abs(made_hand_cats - val) < 0.01
            count = mask.sum()
            if count > 0:
                pct = count / len(validated_folds) * 100
                avg_reward = np.array([f[2] for f in validated_folds])[mask].mean()
                print(f'    {name:<12}: {count:>6} ({pct:>5.1f}%)  avg_reward={avg_reward:+.1f} BB')

    # ── STEP 3: Merge with ratio control ─────────────────────────────────

    print(f'\n{"="*70}')
    print(f'  STEP 3: MERGING DATASETS')
    print(f'{"="*70}')

    n_good = len(all_good)
    n_fold = len(validated_folds)

    if n_fold == 0:
        print('  WARNING: No validated folds. Training data will have no fold labels.')
        target_folds = 0
    else:
        # Calculate how many folds to include based on ratio
        # fold_ratio = n_fold_final / (n_good + n_fold_final)
        # n_fold_final = fold_ratio * n_good / (1 - fold_ratio)
        target_folds = int(args.fold_ratio * n_good / (1 - args.fold_ratio))
        target_folds = min(target_folds, n_fold)  # can't exceed what we have

    print(f'  Good spots: {n_good}')
    print(f'  Available folds: {n_fold}')
    print(f'  Target folds (ratio={args.fold_ratio}): {target_folds}')

    # Subsample folds if needed
    if target_folds < n_fold:
        rng = np.random.RandomState(42)
        fold_indices = rng.choice(n_fold, target_folds, replace=False)
        fold_indices.sort()
        selected_folds = [validated_folds[i] for i in fold_indices]
    else:
        selected_folds = validated_folds

    # Build final arrays
    total_n = n_good + len(selected_folds)
    obs_dim = all_good[0][0].shape[0]
    action_size = 5

    observations = np.zeros((total_n, obs_dim), dtype=np.float32)
    actions = np.zeros(total_n, dtype=np.int64)
    weights = np.zeros(total_n, dtype=np.float32)
    streets = np.zeros(total_n, dtype=np.int8)
    legal_masks = np.zeros((total_n, action_size), dtype=np.float32)

    # Fill good player spots
    for i, spot in enumerate(all_good):
        obs, action, net_bb, street, legal_mask, hand_id = spot[0], spot[1], spot[2], spot[3], spot[4], spot[5]
        observations[i] = obs
        actions[i] = action
        weights[i] = max(float(net_bb), 0.1)  # minimum weight 0.1 for break-even
        streets[i] = street
        legal_masks[i] = legal_mask

    # Fill fold spots
    offset = n_good
    for i, (obs, action, reward, street, legal_mask, hand_id) in enumerate(selected_folds):
        idx = offset + i
        observations[idx] = obs
        actions[idx] = action  # FOLD
        weights[idx] = float(reward)  # |call loss|
        streets[idx] = street
        legal_masks[idx] = legal_mask

    # Normalize weights to mean=1
    mean_w = weights.mean()
    if mean_w > 0:
        weights /= mean_w

    # Shuffle (but keep hand ordering within groups? actually shuffle is fine
    # since we're doing mini-batch SGD, ordering doesn't matter)
    rng = np.random.RandomState(42)
    perm = rng.permutation(total_n)
    observations = observations[perm]
    actions = actions[perm]
    weights = weights[perm]
    streets = streets[perm]
    legal_masks = legal_masks[perm]

    # Print final stats
    actual_fold_pct = (actions == FOLD).mean() * 100
    print(f'\n  Final dataset:')
    print(f'    Total samples: {total_n}')
    print(f'    Observation dim: {obs_dim}')

    print(f'    Street distribution:')
    for s_name, s_idx in STREET_MAP.items():
        if s_idx == 0:
            continue
        count = (streets == s_idx).sum()
        pct = count / total_n * 100
        print(f'      {s_name:<10} {count:>8}  ({pct:.1f}%)')

    print(f'    Action distribution:')
    for a_id, a_name in ACTION_NAMES.items():
        count = (actions == a_id).sum()
        pct = count / total_n * 100
        print(f'      {a_name:<12} {count:>8}  ({pct:.1f}%)')

    print(f'    Actual fold ratio: {actual_fold_pct:.1f}%')
    print(f'    Weight range: {weights.min():.3f} - {weights.max():.3f} (mean={weights.mean():.3f})')

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        weights=weights,
        streets=streets,
        legal_masks=legal_masks,
    )
    mb = output_path.stat().st_size / 1e6
    print(f'\n  Saved to {output_path} ({mb:.0f} MB)')


if __name__ == '__main__':
    main()
