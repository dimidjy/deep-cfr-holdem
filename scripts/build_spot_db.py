"""
Build a spot database from hand histories for similarity-based decision lookup.

Parses showdown hands, extracts decision points with observation vectors,
and stores them with outcomes (action taken + net profit in BB).

Usage:
  python3 scripts/build_spot_db.py --output data/spot_db.npz
  python3 scripts/build_spot_db.py --output data/spot_db.npz --workers 47 --num-players 6
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import multiprocessing as mp
import time
import numpy as np
from collections import defaultdict
from pathlib import Path

from src.hand_history.parser import parse_file
from src.hand_history import ObservationBuilder


# ── Worker setup ─────────────────────────────────────────────────────────────

_CFG = {}


def _worker_init(num_players):
    _CFG['num_players'] = num_players
    _CFG['builder'] = ObservationBuilder(num_players=num_players)


def _worker(file_path):
    """Parse one file, return list of (observation, action, net_bb, street_idx, position_idx)."""
    num_players = _CFG['num_players']
    builder = _CFG['builder']

    street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    pos_map = {'UTG': 0, 'MP': 1, 'CO': 2, 'BTN': 3, 'SB': 4, 'BB': 5}

    results = []
    try:
        hands = parse_file(str(file_path))
    except Exception:
        return results

    for hand in hands:
        if hand.big_blind_dollars <= 0:
            continue
        if hand.num_active_players != num_players:
            continue
        if not hand.has_showdown:
            continue

        bb = hand.big_blind_dollars

        dps = builder.build_decision_points(hand, allow_unknown_cards=False,
                                            source_file=str(file_path))
        if not dps:
            continue

        # Compute player net profit
        player_put_in = defaultdict(float)
        round_committed = defaultdict(float)
        current_street = 'preflop'

        for action in hand.actions:
            if action.street != current_street:
                current_street = action.street
                round_committed.clear()
            name = action.player_name
            if action.is_blind:
                player_put_in[name] += action.amount_dollars
                round_committed[name] += action.amount_dollars
            elif action.action_type == 'call':
                player_put_in[name] += action.amount_dollars
                round_committed[name] += action.amount_dollars
            elif action.action_type == 'bet':
                player_put_in[name] += action.amount_dollars
                round_committed[name] += action.amount_dollars
            elif action.action_type == 'raise':
                to_amt = action.amount_dollars
                already = round_committed.get(name, 0)
                delta = to_amt - already
                if delta > 0:
                    player_put_in[name] += delta
                round_committed[name] = to_amt

        player_won = defaultdict(float)
        for w in hand.result.winners:
            player_won[w['name']] += w['amount_dollars']

        # Fix uncalled bets
        total_won = sum(player_won.values())
        total_put = sum(player_put_in.values())
        rake = hand.result.rake_dollars
        gap = total_won - total_put + rake
        if abs(gap) > 0.001:
            for action in reversed(hand.actions):
                if action.action_type in ('raise', 'bet') and not action.is_blind:
                    player_put_in[action.player_name] += gap
                    break

        player_net = {}
        for p in hand.active_players:
            player_net[p.name] = player_won.get(p.name, 0.0) - player_put_in.get(p.name, 0.0)

        for dp in dps:
            net_bb = player_net.get(dp.player_name, 0.0) / bb
            street_idx = street_map.get(dp.street, 0)
            pos_idx = pos_map.get(dp.position_label, 0)

            results.append((
                dp.observation.astype(np.float32),
                dp.actual_action,
                np.float32(net_bb),
                np.int8(street_idx),
                np.int8(pos_idx),
                np.uint8(len(dp.legal_actions)),
            ))

    return results


def main():
    parser = argparse.ArgumentParser(description='Build spot database from hand histories')
    parser.add_argument('--hh-dir', type=str, default='data/poker_archive/',
                        help='Directory with HH files')
    parser.add_argument('--output', type=str, default='data/spot_db.npz',
                        help='Output .npz file')
    parser.add_argument('--num-players', type=int, default=6,
                        help='Filter for N-player hands (default: 6)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of workers (0=auto)')
    parser.add_argument('--max-files', type=int, default=0,
                        help='Max files to process (0=all)')
    args = parser.parse_args()

    hh_path = Path(args.hh_dir)
    all_files = sorted(hh_path.rglob('*.txt'))
    if args.max_files > 0:
        all_files = all_files[:args.max_files]

    print(f'Found {len(all_files)} HH files')

    workers = args.workers if args.workers > 0 else min(120, max(1, mp.cpu_count() - 1))

    all_obs = []
    all_actions = []
    all_net_bb = []
    all_streets = []
    all_positions = []
    all_n_legal = []

    t0 = time.time()
    done = 0

    print(f'Parsing with {workers} workers (num_players={args.num_players})...')

    with mp.Pool(processes=workers, initializer=_worker_init,
                 initargs=(args.num_players,)) as pool:
        for batch in pool.imap_unordered(_worker, all_files, chunksize=16):
            done += 1
            for obs, action, net_bb, street, pos, n_legal in batch:
                all_obs.append(obs)
                all_actions.append(action)
                all_net_bb.append(net_bb)
                all_streets.append(street)
                all_positions.append(pos)
                all_n_legal.append(n_legal)

            if done % 2000 == 0:
                elapsed = time.time() - t0
                print(f'  [{done}/{len(all_files)}] {len(all_obs)} spots  ({elapsed:.0f}s)')

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.0f}s. {len(all_obs)} total spots.\n')

    if not all_obs:
        print('No spots found!')
        return

    # Stack into arrays
    observations = np.array(all_obs, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.int8)
    net_bb = np.array(all_net_bb, dtype=np.float32)
    streets = np.array(all_streets, dtype=np.int8)
    positions = np.array(all_positions, dtype=np.int8)
    n_legal = np.array(all_n_legal, dtype=np.uint8)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        net_bb=net_bb,
        streets=streets,
        positions=positions,
        n_legal=n_legal,
    )

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f'Saved {len(observations)} spots to {output_path} ({file_size_mb:.1f} MB)')
    print(f'  Obs shape: {observations.shape}')
    print(f'  Actions: {dict(zip(*np.unique(actions, return_counts=True)))}')
    print(f'  Streets: {dict(zip(*np.unique(streets, return_counts=True)))}')
    print(f'  Avg net BB: {net_bb.mean():.2f}')


if __name__ == '__main__':
    main()
