"""
Find winning, tight-ish players from PokerStars hand history archives.

Uses the project's existing hand history parser for correctness.

Criteria:
  - Win rate >= 5 BB/100 hands
  - VPIP 15-30% (voluntarily put money in pot — not too loose, not too nitty)
  - PFR 10-25% (preflop raise — shows initiative but not maniac)
  - Min 3000 hands (statistically meaningful sample)

Usage:
  python3 scripts/find_good_players.py --hh-dir data/poker_archive/ --output data/good_players.txt
  python3 scripts/find_good_players.py --hh-dir data/poker_archive/ --min-hands 5000 --min-bb100 7
  python3 scripts/find_good_players.py --hh-dir data/poker_archive/ --workers 47
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

from src.hand_history.parser import parse_file


def _worker(file_path):
    """Parse one file and extract per-hand player stats."""
    results = []
    try:
        hands = parse_file(str(file_path))
    except Exception:
        return results

    for hand in hands:
        if "Hold'em No Limit" not in str(hand.hand_id):
            # parse_file only parses NL Hold'em already, but just in case
            pass

        bb = hand.big_blind_dollars
        if bb <= 0:
            continue

        active = hand.active_players
        if len(active) < 2:
            continue

        all_names = set(p.name for p in active)

        # ── VPIP and PFR (preflop voluntary actions) ──────────────────
        vpip_players = set()
        pfr_players = set()
        for action in hand.actions:
            if action.street != 'preflop' or action.is_blind:
                continue
            if action.action_type == 'call':
                vpip_players.add(action.player_name)
            elif action.action_type in ('raise', 'bet'):
                vpip_players.add(action.player_name)
                pfr_players.add(action.player_name)

        # ── Player net profit ─────────────────────────────────────────
        # Track total put in per player using round-committed for raises
        player_put_in = defaultdict(float)
        round_committed = defaultdict(float)
        current_street = 'preflop'

        for action in hand.actions:
            # Track street changes
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
                # amount_dollars is the "to" amount (total committed this round)
                to_amt = action.amount_dollars
                already = round_committed.get(name, 0)
                delta = to_amt - already
                if delta > 0:
                    player_put_in[name] += delta
                round_committed[name] = to_amt

        # Winnings from collected
        player_won = defaultdict(float)
        for w in hand.result.winners:
            player_won[w['name']] += w['amount_dollars']

        # Fix uncalled bets: total net for all players should equal -rake
        # expected: total_won - total_put = -rake
        # gap = (total_won - total_put) + rake  (should be 0 if balanced)
        total_won = sum(player_won.values())
        total_put = sum(player_put_in.values())
        rake = hand.result.rake_dollars
        gap = total_won - total_put + rake
        if abs(gap) > 0.001:
            # Unbalanced — uncalled bet was over-counted in put_in
            # Reduce the last aggressor's put_in by the gap amount
            for action in reversed(hand.actions):
                if action.action_type in ('raise', 'bet') and not action.is_blind:
                    player_put_in[action.player_name] += gap  # gap is negative: adds to won; positive: reduces put_in
                    break

        # Net = won - put_in
        player_net = {}
        for name in all_names:
            player_net[name] = player_won.get(name, 0.0) - player_put_in.get(name, 0.0)

        results.append({
            'bb': bb,
            'players': all_names,
            'vpip': vpip_players,
            'pfr': pfr_players,
            'player_net': player_net,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Find good players from hand histories')
    parser.add_argument('--hh-dir', type=str, default='data/poker_archive/',
                        help='Directory with PokerStars HH files')
    parser.add_argument('--output', type=str, default='data/good_players.txt',
                        help='Output file for player names')
    parser.add_argument('--min-hands', type=int, default=3000,
                        help='Minimum hands played (default: 3000)')
    parser.add_argument('--min-bb100', type=float, default=5.0,
                        help='Minimum BB/100 win rate (default: 5.0)')
    parser.add_argument('--max-bb100', type=float, default=999.0,
                        help='Maximum BB/100 win rate (default: 999)')
    parser.add_argument('--vpip-min', type=float, default=15.0,
                        help='Minimum VPIP %% (default: 15)')
    parser.add_argument('--vpip-max', type=float, default=30.0,
                        help='Maximum VPIP %% (default: 30)')
    parser.add_argument('--pfr-min', type=float, default=10.0,
                        help='Minimum PFR %% (default: 10)')
    parser.add_argument('--pfr-max', type=float, default=25.0,
                        help='Maximum PFR %% (default: 25)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of workers (0=auto)')
    parser.add_argument('--show-all', action='store_true',
                        help='Show all players with enough hands (before filtering)')
    args = parser.parse_args()

    hh_path = Path(args.hh_dir)
    all_files = sorted(hh_path.rglob('*.txt'))
    print(f'Found {len(all_files)} HH files in {args.hh_dir}')

    workers = args.workers if args.workers > 0 else min(120, max(1, mp.cpu_count() - 1))

    # Per-player stats
    player_hands = defaultdict(int)
    player_vpip = defaultdict(int)
    player_pfr = defaultdict(int)
    player_net_bb = defaultdict(float)

    t0 = time.time()
    done = 0

    print(f'Parsing with {workers} workers...')

    with mp.Pool(processes=workers) as pool:
        for hand_list in pool.imap_unordered(_worker, all_files, chunksize=32):
            done += 1
            for hand in hand_list:
                bb = hand['bb']
                for name in hand['players']:
                    player_hands[name] += 1
                    if name in hand['vpip']:
                        player_vpip[name] += 1
                    if name in hand['pfr']:
                        player_pfr[name] += 1
                    player_net_bb[name] += hand['player_net'].get(name, 0.0) / bb

            if done % 2000 == 0:
                elapsed = time.time() - t0
                total_players = len(player_hands)
                print(f'  [{done}/{len(all_files)}] {total_players} unique players  ({elapsed:.0f}s)')

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.0f}s. {len(player_hands)} unique players found.\n')

    # Compute stats
    player_stats = []
    for name in player_hands:
        hands = player_hands[name]
        if hands < args.min_hands:
            continue

        bb100 = player_net_bb[name] / hands * 100
        vpip_pct = player_vpip[name] / hands * 100
        pfr_pct = player_pfr[name] / hands * 100

        player_stats.append({
            'name': name,
            'hands': hands,
            'bb100': bb100,
            'vpip': vpip_pct,
            'pfr': pfr_pct,
            'net_bb': player_net_bb[name],
        })

    player_stats.sort(key=lambda x: x['bb100'], reverse=True)

    if args.show_all:
        print(f'{"Player":<25} {"Hands":>7} {"BB/100":>8} {"VPIP%":>6} {"PFR%":>6} {"NetBB":>9}')
        print('-' * 65)
        for p in player_stats:
            print(f'{p["name"]:<25} {p["hands"]:>7} {p["bb100"]:>8.1f} '
                  f'{p["vpip"]:>5.1f}% {p["pfr"]:>5.1f}% {p["net_bb"]:>9.1f}')
        print()

    # Filter
    good = []
    for p in player_stats:
        if p['bb100'] < args.min_bb100 or p['bb100'] > args.max_bb100:
            continue
        if not (args.vpip_min <= p['vpip'] <= args.vpip_max):
            continue
        if not (args.pfr_min <= p['pfr'] <= args.pfr_max):
            continue
        good.append(p)

    print(f'Filters: BB/100 {args.min_bb100} to {args.max_bb100}, VPIP {args.vpip_min}-{args.vpip_max}%, '
          f'PFR {args.pfr_min}-{args.pfr_max}%, min {args.min_hands} hands')
    print(f'Players with >= {args.min_hands} hands: {len(player_stats)}')
    print(f'Good players found: {len(good)}\n')

    if good:
        print(f'{"Player":<25} {"Hands":>7} {"BB/100":>8} {"VPIP%":>6} {"PFR%":>6} {"NetBB":>9}')
        print('-' * 65)
        for p in good:
            print(f'{p["name"]:<25} {p["hands"]:>7} {p["bb100"]:>8.1f} '
                  f'{p["vpip"]:>5.1f}% {p["pfr"]:>5.1f}% {p["net_bb"]:>9.1f}')

        # Save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(f'# Good players: BB/100 >= {args.min_bb100}, '
                    f'VPIP {args.vpip_min}-{args.vpip_max}%, '
                    f'PFR {args.pfr_min}-{args.pfr_max}%, '
                    f'min {args.min_hands} hands\n')
            f.write(f'# Generated from {len(all_files)} HH files\n')
            for p in good:
                f.write(f'{p["name"]}\n')

        print(f'\nSaved {len(good)} player names to {args.output}')
    else:
        print('No players matched the criteria. Try relaxing filters:')
        print('  --min-bb100 3 --vpip-min 12 --vpip-max 35 --min-hands 2000')


if __name__ == '__main__':
    main()
