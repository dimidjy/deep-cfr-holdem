#!/usr/bin/env python3
"""Analyze action distribution for good/bad/all players from hand histories.

Shows fold/call/raise percentages by street and overall, broken down by
player category. This helps calibrate fold oversampling targets.

Usage:
    python3 scripts/analyze_player_actions.py
    python3 scripts/analyze_player_actions.py --good-players data/good_players.txt
    python3 scripts/analyze_player_actions.py --max-hands 50000
"""

import sys
import os
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.hand_history.parser import parse_file


def main():
    parser = argparse.ArgumentParser(description='Analyze player action distributions')
    parser.add_argument('--hh-dir', default='data/poker_archive/')
    parser.add_argument('--good-players', default='')
    parser.add_argument('--bad-players', default='')
    parser.add_argument('--max-hands', type=int, default=100000)
    parser.add_argument('--num-players', type=int, default=6)
    args = parser.parse_args()

    def load_list(path):
        if not path or not Path(path).exists():
            return set()
        return set(line.strip() for line in open(path)
                   if line.strip() and not line.startswith('#'))

    good = load_list(args.good_players)
    bad = load_list(args.bad_players)
    print(f'Good players: {len(good)}')
    print(f'Bad players: {len(bad)}')

    # action counters: category -> street -> action_type -> count
    cats = ['good', 'bad', 'other', 'all']
    counts = {c: defaultdict(lambda: defaultdict(int)) for c in cats}
    # Also track preflop VPIP/PFR
    preflop = {c: {'hands': 0, 'vpip': 0, 'pfr': 0} for c in cats}

    hh_path = Path(args.hh_dir)
    files = sorted(hh_path.rglob('*.txt'))
    print(f'Found {len(files)} files')

    total_hands = 0
    for fi, fp in enumerate(files):
        if total_hands >= args.max_hands:
            break
        try:
            hands = parse_file(str(fp))
        except Exception:
            continue

        for hand in hands:
            if total_hands >= args.max_hands:
                break
            if abs(hand.big_blind_dollars - 0.05) > 0.001:
                continue
            if hand.num_active_players != args.num_players:
                continue
            total_hands += 1

            # Track who posted blinds
            blind_players = set()
            for action in hand.actions:
                if action.is_blind:
                    blind_players.add(action.player_name)

            # Track preflop first voluntary action per player
            preflop_acted = set()

            for action in hand.actions:
                name = action.player_name
                if action.is_blind:
                    continue

                # Determine category
                if name in good:
                    cat = 'good'
                elif name in bad:
                    cat = 'bad'
                else:
                    cat = 'other'

                street = action.street
                atype = action.action_type

                # Map to fold/call/raise
                if atype == 'fold':
                    mapped = 'fold'
                elif atype in ('check', 'call'):
                    mapped = 'call'
                elif atype in ('bet', 'raise'):
                    mapped = 'raise'
                else:
                    continue

                counts[cat][street][mapped] += 1
                counts['all'][street][mapped] += 1

                if action.is_all_in:
                    counts[cat][street]['all_in'] += 1
                    counts['all'][street]['all_in'] += 1

                # Preflop VPIP/PFR tracking
                if street == 'preflop' and name not in preflop_acted:
                    preflop_acted.add(name)
                    preflop[cat]['hands'] += 1
                    preflop['all']['hands'] += 1
                    if mapped != 'fold':
                        preflop[cat]['vpip'] += 1
                        preflop['all']['vpip'] += 1
                    if mapped == 'raise':
                        preflop[cat]['pfr'] += 1
                        preflop['all']['pfr'] += 1

        if (fi + 1) % 500 == 0:
            print(f'  [{fi+1}/{len(files)}] {total_hands} hands...')

    print(f'\nTotal hands analyzed: {total_hands}')
    print()

    # Print results
    streets = ['preflop', 'flop', 'turn', 'river']

    for cat in cats:
        if cat == 'other' and not good and not bad:
            continue
        if cat in ('good', 'bad') and not (good or bad):
            continue

        label = cat.upper()
        print(f'{"="*60}')
        print(f'  {label} PLAYERS')
        print(f'{"="*60}')

        # Preflop stats
        pf = preflop[cat]
        if pf['hands'] > 0:
            print(f'  Preflop: VPIP={100*pf["vpip"]/pf["hands"]:.1f}%  '
                  f'PFR={100*pf["pfr"]/pf["hands"]:.1f}%  '
                  f'(hands={pf["hands"]})')
        print()

        # Per-street action distribution
        print(f'  {"Street":<10} {"Fold":>8} {"Call/Chk":>10} {"Raise/Bet":>10} {"AllIn":>8} {"Total":>8}')
        print(f'  {"-"*50}')

        total_all = defaultdict(int)
        for street in streets:
            sc = counts[cat][street]
            total = sc['fold'] + sc['call'] + sc['raise']
            if total == 0:
                continue
            f_pct = 100 * sc['fold'] / total
            c_pct = 100 * sc['call'] / total
            r_pct = 100 * sc['raise'] / total
            ai_pct = 100 * sc['all_in'] / total if sc['all_in'] else 0

            print(f'  {street:<10} {f_pct:>7.1f}% {c_pct:>9.1f}% {r_pct:>9.1f}% {ai_pct:>7.1f}% {total:>8,}')

            for k in ['fold', 'call', 'raise', 'all_in']:
                total_all[k] += sc[k]

        # Overall postflop
        pf_total = total_all['fold'] + total_all['call'] + total_all['raise']
        postflop_f = sum(counts[cat][s]['fold'] for s in ['flop','turn','river'])
        postflop_c = sum(counts[cat][s]['call'] for s in ['flop','turn','river'])
        postflop_r = sum(counts[cat][s]['raise'] for s in ['flop','turn','river'])
        postflop_total = postflop_f + postflop_c + postflop_r

        if postflop_total > 0:
            print(f'  {"─"*50}')
            print(f'  {"POSTFLOP":<10} {100*postflop_f/postflop_total:>7.1f}% '
                  f'{100*postflop_c/postflop_total:>9.1f}% '
                  f'{100*postflop_r/postflop_total:>9.1f}%'
                  f' {"":>8} {postflop_total:>8,}')

        if pf_total > 0:
            print(f'  {"OVERALL":<10} {100*total_all["fold"]/pf_total:>7.1f}% '
                  f'{100*total_all["call"]/pf_total:>9.1f}% '
                  f'{100*total_all["raise"]/pf_total:>9.1f}%'
                  f' {"":>8} {pf_total:>8,}')
        print()


if __name__ == '__main__':
    main()
