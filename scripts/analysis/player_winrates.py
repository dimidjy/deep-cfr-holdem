#!/usr/bin/env python3
"""
Player win rate analysis: compare BB/100 from ALL hands vs SHOWDOWN-ONLY hands.

Shows the survivorship bias in showdown-only data: players who go to showdown
and win have inflated win rates because the data excludes hands they folded.

This is critical for understanding CQL training bias — models trained on
showdown data see a world where everyone calls and "good" play means calling.

Usage:
  python3 scripts/analysis/player_winrates.py
  python3 scripts/analysis/player_winrates.py --min-hands 5000 --top 30
  python3 scripts/analysis/player_winrates.py --good-players data/good_players.txt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import multiprocessing as mp
import time
import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from src.hand_history.parser import parse_file


def _worker(file_path):
    """Parse one file, return per-hand results with showdown flag."""
    results = []
    try:
        hands = parse_file(str(file_path))
    except Exception:
        return results

    for hand in hands:
        bb = hand.big_blind_dollars
        if bb <= 0:
            continue
        active = hand.active_players
        if len(active) < 2:
            continue
        if hand.num_active_players != 6:
            continue

        all_names = set(p.name for p in active)
        has_showdown = hand.has_showdown
        showdown_players = set(hand.shown_hands.keys()) if has_showdown else set()

        # VPIP / PFR
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

        # Player net profit
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
        for name in all_names:
            player_net[name] = player_won.get(name, 0.0) - player_put_in.get(name, 0.0)

        results.append({
            'bb': bb,
            'players': all_names,
            'vpip': vpip_players,
            'pfr': pfr_players,
            'player_net': player_net,
            'has_showdown': has_showdown,
            'showdown_players': showdown_players,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compare player BB/100 from all hands vs showdown-only')
    parser.add_argument('--hh-dir', type=str, default='data/poker_archive/')
    parser.add_argument('--min-hands', type=int, default=3000)
    parser.add_argument('--top', type=int, default=50,
                        help='Show top N players (default: 50)')
    parser.add_argument('--good-players', type=str, default='',
                        help='Highlight these players in output')
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()

    hh_path = Path(args.hh_dir)
    all_files = sorted(hh_path.rglob('*.txt'))
    print(f'Found {len(all_files)} HH files in {args.hh_dir}')

    good_names = set()
    if args.good_players and Path(args.good_players).exists():
        good_names = set(
            l.strip() for l in Path(args.good_players).read_text().splitlines()
            if l.strip() and not l.startswith('#'))
        print(f'Good players list: {len(good_names)} names loaded')

    workers = args.workers if args.workers > 0 else min(120, max(1, mp.cpu_count() - 1))

    # All-hands stats
    p_hands_all = defaultdict(int)
    p_net_all = defaultdict(float)
    # Showdown-only stats (only count hands where player went to showdown)
    p_hands_sd = defaultdict(int)
    p_net_sd = defaultdict(float)
    # VPIP/PFR (from all hands)
    p_vpip = defaultdict(int)
    p_pfr = defaultdict(int)

    total_hands = 0
    showdown_hands = 0

    t0 = time.time()
    done = 0

    print(f'Parsing with {workers} workers...')
    with mp.Pool(processes=workers) as pool:
        for hand_list in pool.imap_unordered(_worker, all_files, chunksize=32):
            done += 1
            for hand in hand_list:
                bb = hand['bb']
                total_hands += 1
                is_sd = hand['has_showdown']
                if is_sd:
                    showdown_hands += 1

                for name in hand['players']:
                    net_bb = hand['player_net'].get(name, 0.0) / bb
                    p_hands_all[name] += 1
                    p_net_all[name] += net_bb
                    if name in hand['vpip']:
                        p_vpip[name] += 1
                    if name in hand['pfr']:
                        p_pfr[name] += 1

                    # Showdown stats: only if this hand had showdown
                    if is_sd:
                        p_hands_sd[name] += 1
                        p_net_sd[name] += net_bb

            if done % 2000 == 0:
                elapsed = time.time() - t0
                print(f'  [{done}/{len(all_files)}] {len(p_hands_all)} players  ({elapsed:.0f}s)')

    elapsed = time.time() - t0
    print(f'\nParsed {total_hands:,} hands ({showdown_hands:,} with showdown = '
          f'{100*showdown_hands/total_hands:.1f}%) in {elapsed:.0f}s')
    print(f'Unique players: {len(p_hands_all):,}\n')

    # Build stats for players with enough hands
    stats = []
    for name in p_hands_all:
        h_all = p_hands_all[name]
        if h_all < args.min_hands:
            continue

        bb100_all = p_net_all[name] / h_all * 100
        h_sd = p_hands_sd.get(name, 0)
        bb100_sd = (p_net_sd[name] / h_sd * 100) if h_sd > 0 else 0.0
        vpip = p_vpip[name] / h_all * 100
        pfr = p_pfr[name] / h_all * 100

        stats.append({
            'name': name,
            'hands_all': h_all,
            'bb100_all': bb100_all,
            'hands_sd': h_sd,
            'bb100_sd': bb100_sd,
            'sd_pct': 100 * h_sd / h_all if h_all > 0 else 0,
            'bias': bb100_sd - bb100_all,  # showdown bias
            'vpip': vpip,
            'pfr': pfr,
            'is_good': name in good_names,
        })

    stats.sort(key=lambda x: x['bb100_all'], reverse=True)

    # ── Report ──
    lines = []
    w = lines.append

    w('PLAYER WIN RATE ANALYSIS: ALL HANDS vs SHOWDOWN-ONLY')
    w('=' * 95)
    w(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    w(f'Total hands: {total_hands:,}  |  Showdown hands: {showdown_hands:,} '
      f'({100*showdown_hands/total_hands:.1f}%)')
    w(f'Players with {args.min_hands}+ hands: {len(stats)}')
    w('')

    # Aggregate stats
    winners = [s for s in stats if s['bb100_all'] > 0]
    losers = [s for s in stats if s['bb100_all'] <= 0]
    good_found = [s for s in stats if s['is_good']]

    if winners:
        avg_bias_winners = sum(s['bias'] for s in winners) / len(winners)
        avg_bb100_winners = sum(s['bb100_all'] for s in winners) / len(winners)
        avg_bb100_sd_winners = sum(s['bb100_sd'] for s in winners) / len(winners)
    else:
        avg_bias_winners = avg_bb100_winners = avg_bb100_sd_winners = 0

    if losers:
        avg_bias_losers = sum(s['bias'] for s in losers) / len(losers)
        avg_bb100_losers = sum(s['bb100_all'] for s in losers) / len(losers)
        avg_bb100_sd_losers = sum(s['bb100_sd'] for s in losers) / len(losers)
    else:
        avg_bias_losers = avg_bb100_losers = avg_bb100_sd_losers = 0

    w('─' * 95)
    w('SUMMARY: SHOWDOWN BIAS')
    w('─' * 95)
    w(f'  {"Group":<20} {"Count":>6} {"BB/100 All":>12} {"BB/100 SD":>12} '
      f'{"Bias":>10}  Explanation')
    w(f'  {"-"*20} {"-"*6} {"-"*12} {"-"*12} {"-"*10}  {"-"*20}')
    w(f'  {"Winners (BB/100>0)":<20} {len(winners):>6} '
      f'{avg_bb100_winners:>+12.1f} {avg_bb100_sd_winners:>+12.1f} '
      f'{avg_bias_winners:>+10.1f}  SD inflates winners')
    w(f'  {"Losers (BB/100<=0)":<20} {len(losers):>6} '
      f'{avg_bb100_losers:>+12.1f} {avg_bb100_sd_losers:>+12.1f} '
      f'{avg_bias_losers:>+10.1f}  SD makes losers look less bad')
    if good_found:
        avg_bb_good = sum(s['bb100_all'] for s in good_found) / len(good_found)
        avg_sd_good = sum(s['bb100_sd'] for s in good_found) / len(good_found)
        avg_bias_good = sum(s['bias'] for s in good_found) / len(good_found)
        w(f'  {"Good players":<20} {len(good_found):>6} '
          f'{avg_bb_good:>+12.1f} {avg_sd_good:>+12.1f} '
          f'{avg_bias_good:>+10.1f}  Training data source')
    w('')

    w('─' * 95)
    w(f'TOP {args.top} PLAYERS BY TRUE WIN RATE (all hands)')
    w('─' * 95)
    w(f'  {"Player":<22} {"Hands":>7} {"BB/100":>8} {"SD Hands":>9} '
      f'{"BB/100 SD":>10} {"Bias":>8} {"SD%":>5} {"VPIP":>6} {"PFR":>5} '
      f'{"":>4}')
    w(f'  {"-"*22} {"-"*7} {"-"*8} {"-"*9} {"-"*10} {"-"*8} {"-"*5} '
      f'{"-"*6} {"-"*5} {"-"*4}')

    for p in stats[:args.top]:
        marker = ' *' if p['is_good'] else ''
        w(f'  {p["name"]:<22} {p["hands_all"]:>7} {p["bb100_all"]:>+8.1f} '
          f'{p["hands_sd"]:>9} {p["bb100_sd"]:>+10.1f} '
          f'{p["bias"]:>+8.1f} {p["sd_pct"]:>4.0f}% '
          f'{p["vpip"]:>5.1f}% {p["pfr"]:>4.1f}%{marker}')

    w('')
    w('─' * 95)
    w(f'BOTTOM {args.top} PLAYERS (biggest losers)')
    w('─' * 95)
    w(f'  {"Player":<22} {"Hands":>7} {"BB/100":>8} {"SD Hands":>9} '
      f'{"BB/100 SD":>10} {"Bias":>8} {"SD%":>5} {"VPIP":>6} {"PFR":>5}')
    w(f'  {"-"*22} {"-"*7} {"-"*8} {"-"*9} {"-"*10} {"-"*8} {"-"*5} '
      f'{"-"*6} {"-"*5}')

    for p in reversed(stats[-args.top:]):
        w(f'  {p["name"]:<22} {p["hands_all"]:>7} {p["bb100_all"]:>+8.1f} '
          f'{p["hands_sd"]:>9} {p["bb100_sd"]:>+10.1f} '
          f'{p["bias"]:>+8.1f} {p["sd_pct"]:>4.0f}% '
          f'{p["vpip"]:>5.1f}% {p["pfr"]:>4.1f}%')

    # Good players detail
    if good_found:
        w('')
        w('─' * 95)
        w('GOOD PLAYERS (from list) — DETAIL')
        w('─' * 95)
        w(f'  {"Player":<22} {"Hands":>7} {"BB/100":>8} {"SD Hands":>9} '
          f'{"BB/100 SD":>10} {"Bias":>8} {"SD%":>5} {"VPIP":>6} {"PFR":>5}')
        w(f'  {"-"*22} {"-"*7} {"-"*8} {"-"*9} {"-"*10} {"-"*8} {"-"*5} '
          f'{"-"*6} {"-"*5}')
        good_found.sort(key=lambda x: x['bb100_all'], reverse=True)
        for p in good_found:
            w(f'  {p["name"]:<22} {p["hands_all"]:>7} {p["bb100_all"]:>+8.1f} '
              f'{p["hands_sd"]:>9} {p["bb100_sd"]:>+10.1f} '
              f'{p["bias"]:>+8.1f} {p["sd_pct"]:>4.0f}% '
              f'{p["vpip"]:>5.1f}% {p["pfr"]:>4.1f}%')

    w('')
    w('=' * 95)
    w('KEY INSIGHT: "Bias" = BB/100(showdown) - BB/100(all)')
    w('  Positive bias means showdown data inflates this player\'s apparent skill.')
    w('  CQL models trained only on showdown data see the inflated numbers.')
    w('=' * 95)

    report = '\n'.join(lines)
    print(report)

    # Save
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    txt_path = results_dir / f'player_winrates_{timestamp}.txt'
    txt_path.write_text(report)

    json_data = {
        'total_hands': total_hands,
        'showdown_hands': showdown_hands,
        'showdown_pct': round(100 * showdown_hands / total_hands, 1),
        'players_analyzed': len(stats),
        'avg_bias_winners': round(avg_bias_winners, 1),
        'avg_bias_losers': round(avg_bias_losers, 1),
        'players': [
            {k: (round(v, 1) if isinstance(v, float) else v)
             for k, v in p.items()}
            for p in stats
        ],
    }
    json_path = results_dir / f'player_winrates_{timestamp}.json'
    json_path.write_text(json.dumps(json_data, indent=2))

    # Save reward scale factors for CQL training
    # scale = true_bb100 / showdown_bb100 (clamped to [0.05, 1.0])
    # For losers or tiny SD samples, use the population average scale
    avg_scale = (avg_bb100_winners / avg_bb100_sd_winners
                 if avg_bb100_sd_winners > 1 else 0.2)
    reward_scales = {}
    for p in stats:
        if p['hands_sd'] >= 100 and abs(p['bb100_sd']) > 1.0:
            raw = p['bb100_all'] / p['bb100_sd']
            # Clamp: never invert sign, keep in [0.05, 1.0]
            scale = max(0.05, min(1.0, raw))
        else:
            scale = round(avg_scale, 3)
        reward_scales[p['name']] = round(scale, 3)

    scales_path = Path('data/reward_scales.json')
    scales_path.parent.mkdir(parents=True, exist_ok=True)
    scales_path.write_text(json.dumps(reward_scales, indent=2))
    n_scaled = sum(1 for v in reward_scales.values() if v < 0.95)
    print(f'  Reward scales: {len(reward_scales)} players, '
          f'{n_scaled} with scale < 0.95 (avg winner scale: {avg_scale:.3f})')
    print(f'  Saved to: {scales_path}')

    print(f'\nSaved to: {txt_path}')


if __name__ == '__main__':
    main()
