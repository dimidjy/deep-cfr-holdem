"""Validate archetype preflop ranges by simulating 6-max hands.

For each archetype:
  1. Load configs/archetype_ranges/{archetype}.json as HERO's range
  2. Load configs/preflop_ranges.json as OPPONENT's range (reference pool)
  3. Simulate 10,000 6-max hands, rotating hero through all 6 positions
  4. Measure HERO's VPIP, PFR, 3B% — this is how archetype plays vs standard pool
  5. Compare to target stats from src/archetypes.py ARCHETYPE_PROFILES

Important: measuring an archetype in an all-same-archetype table gives wrong stats
(e.g. calling station VPIP only registers when OPPONENTS raise, which rarely happens
if all 6 are stations). Must measure vs a reference field.

Usage:
    python3 scripts/validate_archetype_ranges.py
    python3 scripts/validate_archetype_ranges.py --hands 50000
    python3 scripts/validate_archetype_ranges.py --archetype nit
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preflop_ranges import PreflopRanges, FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN

ROOT = Path(__file__).resolve().parent.parent
RANGES_DIR = ROOT / 'configs' / 'archetype_ranges'

# Target stats per archetype (from src/archetypes.py ARCHETYPE_PROFILES)
# Format: (VPIP, PFR, 3B%)  — we focus on preflop-controllable stats
TARGETS = {
    'nit':             {'vpip': 12.0, 'pfr': 10.0, 'threeb': 2.5},
    'weak_tight':      {'vpip': 18.0, 'pfr': 14.0, 'threeb': 3.0},
    'calling_station': {'vpip': 55.0, 'pfr': 10.0, 'threeb': 1.0},
    'lag':             {'vpip': 32.0, 'pfr': 25.0, 'threeb': 6.0},
    'maniac':          {'vpip': 60.0, 'pfr': 45.0, 'threeb': 12.0},
}

# Card deal helpers
RANKS = '23456789TJQKA'
SUITS = 'shdc'
DECK = [r + s for r in RANKS for s in SUITS]

POSITIONS = ['BTN', 'SB', 'BB', 'LJ', 'HJ', 'CO']  # rel_pos 0..5

RAISE_ACTIONS = {RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN}


def deal_hands(num_players: int = 6) -> List[List[str]]:
    """Deal num_players × 2 hole cards from shuffled deck."""
    deck = DECK.copy()
    random.shuffle(deck)
    return [[deck[2 * i], deck[2 * i + 1]] for i in range(num_players)]


def simulate_preflop_hand(hero_ranges: PreflopRanges,
                          opp_ranges: PreflopRanges,
                          hero_pos: str,
                          num_players: int = 6) -> Dict:
    """Simulate one hand of preflop action.

    `hero_pos` is one of POSITIONS (BTN/SB/BB/LJ/HJ/CO). That seat uses
    hero_ranges; all other seats use opp_ranges.

    Returns per-position action classification:
      - 'fold': folded at some point
      - 'call': voluntarily called (VPIP, no PFR)
      - 'raise': made first raise (RFI, VPIP + PFR, 1 raise in hand at time)
      - '3bet': made second raise (VPIP + PFR + 3B, 2 raises)
      - '4bet': made third raise (VPIP + PFR, 3 raises)
      - 'check': checked option in BB (no VPIP, no PFR)
    """
    hands = deal_hands(num_players)

    # Per-position state
    per_position = {pos: {
        'hole_cards': hands[i],
        'rel_pos': i,
        'active': True,
        'voluntarily_committed': False,
        'raised': False,
        'action_type': None,
        'is_hero': (pos == hero_pos),
    } for i, pos in enumerate(POSITIONS)}

    # Start action: SB=0.5, BB=1, pot=1.5 (in BB units)
    pot_bb = 1.5
    current_bet = 1.0  # Amount to match to stay in
    num_raises = 0     # 0=unopened, 1=single raise (facing open), 2=3bet, 3=4bet

    # Preflop order: LJ, HJ, CO, BTN, SB, BB  (UTG-like order for 6-max)
    action_order = ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']

    # Track who hasn't acted yet in current round
    needs_action = set(action_order)

    max_rounds = 5  # safety cap
    for round_num in range(max_rounds):
        someone_raised_this_round = False
        for pos in action_order:
            player = per_position[pos]
            if not player['active']:
                continue
            if pos not in needs_action:
                continue

            # Determine legal actions based on current state
            if current_bet == 1.0 and pos == 'BB':
                # BB can check their option
                legal = [CALL, RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN]
            elif num_raises == 0:
                # Unopened pot — fold or open
                legal = [FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN]
            else:
                legal = [FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN]

            rel_pos = player['rel_pos']
            ranges = hero_ranges if player['is_hero'] else opp_ranges

            # PreflopRanges scenario detection is based on pot_bb thresholds:
            #   <4 = RFI, 4-18 = facing raise, 18-50 = facing 3bet, >50 = 4bet
            # A min-raise only inflates pot to ~3.5, which incorrectly stays
            # in RFI band. Force effective_pot_bb into the right band based
            # on num_raises so range lookup uses the correct scenario.
            effective_pot_bb = pot_bb
            if num_raises >= 1:
                effective_pot_bb = max(effective_pot_bb, 5.0)
            if num_raises >= 2:
                effective_pot_bb = max(effective_pot_bb, 20.0)
            if num_raises >= 3:
                effective_pot_bb = max(effective_pot_bb, 55.0)

            action = ranges.get_action(
                hole_cards=player['hole_cards'],
                rel_pos=rel_pos,
                pot_bb=effective_pot_bb,
                legal_actions=legal,
                num_limpers=0,
                effective_stack=100.0,
                num_callers=0,
            )

            if action is None:
                # Ranges don't cover — fallback to fold
                action = FOLD

            needs_action.discard(pos)

            if action == FOLD:
                player['active'] = False
                if player['action_type'] is None:
                    player['action_type'] = 'fold'
            elif action == CALL:
                player['voluntarily_committed'] = True
                if player['action_type'] is None:
                    if current_bet == 1.0 and pos == 'BB':
                        # BB just checking option — no voluntary commit
                        player['voluntarily_committed'] = False
                        player['action_type'] = 'check'
                    else:
                        player['action_type'] = 'call'
                # Add call amount to pot
                pot_bb += current_bet  # simplified; real amount depends on prior bets
            elif action in RAISE_ACTIONS:
                player['voluntarily_committed'] = True
                player['raised'] = True
                num_raises += 1
                if player['action_type'] is None:
                    if num_raises == 1:
                        player['action_type'] = 'raise'
                    elif num_raises == 2:
                        player['action_type'] = '3bet'
                    elif num_raises == 3:
                        player['action_type'] = '4bet'
                    else:
                        player['action_type'] = '4bet'  # treat 5bet+ as 4bet for stats

                # Compute new raise size (simplified)
                if action == RAISE_MIN:
                    new_bet = current_bet + 1.0
                elif action == RAISE_HALF:
                    new_bet = current_bet + 0.5 * pot_bb
                elif action == RAISE_POT:
                    new_bet = current_bet + 1.0 * pot_bb
                else:  # ALL_IN
                    new_bet = 100.0  # effective stack

                increment = new_bet - current_bet
                pot_bb += new_bet
                current_bet = new_bet
                someone_raised_this_round = True
                # Everyone else needs to act again
                needs_action = {
                    p for p in action_order
                    if per_position[p]['active'] and p != pos
                }

        # Round complete — check termination
        if not someone_raised_this_round:
            break
        # Check if only one player left
        active_players = [p for p in action_order if per_position[p]['active']]
        if len(active_players) <= 1:
            break

    # Default action_type for players who never acted (shouldn't happen normally)
    for pos in POSITIONS:
        if per_position[pos]['action_type'] is None:
            per_position[pos]['action_type'] = 'fold'

    return {pos: p['action_type'] for pos, p in per_position.items()}


def aggregate_stats(results: List[Tuple[Dict, str]]) -> Dict:
    """Compute HERO's VPIP/PFR/3B per position + overall.

    results is a list of (per_position_actions, hero_pos) tuples.
    Only hero's stats counted.
    """
    per_pos = {pos: {'hands': 0, 'vpip': 0, 'pfr': 0, 'threeb': 0}
               for pos in POSITIONS}
    for result, hero_pos in results:
        action_type = result[hero_pos]
        per_pos[hero_pos]['hands'] += 1
        if action_type in ('call', 'raise', '3bet', '4bet'):
            per_pos[hero_pos]['vpip'] += 1
        if action_type in ('raise', '3bet', '4bet'):
            per_pos[hero_pos]['pfr'] += 1
        if action_type in ('3bet', '4bet'):
            per_pos[hero_pos]['threeb'] += 1

    # Per-position percentages
    summary = {}
    for pos in POSITIONS:
        s = per_pos[pos]
        n = max(s['hands'], 1)
        summary[pos] = {
            'vpip': 100.0 * s['vpip'] / n,
            'pfr': 100.0 * s['pfr'] / n,
            'threeb': 100.0 * s['threeb'] / n,
            'hands': s['hands'],
        }

    # Overall
    total_hands = sum(per_pos[p]['hands'] for p in POSITIONS)
    total_vpip = sum(per_pos[p]['vpip'] for p in POSITIONS)
    total_pfr = sum(per_pos[p]['pfr'] for p in POSITIONS)
    total_threeb = sum(per_pos[p]['threeb'] for p in POSITIONS)
    summary['OVERALL'] = {
        'vpip': 100.0 * total_vpip / max(total_hands, 1),
        'pfr': 100.0 * total_pfr / max(total_hands, 1),
        'threeb': 100.0 * total_threeb / max(total_hands, 1),
        'hands': total_hands,
    }
    return summary


def print_report(archetype: str, stats: Dict, target: Dict) -> None:
    print(f'\n═══ {archetype.upper()} ═══')
    print(f'  Target: VPIP={target["vpip"]:.0f}%  PFR={target["pfr"]:.0f}%  3B={target["threeb"]:.1f}%')

    print(f'  {"Pos":<8} {"Hands":>7} {"VPIP":>7} {"PFR":>7} {"3B":>7}')
    for pos in POSITIONS + ['OVERALL']:
        s = stats[pos]
        print(f'  {pos:<8} {s["hands"]:>7} {s["vpip"]:>6.1f}% {s["pfr"]:>6.1f}% {s["threeb"]:>6.1f}%')

    # Diff line
    ov = stats['OVERALL']
    vpip_diff = ov['vpip'] - target['vpip']
    pfr_diff = ov['pfr'] - target['pfr']
    threeb_diff = ov['threeb'] - target['threeb']

    def status(diff, tolerance=3.0):
        if abs(diff) <= tolerance:
            return '✓ OK'
        elif diff > 0:
            return f'↑ TOO HIGH (+{diff:.1f})'
        else:
            return f'↓ TOO LOW ({diff:.1f})'

    print(f'  VPIP diff: {status(vpip_diff)}')
    print(f'  PFR  diff: {status(pfr_diff)}')
    print(f'  3B   diff: {status(threeb_diff, tolerance=2.0)}')


def validate_archetype(name: str, hands: int, opp_ranges: PreflopRanges) -> Tuple[Dict, bool]:
    """Load + simulate + report one archetype. Returns (stats, passed).

    Hero uses archetype ranges; opponents use `opp_ranges` (reference pool).
    Hero rotates through all 6 positions evenly.
    """
    ranges_path = RANGES_DIR / f'{name}.json'
    if not ranges_path.exists():
        print(f'ERROR: {ranges_path} not found. Run generate_archetype_ranges.py first.')
        return {}, False

    hero_ranges = PreflopRanges(ranges_path=str(ranges_path), num_players=6)

    print(f'\nSimulating {hands:,} hands with {name} hero vs reference pool...', flush=True)
    results = []
    for i in range(hands):
        hero_pos = POSITIONS[i % len(POSITIONS)]  # rotate hero evenly
        actions = simulate_preflop_hand(hero_ranges, opp_ranges, hero_pos)
        results.append((actions, hero_pos))

    stats = aggregate_stats(results)
    target = TARGETS[name]
    print_report(name, stats, target)

    # Pass criteria: within 5% absolute on VPIP and PFR, 3% on 3B
    ov = stats['OVERALL']
    vpip_ok = abs(ov['vpip'] - target['vpip']) <= 5.0
    pfr_ok = abs(ov['pfr'] - target['pfr']) <= 5.0
    threeb_ok = abs(ov['threeb'] - target['threeb']) <= 3.0
    passed = vpip_ok and pfr_ok and threeb_ok
    return stats, passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hands', type=int, default=10000,
                    help='Hands to simulate per archetype (default 10000)')
    ap.add_argument('--archetype', type=str, default=None,
                    choices=list(TARGETS.keys()),
                    help='Test only this archetype (default: all)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    archetypes = [args.archetype] if args.archetype else list(TARGETS.keys())

    # Load reference opponent ranges (standard GTO-ish population)
    opp_ranges_path = ROOT / 'configs' / 'preflop_ranges.json'
    opp_ranges = PreflopRanges(ranges_path=str(opp_ranges_path), num_players=6)
    print(f'Opponent reference pool: {opp_ranges_path.name}')

    results = {}
    for name in archetypes:
        stats, passed = validate_archetype(name, args.hands, opp_ranges)
        results[name] = (stats, passed)

    # Summary table
    print('\n' + '═' * 80)
    print(' VALIDATION SUMMARY')
    print('═' * 80)
    print(f'  {"Archetype":<20} {"VPIP actual":>12} {"target":>8} {"Δ":>7} '
          f'{"PFR actual":>11} {"target":>8} {"Δ":>7}  {"PASS":>5}')
    print('  ' + '-' * 78)
    for name in archetypes:
        stats, passed = results[name]
        if not stats:
            continue
        ov = stats['OVERALL']
        t = TARGETS[name]
        status_str = 'PASS' if passed else 'FAIL'
        print(f'  {name:<20} {ov["vpip"]:>11.1f}% {t["vpip"]:>7.1f}% '
              f'{ov["vpip"]-t["vpip"]:>+6.1f}% '
              f'{ov["pfr"]:>10.1f}% {t["pfr"]:>7.1f}% '
              f'{ov["pfr"]-t["pfr"]:>+6.1f}%  {status_str:>5}')
    print('═' * 80)


if __name__ == '__main__':
    main()
