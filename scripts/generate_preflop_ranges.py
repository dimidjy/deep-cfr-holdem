"""
Generate GTO-approximate preflop range configs for 2-6 player tables.

Uses a hand strength ordering and assigns ranges by position/table size.
Outputs JSON files to configs/preflop_ranges_{N}p.json

Usage:
  python3 scripts/generate_preflop_ranges.py
"""

import json
from pathlib import Path

# Hands ordered roughly by preflop equity (strongest first)
# This ordering determines which hands enter the range as we widen
HAND_ORDER = [
    # Tier 1: Premium (top ~3%)
    'AA', 'KK', 'QQ', 'JJ', 'AKS',
    # Tier 2: Strong (top ~8%)
    'AKO', 'AQS', 'TT', 'AQO', 'AJS', 'KQS',
    # Tier 3: Good (top ~15%)
    '99', 'ATS', 'AJO', 'KQO', 'KJS', 'QJS', 'JTS',
    '88', 'ATO', 'KTS', 'QTS', 'KJO',
    # Tier 4: Playable (top ~25%)
    '77', 'A9S', 'A8S', 'A7S', 'A6S', 'A5S', 'A4S', 'A3S', 'A2S',
    'K9S', 'T9S', 'J9S', '98S', 'QJO', 'KTO',
    # Tier 5: Wider (top ~35%)
    '66', 'A9O', 'K8S', 'Q9S', '87S', 'QTO', 'JTO', '76S', '65S',
    'K7S', 'A8O', '97S', 'T8S', '86S', '54S',
    # Tier 6: Wide (top ~45%)
    '55', 'K9O', 'Q9O', 'J8S', 'K6S', '75S', '96S', '53S', '64S',
    'A7O', 'K5S', 'T9O', '43S', 'Q8S', 'J9O',
    # Tier 7: Very wide (top ~55%)
    '44', 'A6O', 'K8O', 'A5O', 'K4S', 'T7S', 'Q7S', 'K3S', 'J7S',
    '85S', '98O', '74S', 'K2S', 'Q6S', 'A4O', '87O', '63S', '52S',
    # Tier 8: Ultra wide (top ~65%)
    '33', 'A3O', 'Q5S', 'J6S', 'T8O', 'K7O', 'Q8O', 'T6S', '42S',
    '95S', 'A2O', '76O', 'J8O', '84S', 'Q4S', '65O', '97O', '73S',
    # Tier 9: Near any-two (top ~80%)
    '22', 'K6O', 'Q3S', 'J5S', 'T5S', '86O', 'Q2S', '54O', '62S',
    'K5O', '96O', 'J4S', 'T4S', '94S', '75O', '93S', '32S', 'K4O',
    'J3S', 'T3S', 'Q7O', '64O', '92S', '83S', '53O', '43O', '82S',
    'K3O', 'J2S', 'T2S', 'Q6O', '85O', 'K2O', '72S',
    # Tier 10: Any two
    'J7O', 'Q5O', 'T7O', '74O', 'Q4O', '63O', 'J6O', '95O', 'T6O',
    'Q3O', '52O', '42O', '84O', 'J5O', 'Q2O', 'T5O', 'J4O', '73O',
    '94O', '32O', 'J3O', '62O', 'T4O', '93O', '82O', 'J2O', 'T3O',
    '83O', '92O', 'T2O', '72O',
]

# Total = 169 hand types
assert len(set(HAND_ORDER)) == 169, f"Expected 169, got {len(set(HAND_ORDER))}"


def make_range(hands):
    """Convert list of hand strings to range dict with frequencies."""
    return {h: "0.99" for h in hands}


def slice_range(pct):
    """Get hands for a given range percentage (0-100)."""
    n = max(1, int(len(HAND_ORDER) * pct / 100))
    return HAND_ORDER[:n]


def generate_ranges(num_players):
    """Generate complete preflop ranges for a given table size."""

    # Position names per table size
    # rel_pos mapping: 0=BTN, 1=SB, 2=BB, 3+=EP positions
    if num_players == 6:
        positions = {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'LJ', 4: 'HJ', 5: 'CO'}
    elif num_players == 5:
        positions = {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'HJ', 4: 'CO'}
    elif num_players == 4:
        positions = {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'CO'}
    elif num_players == 3:
        positions = {0: 'BTN', 1: 'SB', 2: 'BB'}
    elif num_players == 2:
        positions = {0: 'BTN', 1: 'BB'}

    # RFI ranges (% of hands to open-raise when first in)
    rfi_pct = {
        6: {'LJ': 22, 'HJ': 28, 'CO': 33, 'BTN': 45, 'SB': 40, 'BB': 30},
        5: {'HJ': 26, 'CO': 33, 'BTN': 48, 'SB': 43, 'BB': 33},
        4: {'CO': 33, 'BTN': 50, 'SB': 46, 'BB': 36},
        3: {'BTN': 55, 'SB': 48, 'BB': 38},
        2: {'BTN': 80, 'BB': 45},
    }

    # Call vs RFI ranges (% of hands to call a single raise)
    call_pct = {
        6: {'LJ': 16, 'HJ': 22, 'CO': 25, 'BTN': 28, 'SB': 20, 'BB': 35},
        5: {'HJ': 20, 'CO': 25, 'BTN': 30, 'SB': 22, 'BB': 38},
        4: {'CO': 25, 'BTN': 32, 'SB': 25, 'BB': 40},
        3: {'BTN': 35, 'SB': 28, 'BB': 42},
        2: {'BTN': 55, 'BB': 45},
    }

    # 3-bet ranges (% of hands to 3-bet when facing raise)
    three_bet_pct = {
        6: {'LJ': 5, 'HJ': 5, 'CO': 6, 'BTN': 7, 'SB': 7, 'BB': 7},
        5: {'HJ': 5, 'CO': 6, 'BTN': 8, 'SB': 8, 'BB': 8},
        4: {'CO': 7, 'BTN': 9, 'SB': 9, 'BB': 9},
        3: {'BTN': 10, 'SB': 10, 'BB': 10},
        2: {'BTN': 14, 'BB': 12},
    }

    # 3-bet call ranges (call a 3-bet, % of hands)
    three_bet_call_pct = {
        6: {'LJ': 8, 'HJ': 9, 'CO': 10, 'BTN': 11, 'SB': 10, 'BB': 12},
        5: {'HJ': 9, 'CO': 10, 'BTN': 12, 'SB': 11, 'BB': 13},
        4: {'CO': 10, 'BTN': 13, 'SB': 12, 'BB': 14},
        3: {'BTN': 14, 'SB': 13, 'BB': 15},
        2: {'BTN': 18, 'BB': 16},
    }

    # 4-bet ranges
    four_bet_pct = {
        6: {'LJ': 3, 'HJ': 3, 'CO': 3, 'BTN': 4, 'SB': 4, 'BB': 4},
        5: {'HJ': 3, 'CO': 4, 'BTN': 4, 'SB': 4, 'BB': 4},
        4: {'CO': 4, 'BTN': 5, 'SB': 5, 'BB': 5},
        3: {'BTN': 5, 'SB': 5, 'BB': 5},
        2: {'BTN': 7, 'BB': 6},
    }

    # 4-bet call ranges
    four_bet_call_pct = {
        6: {'LJ': 4, 'HJ': 4, 'CO': 4, 'BTN': 5, 'SB': 5, 'BB': 5},
        5: {'HJ': 4, 'CO': 5, 'BTN': 5, 'SB': 5, 'BB': 5},
        4: {'CO': 5, 'BTN': 6, 'SB': 6, 'BB': 6},
        3: {'BTN': 6, 'SB': 6, 'BB': 6},
        2: {'BTN': 8, 'BB': 7},
    }

    # All-in range (same for all)
    allin_hands = ['AA', 'KK', 'QQ', 'AKS', 'AKO']
    if num_players <= 3:
        allin_hands += ['JJ', 'AQS']
    if num_players == 2:
        allin_hands += ['AQO', 'TT']

    # SB defense (complete vs BB) — only for 3+ players
    sb_def_pct = {
        6: 55, 5: 58, 4: 62, 3: 65, 2: 0,  # no SB in HU
    }

    np = num_players
    pos_names = list(positions.values())

    # Build ranges dict
    ranges = {}

    # RFI
    ranges['rfi'] = {}
    for pos in pos_names:
        pct = rfi_pct[np].get(pos, 25)
        ranges['rfi'][pos] = make_range(slice_range(pct))

    # RFI call (call vs single raise)
    ranges['rfi_call'] = {}
    for pos in pos_names:
        pct = call_pct[np].get(pos, 20)
        ranges['rfi_call'][pos] = make_range(slice_range(pct))

    # 3-bet
    ranges['3b'] = {}
    for pos in pos_names:
        pct = three_bet_pct[np].get(pos, 6)
        key = f'{pos}_3b'
        ranges['3b'][key] = make_range(slice_range(pct))

    # 3-bet call
    ranges['3b_call'] = {}
    for pos in pos_names:
        pct = three_bet_call_pct[np].get(pos, 8)
        key = f'{pos}_call'
        ranges['3b_call'][key] = make_range(slice_range(pct))

    # 4-bet
    ranges['4b'] = {}
    for pos in pos_names:
        pct = four_bet_pct[np].get(pos, 3)
        key = f'{pos}_4b'
        ranges['4b'][key] = make_range(slice_range(pct))

    # 4-bet call
    ranges['4b_call'] = {}
    for pos in pos_names:
        pct = four_bet_call_pct[np].get(pos, 4)
        key = f'{pos}_call'
        ranges['4b_call'][key] = make_range(slice_range(pct))

    # All-in
    ranges['Allin'] = {'allin_base': make_range(allin_hands)}

    # SB defense (limp/complete)
    if num_players >= 3:
        sb_pct = sb_def_pct.get(np, 55)
        ranges['SB_DEF'] = {'SB_DEF_call': make_range(slice_range(sb_pct))}

    return ranges, positions


def print_summary(num_players, ranges, positions):
    """Print range summary for verification."""
    pos_names = list(positions.values())
    print(f"\n{'='*50}")
    print(f"  {num_players}-max preflop ranges")
    print(f"{'='*50}")
    print(f"  {'Position':<8} {'RFI':>6} {'Call':>6} {'3bet':>6} {'3b_call':>6} {'4bet':>6}")
    print(f"  {'-'*44}")
    for pos in pos_names:
        rfi_n = len(ranges['rfi'].get(pos, {}))
        call_n = len(ranges['rfi_call'].get(pos, {}))
        tb_key = f'{pos}_3b'
        tb_n = len(ranges['3b'].get(tb_key, {}))
        tbc_key = f'{pos}_call'
        tbc_n = len(ranges['3b_call'].get(tbc_key, {}))
        fb_key = f'{pos}_4b'
        fb_n = len(ranges['4b'].get(fb_key, {}))
        print(f"  {pos:<8} {rfi_n:>4}({rfi_n/169*100:>3.0f}%) "
              f"{call_n:>4}({call_n/169*100:>3.0f}%) "
              f"{tb_n:>4}({tb_n/169*100:>3.0f}%) "
              f"{tbc_n:>4}({tbc_n/169*100:>3.0f}%) "
              f"{fb_n:>4}({fb_n/169*100:>3.0f}%)")


def main():
    configs_dir = Path(__file__).parent.parent / 'configs'

    for np in [2, 3, 4, 5, 6]:
        ranges, positions = generate_ranges(np)
        print_summary(np, ranges, positions)

        out_path = configs_dir / f'preflop_ranges_{np}p.json'
        with open(out_path, 'w') as f:
            json.dump(ranges, f, indent=2)
        print(f"  Saved to {out_path}")

    print(f"\nDone! Generated configs for 2-6 players.")


if __name__ == '__main__':
    main()
