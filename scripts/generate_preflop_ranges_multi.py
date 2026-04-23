"""Generate tight-value preflop range files for 2p, 3p, 4p, 5p tables.

Each output file matches the structure of configs/preflop_ranges.json (6-max)
but with position-appropriate range widths for short-handed play.

Design principles:
  - VALUE ONLY (no 3-bet bluffs, no speculative hands added)
  - Tight-to-moderate widths (not maximally wide)
  - Consistent with the user's current style: exploit NL5 pool via value,
    don't rely on fold equity that doesn't exist against stations
  - Reuse 6-max ranges as base where appropriate, widen only for positions
    that are structurally different at shorter tables

pos_map keys used per table size (from src/preflop_ranges.py):
  2-max: BTN, BB
  3-max: BTN, SB, BB
  4-max: BTN, SB, BB, LJ
  5-max: BTN, SB, BB, LJ, HJ

Usage:
    python3 scripts/generate_preflop_ranges_multi.py
"""
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'configs' / 'preflop_ranges.json'
CONFIGS = ROOT / 'configs'


def add(rng: dict, hands: list, val: str = '0.99') -> dict:
    """Add hands to a range dict, preserving existing ':' prefix convention."""
    out = dict(rng)
    for h in hands:
        if h in out:
            continue
        out[h] = f':{val}' if (len(h) == 2 and h[0] == h[1]) else val
    return out


def remove(rng: dict, hands: list) -> dict:
    """Remove specific hands from a range dict."""
    return {k: v for k, v in rng.items() if k not in hands}


def build_5p(base: dict) -> dict:
    """5-max ranges — minor adjustments from 6-max.

    pos_map uses BTN, SB, BB, LJ, HJ. LJ = first-to-act (4 behind vs 5 in 6-max).
    HJ = second-to-act. CO key is unused. Widen LJ slightly.
    """
    cfg = deepcopy(base)

    # LJ (first-to-act in 5-max, 4 behind): slightly wider than 6-max LJ
    cfg['rfi']['LJ'] = add(cfg['rfi']['LJ'], [
        '66',                                # add 66 for set mining
        'K9S',                               # value suited king
        'J9S',                               # value suited jack
    ])
    # HJ unchanged from 6-max (structural match)

    # BTN: unchanged (same structural position)
    # SB: unchanged
    # BB defense: unchanged
    # 3b / 4b / 3b_call / 4b_call: all unchanged (value-only already)

    return cfg


def build_4p(base: dict) -> dict:
    """4-max ranges — widen LJ (first-to-act, 3 behind).

    pos_map uses BTN, SB, BB, LJ. LJ = first-to-act.
    """
    cfg = deepcopy(base)

    # LJ (first-to-act in 4-max, 3 behind): wider than 6-max LJ with VALUE hands
    cfg['rfi']['LJ'] = add(cfg['rfi']['LJ'], [
        '66', '55', '44', '33', '22',        # pairs (set-mining is +EV at 4-max)
        'A6S', 'A7S', 'A8S',                 # suited aces for value
        'KTS', 'K9S',                        # suited kings
        'QTS', 'Q9S',                        # suited queens
        'J9S',                               # suited jacks
        'KJO',                               # offsuit broadway
    ])

    # BB defense: slightly wider with VALUE hands (more pot odds vs steals)
    cfg['rfi_call']['BB'] = add(cfg['rfi_call']['BB'], [
        'K5S', 'K6S',                        # suited kings (wheel + backdoor flush)
        'Q5S', 'Q6S', 'Q7S',                 # suited queens
        'J6S', 'J7S',                        # suited jacks
        'T7S',                               # suited tens
        '75S', '76S', '65S', '64S',          # suited connectors
        '54S', '53S',                        # low suited connectors
    ])

    return cfg


def build_3p(base: dict) -> dict:
    """3-max ranges — 3 players, only BTN/SB/BB used.

    BTN has 2 behind (SB, BB). SB has 1 behind (BB). Widen both.
    """
    cfg = deepcopy(base)

    # BTN in 3-max: moderate widening (still value-focused, no bluff gappers)
    cfg['rfi']['BTN'] = add(cfg['rfi']['BTN'], [
        'K3S',                               # value kings for flush/straight potential
        'Q5S',                               # queens
        'J6S',                               # jacks
        'T6S',                               # tens
        'T8O',                               # offsuit broadway
        '98O',                               # offsuit connector
    ])

    # SB in 3-max (first to act HU vs BB): widen moderately
    cfg['rfi']['SB'] = add(cfg['rfi']['SB'], [
        'K4S',                               # suited kings
        'Q6S',                               # suited queens
        'J6S',                               # suited jacks
        'A6O',                               # offsuit aces (value vs BB only)
        'KJO', 'KTO', 'QTO',                 # offsuit broadways
    ])

    # BB defense in 3-max: wider since often facing BTN steals
    cfg['rfi_call']['BB'] = add(cfg['rfi_call']['BB'], [
        'K5S', 'K6S',
        'Q5S', 'Q6S', 'Q7S',
        'J6S', 'J7S',
        'T7S',
        '75S', '76S', '65S', '64S',
        '54S', '53S', '43S',
    ])

    # Slight widening to BB 3-bet range (value vs wider BTN steals)
    cfg['3b']['BB_3b'] = add(cfg['3b']['BB_3b'], [
        '88',                                # pair value vs steal
    ])

    return cfg


def build_2p(base: dict) -> dict:
    """2-max (HU) ranges — BTN (=SB) vs BB.

    HU is a different game. Keep tight VALUE focus but widen meaningfully
    because every hand has showdown value HU.
    """
    cfg = deepcopy(base)

    # BTN in HU: wider than 3-max BTN since only BB to fold
    cfg['rfi']['BTN'] = add(cfg['rfi']['BTN'], [
        'K2S', 'K3S',                        # all suited kings (flush/pair value)
        'Q3S', 'Q4S', 'Q5S',                 # suited queens
        'J5S', 'J6S',                        # suited jacks
        'T5S', 'T6S', 'T7S',                 # suited tens
        '97S', '86S', '75S', '64S', '53S',   # suited connectors / gappers
        '54S', '43S',                        # low suited connectors
        'T8O', '98O', '87O', '76O',          # offsuit connectors (HU value)
        'K7O', 'K8O',                        # offsuit kings (HU value)
        'Q8O', 'Q9O',                        # offsuit queens
        'J8O', 'J9O',                        # offsuit jacks
    ])

    # BB defense HU: wide (getting 2.5:1 on calls vs min-raise)
    cfg['rfi_call']['BB'] = add(cfg['rfi_call']['BB'], [
        'K2S', 'K3S', 'K4S',                 # all suited kings
        'Q2S', 'Q3S', 'Q4S', 'Q5S',          # all suited queens
        'J2S', 'J3S', 'J4S', 'J5S',          # all suited jacks
        'T2S', 'T3S', 'T4S', 'T5S', 'T6S',   # all suited tens
        '92S', '93S', '94S', '95S', '96S',   # suited 9x (value in HU)
        '82S', '83S', '84S', '85S',          # suited 8x
        '72S', '73S', '74S',                 # suited 7x
        '63S', '62S',                        # suited 6x low
        '52S', '42S', '32S',                 # low suited
        'T7O', 'T8O',                        # offsuit tens (HU)
        '98O', '87O', '76O',                 # offsuit connectors
    ])

    # BTN 3-bet (wait, BTN opens — this is BB 3-bet response)
    # BB 3-bet HU: slightly wider value
    cfg['3b']['BB_3b'] = add(cfg['3b']['BB_3b'], [
        '88', '77',                          # pairs value vs wide BTN opens
    ])

    return cfg


# ─────────────────────────────────────────────────────────────────────────────

def summarize(name: str, cfg: dict) -> None:
    def count(rng):
        total = 0
        for h in rng:
            if len(h) == 2: total += 6
            elif h.endswith('S'): total += 4
            elif h.endswith('O'): total += 12
        return total
    print(f'\n── {name} ──')
    for section, name_sec in [('rfi', 'RFI'), ('rfi_call', 'RFI call'), ('3b', '3b')]:
        for pos, rng in cfg[section].items():
            n = count(rng)
            pct = 100 * n / 1326
            print(f'  {section}/{pos:<10} {len(rng):>3} types, {n:>4} combos, {pct:>5.2f}%')


def main():
    with open(SRC) as f:
        base = json.load(f)

    configs = {
        '2p': build_2p(base),
        '3p': build_3p(base),
        '4p': build_4p(base),
        '5p': build_5p(base),
    }

    for suffix, cfg in configs.items():
        out_path = CONFIGS / f'preflop_ranges_{suffix}.json'
        with open(out_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f'Wrote {out_path.name}')

    # Summary: main RFI positions only
    print('\n' + '=' * 60)
    print(' RFI width by table size (position-specific)')
    print('=' * 60)

    def rfi_pct(cfg, pos):
        rng = cfg['rfi'].get(pos, {})
        total = 0
        for h in rng:
            if len(h) == 2: total += 6
            elif h.endswith('S'): total += 4
            elif h.endswith('O'): total += 12
        return 100 * total / 1326

    print(f'{"Position":<6}', end='')
    for t in ['6p (base)', '5p', '4p', '3p', '2p']:
        print(f'  {t:<10}', end='')
    print()

    for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB']:
        print(f'{pos:<6}', end='')
        for cfg in [base, configs['5p'], configs['4p'], configs['3p'], configs['2p']]:
            pct = rfi_pct(cfg, pos)
            print(f'  {pct:>6.1f}%   ', end='')
        print()


if __name__ == '__main__':
    main()
