"""Per-category preflop audit — 13×13 grids per (position, action_history).

Synthesizes a SpotContext for every (hole_class × position × action_history)
combination at a fixed stack depth (100bb) and queries PreflopLookup directly.
Renders an ASCII 13×13 grid per (pos, hist) showing the dominant action, and
prints a per-category source histogram (L0/L1/L2/static_action_cohort_size/
L2_static_fallback/L3_static_fallback/L3_fold_default).

Use this script to:
  - Verify the 3bet+ refactor: every vs_3bet/vs_4bet+ cell should be sourced
    from `static_action_cohort_size` (Phase 1 invariant).
  - Inspect RFI / vs_RFI ranges before & after cohort knob tightening.
  - Catch range-construction bugs (e.g., all-fold or all-raise grids).
  - Compare actual play% per (pos, hist) against GTO bands.

Output is deterministic and fast (~5 seconds for the full sweep).

Usage:
    python3 scripts/analysis/preflop_category_audit.py
    python3 scripts/analysis/preflop_category_audit.py --positions BTN CO
    python3 scripts/analysis/preflop_category_audit.py --histories vs_3bet vs_4bet+
    python3 scripts/analysis/preflop_category_audit.py --json out.json
    python3 scripts/analysis/preflop_category_audit.py \\
        --db data/preflop_lookup.sqlite \\
        --ranges configs/preflop_ranges.json \\
        --num-players 6 --stack-bb 100
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _PROJECT_ROOT)

from src.preflop_lookup import (
    PreflopLookup, FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN,
)
from src.preflop_ranges import PreflopRanges
from src.preflop_spot_key import (
    HIST_OPEN, HIST_LIMPED_POT, HIST_VS_RAISE, HIST_VS_3BET, HIST_VS_4BET,
)


_RANKS = list('AKQJT98765432')   # high → low
_HIST_ORDER = [HIST_OPEN, HIST_LIMPED_POT, HIST_VS_RAISE,
                HIST_VS_3BET, HIST_VS_4BET]
_POSITIONS_6P = ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
_POS_TO_REL = {'BTN': 0, 'SB': 1, 'BB': 2, 'LJ': 3, 'HJ': 4, 'CO': 5}

# Reasonable default pot/call values per spot category — used to feed the
# lookup (it doesn't strictly need exact values to compute strategy, but
# call_amount_bb feeds back into preset selection for raise sizing).
_HIST_CONTEXT = {
    HIST_OPEN:        {'pot_bb': 1.5,  'call_bb': 1.0},   # SB+BB, facing BB
    HIST_LIMPED_POT:  {'pot_bb': 2.5,  'call_bb': 0.5},   # one limper before us
    HIST_VS_RAISE:    {'pot_bb': 4.0,  'call_bb': 2.5},   # facing 2.5bb open
    HIST_VS_3BET:     {'pot_bb': 12.0, 'call_bb': 8.0},   # facing 9bb 3bet
    HIST_VS_4BET:     {'pot_bb': 33.0, 'call_bb': 22.0},  # facing 22bb 4bet
}


# ─── Hole-card combo construction ────────────────────────────────────────

def _hole_for(combo: str) -> List[str]:
    """Build a representative two-card hand for a 169-class combo string.

    For 'AA' returns ['As', 'Ah'] (any pair representative).
    For 'AKS' returns ['As', 'Ks'] (suited).
    For 'AKO' returns ['As', 'Kh'] (offsuit).
    """
    if len(combo) == 2:  # pair, e.g. 'AA'
        r = combo[0]
        return [f'{r}s', f'{r}h']
    high, low, suited = combo[0], combo[1], combo[2].upper()
    if suited == 'S':
        return [f'{high}s', f'{low}s']
    return [f'{high}s', f'{low}h']


def _combo_for(rank_high: str, rank_low: str, suited: bool) -> str:
    """Build canonical combo string from two ranks."""
    if rank_high == rank_low:
        return rank_high + rank_high
    return f'{rank_high}{rank_low}{"S" if suited else "O"}'


def _build_169_grid_combos() -> List[List[str]]:
    """13×13 list-of-lists: row=high rank (top is A), col=low rank (left is A).
    Suited above/at diagonal, offsuit below diagonal, pairs on diagonal.
    """
    grid: List[List[str]] = []
    for hi in _RANKS:
        row = []
        for lo in _RANKS:
            i_hi = _RANKS.index(hi)
            i_lo = _RANKS.index(lo)
            if hi == lo:
                row.append(hi + hi)
            elif i_hi < i_lo:  # row rank > col rank (A row, K col → AKs above diag)
                row.append(_combo_for(hi, lo, suited=True))
            else:
                # below diagonal — col rank > row rank → swap to high+low offsuit
                row.append(_combo_for(lo, hi, suited=False))
        grid.append(row)
    return grid


# ─── Decision classification ─────────────────────────────────────────────

def _dominant_action(strategy: np.ndarray) -> Tuple[str, float]:
    """Return (label, prob) for the dominant action in the strategy vector.

    Labels: 'F' (fold), 'C' (call), 'R' (raise), 'M' (mixed).
    Mixed = no single action ≥ 0.7.
    """
    fold_p = float(strategy[FOLD]) if FOLD < len(strategy) else 0.0
    call_p = float(strategy[CALL]) if CALL < len(strategy) else 0.0
    raise_p = float(sum(strategy[a] for a in (RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN)
                         if a < len(strategy)))
    if fold_p >= 0.7:
        return ('F', fold_p)
    if call_p >= 0.7:
        return ('C', call_p)
    if raise_p >= 0.7:
        return ('R', raise_p)
    return ('M', max(fold_p, call_p, raise_p))


def _which_level_fired(before: Dict[str, int], after: Dict[str, int]) -> str:
    """Diff counters; return the level name whose count incremented."""
    for level in after:
        if after[level] - before.get(level, 0) > 0:
            return level
    return 'unknown'


# ─── Audit driver ────────────────────────────────────────────────────────

def audit_spot(lookup: PreflopLookup, position: str, action_history: str,
               combo: str, num_players: int, stack_bb: float
               ) -> Dict[str, object]:
    """Query the lookup for one spot and return a result dict."""
    rel_pos = _POS_TO_REL[position]
    hole = _hole_for(combo)
    ctx = _HIST_CONTEXT[action_history]

    # All actions legal so the strategy vector can express anything
    legal = [FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN]

    before = dict(lookup.counters)
    strategy, custom_size = lookup.get_strategy(
        hole_cards=hole, rel_pos=rel_pos,
        pot_bb=ctx['pot_bb'], legal_actions=legal, action_size=6,
        num_players=num_players, action_history=action_history,
        num_to_act=2, effective_stack=stack_bb,
        call_amount_bb=ctx['call_bb'],
    )
    after = dict(lookup.counters)
    level = _which_level_fired(before, after)
    dom_label, dom_prob = _dominant_action(strategy)

    return {
        'combo': combo,
        'position': position,
        'action_history': action_history,
        'strategy': strategy.tolist(),
        'custom_size_bb': custom_size,
        'level': level,
        'dominant': dom_label,
        'dominant_prob': dom_prob,
    }


def render_grid(results_for_spot: List[Dict[str, object]], grid: List[List[str]]
                 ) -> str:
    """Render a 13×13 ASCII grid of dominant actions for one (pos, hist)."""
    by_combo = {r['combo']: r for r in results_for_spot}
    lines = []
    # Header row
    header = '     ' + ' '.join(f'{r:>3}' for r in _RANKS)
    lines.append(header)
    for hi in _RANKS:
        cells = []
        for lo in _RANKS:
            combo_idx = _RANKS.index(hi)
            lo_idx = _RANKS.index(lo)
            if combo_idx <= lo_idx:
                # Above diag or on diag
                if hi == lo:
                    combo = hi + hi
                else:
                    combo = _combo_for(hi, lo, suited=True)
            else:
                combo = _combo_for(lo, hi, suited=False)
            r = by_combo.get(combo)
            cells.append(r['dominant'] if r else '?')
        lines.append(f'  {hi}  ' + '   '.join(cells))
    return '\n'.join(lines)


def play_pct(results_for_spot: List[Dict[str, object]]) -> Dict[str, float]:
    """Compute % combos taking each action over the 169-class grid.

    We weight every combo equally (1/169), NOT by combinatorial multiplicity.
    Returns {fold, call, raise, mixed} percentages.
    """
    counts = Counter(r['dominant'] for r in results_for_spot)
    n = max(len(results_for_spot), 1)
    return {
        'fold':  100.0 * counts.get('F', 0) / n,
        'call':  100.0 * counts.get('C', 0) / n,
        'raise': 100.0 * counts.get('R', 0) / n,
        'mixed': 100.0 * counts.get('M', 0) / n,
    }


def source_histogram(results_for_spot: List[Dict[str, object]]) -> Counter:
    return Counter(r['level'] for r in results_for_spot)


# ─── CLI ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', default='data/preflop_lookup.sqlite',
                    help='Preflop lookup DB path')
    ap.add_argument('--ranges', default=None,
                    help='Static range JSON path (default: configs/preflop_ranges.json '
                         'for 6-max). Used as PreflopLookup fallback AND for the '
                         'static-action 3bet+ branch.')
    ap.add_argument('--num-players', type=int, default=6,
                    help='Table size (default: 6)')
    ap.add_argument('--stack-bb', type=float, default=100.0,
                    help='Effective stack in BB (default: 100)')
    ap.add_argument('--positions', nargs='+', default=None,
                    help='Restrict to these positions (default: all)')
    ap.add_argument('--histories', nargs='+', default=None,
                    help='Restrict to these histories (default: all)')
    ap.add_argument('--json', default=None,
                    help='Also write per-spot results to this JSON file')
    ap.add_argument('--no-grids', action='store_true',
                    help='Skip ASCII grid output (just summaries + JSON)')
    args = ap.parse_args()

    # Load fallback (static ranges) — required for the 3bet+ branch and
    # for L3 fallback elsewhere.
    ranges = PreflopRanges(num_players=args.num_players)
    lookup = PreflopLookup(args.db, fallback=ranges)

    positions = args.positions or _POSITIONS_6P
    histories = args.histories or _HIST_ORDER
    grid_combos = _build_169_grid_combos()
    flat_combos = sorted({c for row in grid_combos for c in row})

    print(f'\nPreflop category audit: {len(positions)} pos × '
          f'{len(histories)} hist × {len(flat_combos)} combos = '
          f'{len(positions) * len(histories) * len(flat_combos)} spots')
    print(f'  DB:          {args.db}')
    print(f'  Stack:       {args.stack_bb}bb')
    print(f'  num_players: {args.num_players}\n')

    all_results: Dict[str, List[Dict[str, object]]] = {}

    for pos in positions:
        for hist in histories:
            spot_key = f'{pos}_{hist}'
            results = []
            for combo in flat_combos:
                r = audit_spot(lookup, pos, hist, combo,
                                args.num_players, args.stack_bb)
                results.append(r)
            all_results[spot_key] = results

    # ── Output ────────────────────────────────────────────────────────────
    print('=' * 78)
    print('PER-CATEGORY SUMMARY')
    print('=' * 78)
    print(f'{"pos":<5} {"hist":<12} {"play%":>6} {"raise%":>7} {"call%":>6} '
          f'{"fold%":>6} {"mixed%":>6}  source histogram')
    print('-' * 78)

    for pos in positions:
        for hist in histories:
            results = all_results[f'{pos}_{hist}']
            pct = play_pct(results)
            play = pct['raise'] + pct['call'] + pct['mixed']
            srcs = source_histogram(results)
            src_str = ' '.join(f'{k}={v}' for k, v in
                                sorted(srcs.items(), key=lambda kv: -kv[1])
                                if v > 0)
            print(f'{pos:<5} {hist:<12} {play:>5.1f}% {pct["raise"]:>6.1f}% '
                  f'{pct["call"]:>5.1f}% {pct["fold"]:>5.1f}% '
                  f'{pct["mixed"]:>5.1f}%  {src_str}')

    # ── Grids ─────────────────────────────────────────────────────────────
    if not args.no_grids:
        print('\n' + '=' * 78)
        print('13×13 GRIDS (F=fold, C=call, R=raise, M=mixed)')
        print('=' * 78)
        for pos in positions:
            for hist in histories:
                results = all_results[f'{pos}_{hist}']
                pct = play_pct(results)
                srcs = source_histogram(results)
                top_src = srcs.most_common(1)[0] if srcs else ('?', 0)
                print(f'\n[{pos} / {hist}]  '
                      f'play={pct["raise"]+pct["call"]+pct["mixed"]:.0f}%  '
                      f'raise={pct["raise"]:.0f}%  call={pct["call"]:.0f}%  '
                      f'fold={pct["fold"]:.0f}%  '
                      f'(source: {top_src[0]} dominates {top_src[1]}/169)')
                print(render_grid(results, grid_combos))

    # ── JSON output ───────────────────────────────────────────────────────
    if args.json:
        out = {
            'config': {
                'db': args.db, 'num_players': args.num_players,
                'stack_bb': args.stack_bb, 'positions': positions,
                'histories': histories,
            },
            'spots': {
                k: {
                    'play_pct': play_pct(v),
                    'source_histogram': dict(source_histogram(v)),
                    'cells': v,
                }
                for k, v in all_results.items()
            },
        }
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nWrote per-spot results to {args.json}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
