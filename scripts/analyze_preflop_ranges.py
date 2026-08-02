"""Empirical preflop range analysis from cohort lookup DB.

For each position (LJ/HJ/CO/BTN/SB), computes the EMPIRICAL opening range
('open' action_history) by mirroring the runtime play-rate gate:

    play_rate(hand) = (n_samples / combos) / (AA_total / 6)

Hands with play_rate >= position threshold are "in range". Renders a 13x13
hand-chart grid per position and reports VPIP/RFI% alongside the matching
static range from configs/preflop_ranges_nl5.json.

Usage:
    python3 scripts/analyze_preflop_ranges.py
    python3 scripts/analyze_preflop_ranges.py --threshold-sweep
    python3 scripts/analyze_preflop_ranges.py --position BTN --threshold 0.30
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / 'data' / 'preflop_lookup.sqlite'
STATIC_RANGES_PATH = PROJECT_ROOT / 'configs' / 'preflop_ranges_nl5.json'

RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
RFI_POSITIONS = ('LJ', 'HJ', 'CO', 'BTN', 'SB')  # BB never RFIs (already in pot)


def combos_for_class(hole_class: str) -> int:
    """6 combos for pairs, 4 suited, 12 offsuit (per-class total in a deck)."""
    if len(hole_class) == 2:
        return 6  # pair
    suffix = hole_class[2].upper()
    return 4 if suffix == 'S' else 12


def class_for_grid(row: int, col: int) -> str:
    """Map 13x13 grid cell -> hole_class label (DB casing: 'AKS', 'AKO')."""
    r1, r2 = RANKS[row], RANKS[col]
    if row == col:
        return r1 + r2  # pair
    if row < col:
        return r1 + r2 + 'S'  # suited (top-right triangle)
    return r2 + r1 + 'O'  # offsuit (bottom-left)


def fetch_aa_per_combo(conn: sqlite3.Connection, action_history: str) -> Dict[str, float]:
    """{position: AA_per_combo} aggregated across stack buckets."""
    out: Dict[str, float] = {}
    for pos, total in conn.execute(
        "SELECT position, SUM(n_samples) FROM lookup "
        "WHERE hole_class='AA' AND action_history=? GROUP BY position",
        (action_history,),
    ):
        out[pos] = total / 6.0
    return out


def fetch_hand_counts(conn: sqlite3.Connection, position: str,
                      action_history: str) -> Dict[str, int]:
    """{hole_class: total n_samples} aggregated across stack/num_to_act."""
    out: Dict[str, int] = {}
    for hc, total in conn.execute(
        "SELECT hole_class, SUM(n_samples) FROM lookup "
        "WHERE position=? AND action_history=? GROUP BY hole_class",
        (position, action_history),
    ):
        out[hc] = total
    return out


def compute_play_rates(hand_counts: Dict[str, int],
                        aa_per_combo: float) -> Dict[str, float]:
    """{hole_class: play_rate (unclamped)}. Empty/zero -> 0.0."""
    if aa_per_combo <= 0:
        return {}
    rates: Dict[str, float] = {}
    for row in range(13):
        for col in range(13):
            hc = class_for_grid(row, col)
            n = hand_counts.get(hc, 0)
            rates[hc] = (n / combos_for_class(hc)) / aa_per_combo
    return rates


def load_static_rfi() -> Dict[str, Dict[str, float]]:
    """{position: {hole_class_upper: frequency_in_[0,1]}} from static range JSON."""
    raw = json.loads(STATIC_RANGES_PATH.read_text())
    rfi = raw.get('rfi', {})
    out: Dict[str, Dict[str, float]] = {}
    for pos, mapping in rfi.items():
        normalized: Dict[str, float] = {}
        for hc, val in mapping.items():
            key = hc.upper()  # 'AKs' -> 'AKS'
            sval = str(val).lstrip(':')
            try:
                normalized[key] = float(sval)
            except ValueError:
                normalized[key] = 0.0
        out[pos] = normalized
    return out


def load_thresholds() -> Tuple[Dict[str, float], float]:
    """Returns ({position: threshold}, global_default) from DecisionConfig."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.decision_engine import DecisionConfig
    dc = DecisionConfig()
    return dict(dc.preflop_pos_min_play_rate), float(dc.preflop_min_play_rate)


def render_grid(play_rates: Dict[str, float], threshold: float,
                static_range: Optional[Dict[str, float]] = None) -> str:
    """13x13 grid. Each cell shows play_rate as integer percent.
       '##' = in current range (play_rate >= threshold).
       '..' = in static range but not in cohort range (gap).
       '  ' = neither."""
    lines = []
    header = '       ' + '  '.join(f'{r:>3}' for r in RANKS)
    lines.append(header)
    for row in range(13):
        cells = []
        for col in range(13):
            hc = class_for_grid(row, col)
            pr = play_rates.get(hc, 0.0)
            in_cohort = pr >= threshold
            in_static = (static_range or {}).get(hc, 0.0) > 0.05
            pct = min(int(round(pr * 100)), 99)
            if in_cohort:
                marker = '##'
            elif in_static:
                marker = '..'  # static plays it but cohort doesn't
            else:
                marker = '  '
            cells.append(f'{pct:>2}{marker}')
        lines.append(f' {RANKS[row]}  ' + ' '.join(cells))
    return '\n'.join(lines)


def render_diff_grid(play_rates: Dict[str, float],
                      old_thr: float, new_thr: float) -> str:
    """13x13 grid showing OLD vs NEW threshold gates.
       '##' = in both (kept), '++' = added by new only,
       '--' = removed by new only, '. ' = in neither."""
    lines = []
    header = '       ' + '  '.join(f'{r:>3}' for r in RANKS)
    lines.append(header)
    for row in range(13):
        cells = []
        for col in range(13):
            hc = class_for_grid(row, col)
            pr = play_rates.get(hc, 0.0)
            in_old = pr >= old_thr
            in_new = pr >= new_thr
            if in_old and in_new:
                marker = '##'
            elif in_new and not in_old:
                marker = '++'
            elif in_old and not in_new:
                marker = '--'
            else:
                marker = '. '
            cells.append(marker)
        lines.append(f' {RANKS[row]}   ' + '   '.join(cells))
    return '\n'.join(lines)


# Old thresholds before the 2026-05-19 retuning — used for diff mode.
OLD_THRESHOLDS = {
    'BTN': 0.30, 'SB': 0.32, 'BB': 0.22, 'LJ': 0.35, 'HJ': 0.32, 'CO': 0.28,
}


def count_in_range(play_rates: Dict[str, float], threshold: float) -> Tuple[int, float]:
    """Returns (n_classes_in_range, VPIP% by combos)."""
    classes_in = 0
    combos_in = 0
    for row in range(13):
        for col in range(13):
            hc = class_for_grid(row, col)
            if play_rates.get(hc, 0.0) >= threshold:
                classes_in += 1
                combos_in += combos_for_class(hc)
    return classes_in, combos_in / 1326.0 * 100.0  # 1326 = total combos in deck


def count_static(static_range: Dict[str, float]) -> Tuple[int, float]:
    """Same metric but for the static JSON range (freq > 0.5 counted as in)."""
    classes_in = 0
    combos_in = 0
    for hc, freq in static_range.items():
        if freq >= 0.5:
            classes_in += 1
            combos_in += combos_for_class(hc)
    return classes_in, combos_in / 1326.0 * 100.0


def analyze_position(conn: sqlite3.Connection, position: str, threshold: float,
                     static_rfi: Dict[str, Dict[str, float]],
                     aa_per_combo: Dict[str, float]) -> None:
    aa = aa_per_combo.get(position)
    if not aa:
        print(f'  [no AA reference for {position}]')
        return
    hand_counts = fetch_hand_counts(conn, position, 'open')
    play_rates = compute_play_rates(hand_counts, aa)
    classes, vpip = count_in_range(play_rates, threshold)

    static_key = position  # try exact match first
    static = static_rfi.get(static_key)
    if static is None:
        # try 'BTN_o' style if needed
        static = {}
    s_classes, s_vpip = count_static(static)

    print(f'\n══ {position}  (threshold {threshold:.2f},  AA_per_combo={aa:.1f}) ══')
    print(f'  Cohort range: {classes:>3} classes  /  {vpip:5.1f}% combo-weighted RFI')
    print(f'  Static range: {s_classes:>3} classes  /  {s_vpip:5.1f}% combo-weighted RFI')
    print(f'  Δ (cohort − static): {vpip - s_vpip:+5.1f}%\n')
    print(render_grid(play_rates, threshold, static))
    print('\n  Legend:  ##=in cohort range   ..=static only (cohort folds)   blank=both fold')
    print('  Cell number = play_rate × 100 (empirical play frequency vs AA baseline)')


def threshold_sweep(conn: sqlite3.Connection,
                     aa_per_combo: Dict[str, float]) -> None:
    """Show RFI% at a range of thresholds for each position."""
    print('\n══ Threshold sweep — RFI% by position ══')
    header = f'  {"pos":<4} ' + ' '.join(f'{t:>5.2f}' for t in
                                          [0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40, 0.50])
    print(header)
    for pos in RFI_POSITIONS:
        aa = aa_per_combo.get(pos)
        if not aa:
            continue
        hand_counts = fetch_hand_counts(conn, pos, 'open')
        play_rates = compute_play_rates(hand_counts, aa)
        row = f'  {pos:<4} '
        for thr in [0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40, 0.50]:
            _, vpip = count_in_range(play_rates, thr)
            row += f' {vpip:4.1f} '
        print(row)


def main() -> int:
    ap = argparse.ArgumentParser(description='Empirical preflop range analysis')
    ap.add_argument('--position', default=None,
                    help='Analyze single position only (LJ/HJ/CO/BTN/SB)')
    ap.add_argument('--threshold', type=float, default=None,
                    help='Override threshold (default: from DecisionConfig)')
    ap.add_argument('--threshold-sweep', action='store_true',
                    help='Show RFI%% at multiple thresholds for each position')
    ap.add_argument('--diff', action='store_true',
                    help='Diff current thresholds vs old defaults (pre-2026-05-19)')
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f'ERROR: lookup DB not found at {DB_PATH}', file=sys.stderr)
        return 1

    pos_thr, global_thr = load_thresholds()
    static_rfi = load_static_rfi()
    conn = sqlite3.connect(str(DB_PATH))
    aa_per_combo = fetch_aa_per_combo(conn, 'open')

    print(f'DB: {DB_PATH}')
    print(f'Static ranges: {STATIC_RANGES_PATH}')
    print(f'Thresholds from: {AGENT_CONFIG_PATH}')
    print(f'  global preflop_min_play_rate = {global_thr}')
    print(f'  per-position = {pos_thr}')

    positions = [args.position] if args.position else list(RFI_POSITIONS)
    for pos in positions:
        thr = args.threshold if args.threshold is not None \
            else pos_thr.get(pos, global_thr)
        analyze_position(conn, pos, thr, static_rfi, aa_per_combo)

    if args.threshold_sweep:
        threshold_sweep(conn, aa_per_combo)

    if args.diff:
        print('\n══════════════════════════════════════════════════════════')
        print('  DIFF: old thresholds (pre-2026-05-19) vs current')
        print('══════════════════════════════════════════════════════════')
        for pos in positions:
            aa = aa_per_combo.get(pos)
            if not aa:
                continue
            old_thr = OLD_THRESHOLDS.get(pos, 0.25)
            new_thr = args.threshold if args.threshold is not None \
                else pos_thr.get(pos, global_thr)
            hand_counts = fetch_hand_counts(conn, pos, 'open')
            play_rates = compute_play_rates(hand_counts, aa)
            _, old_vpip = count_in_range(play_rates, old_thr)
            _, new_vpip = count_in_range(play_rates, new_thr)
            print(f'\n── {pos}  old thr={old_thr:.2f} ({old_vpip:.1f}% RFI)  →  '
                   f'new thr={new_thr:.2f} ({new_vpip:.1f}% RFI)  '
                   f'Δ={new_vpip - old_vpip:+.1f}%')
            print(render_diff_grid(play_rates, old_thr, new_thr))
            added = [class_for_grid(r, c) for r in range(13) for c in range(13)
                     if play_rates.get(class_for_grid(r, c), 0) >= new_thr
                     and play_rates.get(class_for_grid(r, c), 0) < old_thr]
            removed = [class_for_grid(r, c) for r in range(13) for c in range(13)
                       if play_rates.get(class_for_grid(r, c), 0) >= old_thr
                       and play_rates.get(class_for_grid(r, c), 0) < new_thr]
            if added:
                print(f'  ++ ADDED ({len(added)}): {" ".join(added)}')
            if removed:
                print(f'  -- REMOVED ({len(removed)}): {" ".join(removed)}')
        print('\n  Legend:  ##=kept   ++=added   --=removed   . =fold in both')

    conn.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
