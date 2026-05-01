"""Validate the preflop lookup against held-out HH and a canonical spot list.

Three modes:
  --holdout       Replay every cohort decision in `data/preflop_lookup_holdout.txt`
                  and report agreement %, sizing MAE, coverage, fallback rate.
  --spots         Print canonical preflop spots side-by-side: lookup vs static
                  ranges. Shows action distribution + raise size for each.
  --self-play     Stub that prints the command to run self-play with both
                  backends. (Self-play already piggybacks via DecisionConfig.)

Default: run --spots (fast).

Usage:
    python3 scripts/validate_preflop_lookup.py --spots
    python3 scripts/validate_preflop_lookup.py --holdout
    python3 scripts/validate_preflop_lookup.py --holdout --max-files 100
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hand_history.parser import parse_file
from src.preflop_lookup import (
    PreflopLookup, FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN,
    RAISE_CUSTOM,
)
from src.preflop_ranges import PreflopRanges
from src.preflop_spot_key import (
    BB_USD, build_spot_key, cards_to_class, position_label,
    HIST_OPEN, HIST_VS_RAISE, HIST_VS_3BET, HIST_LIMPED_POT,
)


# ─── Spot-collection diff ────────────────────────────────────────────────

CANONICAL_SPOTS = [
    # (label, hole_cards, rel_pos, history, num_to_act, eff_stack, pot_bb, call_bb)
    ('AA   BTN open',  ['As','Ah'], 0, HIST_OPEN,     2, 100.0, 1.5, 1.0),
    ('AA   LJ  open',  ['As','Ah'], 3, HIST_OPEN,     4, 100.0, 1.5, 1.0),
    ('KK   CO  open',  ['Ks','Kh'], 5, HIST_OPEN,     3, 100.0, 1.5, 1.0),
    ('AKS  BTN open',  ['As','Ks'], 0, HIST_OPEN,     2, 100.0, 1.5, 1.0),
    ('AKO  LJ  open',  ['As','Kc'], 3, HIST_OPEN,     4, 100.0, 1.5, 1.0),
    ('72O  UTG open',  ['7d','2c'], 3, HIST_OPEN,     4, 100.0, 1.5, 1.0),
    ('72O  BTN open',  ['7d','2c'], 0, HIST_OPEN,     2, 100.0, 1.5, 1.0),
    ('JJ   HJ  open',  ['Jc','Jd'], 4, HIST_OPEN,     4, 100.0, 1.5, 1.0),
    ('AKS  CO vs raise',['As','Ks'], 5, HIST_VS_RAISE, 3, 100.0, 4.0, 2.5),
    ('AA   SB vs raise',['As','Ah'], 1, HIST_VS_RAISE, 1, 100.0, 4.0, 2.5),
    ('AA   BB vs raise',['As','Ah'], 2, HIST_VS_RAISE, 0, 100.0, 4.0, 2.5),
    ('AKO  BB vs raise',['As','Kc'], 2, HIST_VS_RAISE, 0, 100.0, 4.0, 2.5),
    ('AJO  BB vs raise',['As','Jc'], 2, HIST_VS_RAISE, 0, 100.0, 4.0, 2.5),
    ('KQS  BB vs raise',['Ks','Qs'], 2, HIST_VS_RAISE, 0, 100.0, 4.0, 2.5),
    ('22   BB vs raise',['2c','2d'], 2, HIST_VS_RAISE, 0, 100.0, 4.0, 2.5),
    ('AKS  SB vs 3bet', ['As','Ks'], 1, HIST_VS_3BET,  0, 100.0, 12.0, 9.0),
    ('AA   LJ vs 3bet', ['As','Ah'], 3, HIST_VS_3BET,  0, 100.0, 12.0, 9.0),
]


def cmd_spots(args):
    fb = PreflopRanges(num_players=6)
    lookup = PreflopLookup(args.db_path, fallback=fb)
    print(f'Loaded {lookup.num_spots} spots from {args.db_path}')
    print(f'{"Spot":<22} {"Source":<8} {"Strategy [F C Rh Rp AI Rm]":<32} '
          f'{"size":>6}')
    print('-' * 75)
    legal = [FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN, RAISE_CUSTOM]

    for (label, hole, rel, hist, nta, eff, pot, call_amt) in CANONICAL_SPOTS:
        # Lookup
        strat, size = lookup.get_strategy(
            hole_cards=hole, rel_pos=rel, pot_bb=pot,
            legal_actions=legal, action_size=6,
            num_players=6, action_history=hist, num_to_act=nta,
            effective_stack=eff, call_amount_bb=call_amt,
        )
        # Source classification:
        #   exact    — L0 hit, custom size emitted
        #   coarse   — L1/L2 hit, action data only (size via preset)
        #   default  — no data, fold default fired
        if size is not None:
            src = 'exact'
        elif strat[FOLD] >= 0.99 and 100.0 * strat[FOLD] / max(strat.sum(), 1e-9) >= 99:
            src = 'default'
        else:
            src = 'coarse'
        s = '[' + ' '.join(f'{x:.2f}' for x in strat) + ']'
        sz = f'{size:.1f}bb' if size is not None else '-'
        print(f'{label:<22} {src:<8} {s:<32} {sz:>6}')

        # Static ranges (for comparison)
        legacy = fb.get_strategy(
            hole_cards=hole, rel_pos=rel, pot_bb=pot,
            legal_actions=[a for a in legal if a < 6],
            action_size=6, num_limpers=0,
            effective_stack=eff, num_callers=0,
        )
        s_legacy = '[' + ' '.join(f'{float(x):.2f}' for x in legacy) + ']'
        print(f'{"   (vs static)":<22} {"static":<8} {s_legacy:<32} {"-":>6}')


# ─── Holdout replay ──────────────────────────────────────────────────────

def cmd_holdout(args):
    fb = PreflopRanges(num_players=6)
    lookup = PreflopLookup(args.db_path, fallback=fb)
    print(f'Loaded {lookup.num_spots} spots from {args.db_path}')

    holdout_path = Path(args.holdout)
    if not holdout_path.exists():
        sys.exit(f'Holdout file not found: {holdout_path}. '
                 f'Re-run scripts/build_preflop_lookup.py to generate it.')

    cohort_path = Path(args.good_players)
    if not cohort_path.exists():
        sys.exit(f'Cohort file not found: {cohort_path}.')
    with open(cohort_path) as f:
        cohort = {line.strip() for line in f if line.strip()}

    files = [Path(line.strip()) for line in holdout_path.read_text().splitlines()
             if line.strip()]
    if args.max_files > 0:
        files = files[:args.max_files]
    print(f'Replaying {len(files)} holdout files against {len(cohort)} cohort players')

    total_decisions = 0
    action_match = 0
    size_errors: List[float] = []
    fallback_count = 0
    spot_coverage = 0

    legal = [FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN, RAISE_CUSTOM]

    for path in files:
        try:
            hands = parse_file(str(path))
        except Exception:
            continue
        for h in hands:
            if abs(h.big_blind_dollars - BB_USD) > 0.001:
                continue
            if h.num_active_players != 6:
                continue
            for row in _walk_hand_for_eval(h, cohort, lookup, legal):
                total_decisions += 1
                actual_action, actual_size, predicted_action, predicted_size, was_fallback = row
                if was_fallback:
                    fallback_count += 1
                else:
                    spot_coverage += 1
                if actual_action == predicted_action:
                    action_match += 1
                if (actual_action == 'raise' and predicted_action == 'raise'
                        and actual_size is not None and predicted_size is not None):
                    size_errors.append(abs(actual_size - predicted_size))

    if total_decisions == 0:
        print('No decisions captured. Check holdout file paths and cohort.')
        return

    agreement = 100.0 * action_match / total_decisions
    fallback_rate = 100.0 * fallback_count / total_decisions
    coverage_rate = 100.0 * spot_coverage / total_decisions
    size_mae = (sum(size_errors) / len(size_errors)) if size_errors else float('nan')

    print()
    print(f'Total decisions:    {total_decisions}')
    print(f'Action agreement:   {agreement:.1f}%')
    print(f'Coverage (data):    {coverage_rate:.1f}%')
    print(f'Fallback rate:      {fallback_rate:.1f}%')
    print(f'Raise size MAE:     {size_mae:.2f} BB  (n={len(size_errors)})')


def _walk_hand_for_eval(hand, cohort, lookup, legal):
    """Walk a hand's preflop and yield (actual_action, actual_size, predicted_action,
    predicted_size, was_fallback) for each cohort decision with known hole cards.

    Mirrors the build script's hand walker but evaluates the LOOKUP against
    the actual decision instead of recording it. Decisions without known hole
    cards (showdown limit) are skipped.
    """
    active = list(hand.active_players)
    if len(active) != 6:
        return
    button_idx = next((i for i, p in enumerate(active)
                       if p.seat == hand.button_seat), 0)
    name_to_idx = {p.name: i for i, p in enumerate(active)}
    starting_stack = {p.name: p.stack_dollars for p in active}
    committed = {p.name: 0.0 for p in active}
    folded = set()
    acted = set()
    voluntary_history = []

    for a in hand.actions:
        if a.street != 'preflop':
            break
        if a.is_blind:
            committed[a.player_name] = committed.get(a.player_name, 0.0) + a.amount_dollars
            continue
        actor = a.player_name
        if actor not in name_to_idx or actor in folded:
            continue
        # Map action_type to label
        if a.action_type == 'fold':
            label = 'fold'
        elif a.action_type == 'raise':
            label = 'raise'
        elif a.action_type in ('call', 'check'):
            label = 'call'
        else:
            continue

        if actor in cohort:
            hc = hand.shown_hands.get(actor)
            if hc and len(hc) >= 2:
                actor_idx = name_to_idx[actor]
                rel_pos = (actor_idx - button_idx) % 6
                remaining = starting_stack[actor] - committed[actor]
                eff_bb = remaining / BB_USD
                num_to_act = (
                    len(active) - len(acted) - 1
                    if actor not in acted else max(0, len(active) - len(acted))
                )
                pot_bb = sum(committed.values()) / BB_USD

                # Predict via lookup
                strat, predicted_size = lookup.get_strategy(
                    hole_cards=hc, rel_pos=rel_pos, pot_bb=pot_bb,
                    legal_actions=legal, action_size=6,
                    num_players=6, action_history=_history_from(voluntary_history),
                    num_to_act=num_to_act, effective_stack=eff_bb,
                    call_amount_bb=max(0.0, max((c for c in committed.values() if c > 0), default=0.0) / BB_USD),
                )
                # Map predicted strategy → action category
                pred_idx = int(strat.argmax())
                predicted_action = (
                    'fold' if pred_idx == FOLD else
                    'call' if pred_idx == CALL else 'raise'
                )
                # was_fallback if size is None AND raise mass > 0 (lookup
                # returns size only on data hits with raise component)
                was_fallback = (predicted_size is None
                                and strat[FOLD] != 1.0
                                and strat[CALL] != 1.0)

                actual_size = (a.amount_dollars / BB_USD
                               if label == 'raise' else None)
                yield (label, actual_size, predicted_action, predicted_size, was_fallback)

        # Apply action
        voluntary_history.append((label, a.amount_dollars / BB_USD if a.amount_dollars > 0 else 0.0))
        acted.add(actor)
        if label == 'fold':
            folded.add(actor)
        elif a.action_type == 'raise':
            committed[actor] = a.amount_dollars
        elif a.action_type in ('call', 'check'):
            committed[actor] = committed.get(actor, 0.0) + a.amount_dollars


def _history_from(voluntary):
    """Re-derive the action_history string the lookup would key on."""
    from src.preflop_spot_key import derive_action_history
    return derive_action_history(voluntary)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest='cmd')

    sp = sub.add_parser('spots', help='Print canonical spots vs static')
    sp.add_argument('--db-path', default='data/preflop_lookup.sqlite')
    sp.set_defaults(func=cmd_spots)

    sh = sub.add_parser('holdout', help='Replay holdout HH against the lookup')
    sh.add_argument('--db-path', default='data/preflop_lookup.sqlite')
    sh.add_argument('--holdout', default='data/preflop_lookup_holdout.txt')
    sh.add_argument('--good-players', default='data/good_players_winrate10.txt')
    sh.add_argument('--max-files', type=int, default=0)
    sh.set_defaults(func=cmd_holdout)

    # Default → spots
    args = ap.parse_args()
    if args.cmd is None:
        ap.parse_args(['spots'] + sys.argv[1:]).func(
            ap.parse_args(['spots'] + sys.argv[1:]))
        return
    args.func(args)


if __name__ == '__main__':
    main()
