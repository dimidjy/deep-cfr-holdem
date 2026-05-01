"""Build the preflop lookup table from cohort players' hand histories.

Walks the HH archive (default `data/poker_archive/`), filters to NL5 6-active
hands, and for every preflop decision made by a player in the cohort
(default `data/good_players_winrate10.txt`) records a row in the
`decisions` table of `data/preflop_lookup.sqlite`. After the walk it
rebuilds the aggregated `lookup` table.

Showdown-only data limitation
-----------------------------
PokerStars HH only reveals hole cards when a player reaches showdown.
Therefore we can ONLY record cohort decisions where the cohort player
either reached showdown or showed cards. Implications:

  - Premium hands (AA, AKs, KK, ...) are well-covered: they almost always
    continue to showdown when entered.
  - Marginal hands that fold to flop bets are under-represented (no card
    reveal).
  - Trash hands (folded preflop, never shown) get ZERO coverage → the
    lookup falls back to static ranges for those, which is the correct
    "fold" behavior anyway.

This is the same limitation as the existing obs_builder. Lookup is
"data-driven for showdown-reachable spots, fallback otherwise." Document
the fallback rate in the validation harness.

Resumable: tracks (path, mtime) in `processed_files` and skips already-
ingested HH files unless `--rebuild` (which truncates lookup) or
`--no-cache` (which re-ingests every file) is passed.

Holdout split: `--holdout-frac F --holdout-seed S` deterministically
partitions HH file paths and writes the held-out paths to
`data/preflop_lookup_holdout.txt` for later use by the validation harness.
Holdout files are NOT ingested into the lookup.

Usage:
    python3 scripts/build_preflop_lookup.py
    python3 scripts/build_preflop_lookup.py --rebuild
    python3 scripts/build_preflop_lookup.py --max-files 100  # smoke
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hand_history.parser import parse_file
from src.hand_history.models import ParsedHand, ParsedAction
from src.preflop_spot_key import (
    BB_USD, build_spot_key, SpotKey,
)
from src.preflop_db import (
    connect, insert_decisions_bulk, rebuild_lookup,
    is_file_processed, mark_file_processed, lookup_summary,
)


def _find_button_idx(active_players, button_seat: int) -> int:
    """Mirror obs_builder._find_button_idx for the dead-button edge case."""
    for i, p in enumerate(active_players):
        if p.seat == button_seat:
            return i
    return 0


def _action_label(a: ParsedAction) -> Optional[str]:
    """Map ParsedAction.action_type → spot-key action label, or None to skip."""
    if a.action_type == 'fold':
        return 'fold'
    if a.action_type == 'raise':
        return 'raise'
    if a.action_type in ('call', 'check'):
        # Preflop checks only happen for the BB when nobody raised — same
        # category as a passive call for our purposes (no aggression).
        return 'call'
    return None  # blinds, bets — bets shouldn't appear preflop


def _walk_hand(hand: ParsedHand, cohort: set,
               source_file: str) -> Iterable[Tuple]:
    """Yield decision rows for every cohort player's preflop decision.

    Each row matches the schema of `preflop_db.insert_decisions_bulk`.
    """
    active = list(hand.active_players)
    if len(active) != 6:
        return
    button_idx = _find_button_idx(active, hand.button_seat)
    name_to_idx = {p.name: i for i, p in enumerate(active)}
    starting_stack = {p.name: p.stack_dollars for p in active}
    committed = {p.name: 0.0 for p in active}    # money in this hand
    folded: set = set()
    acted_this_street: set = set()                # voluntary actors so far
    voluntary_history: List[Tuple[str, float]] = []  # (label, size_bb)

    for a in hand.actions:
        if a.street != 'preflop':
            break

        if a.is_blind:
            # Posted blind — counts toward committed but not voluntary
            committed[a.player_name] = committed.get(a.player_name, 0.0) + a.amount_dollars
            continue

        actor = a.player_name
        if actor not in name_to_idx:
            continue  # unknown player (shouldn't happen for active hand)
        if actor in folded:
            continue

        label = _action_label(a)
        if label is None:
            continue  # weird postflop action leaked in; ignore

        # ── Build the row IF this actor is in the cohort ──
        if actor in cohort:
            actor_idx = name_to_idx[actor]
            rel_pos = (actor_idx - button_idx) % 6

            # Effective stack: hero's remaining stack at decision time.
            # Bucket boundaries (40/70/150 BB) capture short/standard/deep.
            remaining = starting_stack[actor] - committed[actor]
            eff_bb = remaining / BB_USD

            # num_to_act = active players who haven't voluntarily acted yet,
            # excluding hero. After hero acts, this drops by 1.
            num_to_act = (
                len(active) - len(acted_this_street) - 1
                if actor not in acted_this_street
                else max(0, len(active) - len(acted_this_street))
            )

            # Pot in BB at decision time. Sum of committed across all players.
            pot_bb = sum(committed.values()) / BB_USD

            spot = build_spot_key(
                hole_cards=['As', 'Ah'],  # placeholder — see note below
                rel_pos=rel_pos,
                num_players=6,
                prior_actions=voluntary_history,
                effective_bb=eff_bb,
                num_to_act=num_to_act,
            )
            # We don't know cohort player's hole cards unless they showed
            # at showdown. For showdown-only we'd lose ~80% of decisions.
            # Fix: derive hole_class from shown_hands when available;
            # otherwise SKIP this decision (lookup needs the card class).
            hc = hand.shown_hands.get(actor)
            if not hc or len(hc) < 2:
                # Skip — we don't know what they were holding.
                pass
            else:
                from src.preflop_spot_key import cards_to_class
                spot = SpotKey(
                    hole_class=cards_to_class(hc[0], hc[1]),
                    position=spot.position,
                    action_history=spot.action_history,
                    stack_bucket=spot.stack_bucket,
                    num_to_act=spot.num_to_act,
                )

                size_bb = None
                if label == 'raise':
                    size_bb = a.amount_dollars / BB_USD  # cumulative "to" / BB
                elif label == 'call':
                    # Total committed by hero through this action
                    size_bb = (committed[actor] + a.amount_dollars) / BB_USD

                yield (
                    spot.hole_class, spot.position, spot.action_history,
                    spot.stack_bucket, spot.num_to_act,
                    actor, label, size_bb,
                    pot_bb, hand.hand_id, source_file,
                )

        # ── Apply the action to running state ──
        voluntary_history.append((
            label,
            a.amount_dollars / BB_USD if a.amount_dollars > 0 else 0.0,
        ))
        acted_this_street.add(actor)

        if label == 'fold':
            folded.add(actor)
        elif a.action_type == 'raise':
            committed[actor] = a.amount_dollars  # raise.amount_dollars is cumulative
        elif a.action_type in ('call', 'check'):
            committed[actor] = committed.get(actor, 0.0) + a.amount_dollars


def _split_holdout(files: List[Path], frac: float, seed: int) -> Tuple[List[Path], List[Path]]:
    """Deterministic file-level holdout split."""
    if frac <= 0:
        return list(files), []
    rng = random.Random(seed)
    files_sorted = sorted(files)
    holdout_size = int(len(files_sorted) * frac)
    holdout_set = set(rng.sample(files_sorted, holdout_size))
    train = [f for f in files_sorted if f not in holdout_set]
    holdout = [f for f in files_sorted if f in holdout_set]
    return train, holdout


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--hh-dir', default='data/poker_archive')
    ap.add_argument('--good-players',
                    default='data/good_players_winrate10.txt')
    ap.add_argument('--db-path', default='data/preflop_lookup.sqlite')
    ap.add_argument('--rebuild', action='store_true',
                    help='Drop+rebuild lookup table from existing decisions; '
                         'do not re-ingest HH files')
    ap.add_argument('--no-cache', action='store_true',
                    help='Re-ingest every HH file even if previously processed')
    ap.add_argument('--holdout-frac', type=float, default=0.2)
    ap.add_argument('--holdout-seed', type=int, default=42)
    ap.add_argument('--holdout-out',
                    default='data/preflop_lookup_holdout.txt')
    ap.add_argument('--max-files', type=int, default=0,
                    help='0 = all (default)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    cohort = set()
    if Path(args.good_players).exists():
        with open(args.good_players) as f:
            cohort = {line.strip() for line in f if line.strip()}
    print(f'Cohort: {len(cohort)} players from {args.good_players}')
    if not cohort and not args.rebuild:
        sys.exit('Cohort empty — run scripts/compute_player_winrates.py first.')

    conn = connect(args.db_path)

    if args.rebuild and not cohort:
        # Pure rebuild: just regenerate lookup from existing decisions
        n = rebuild_lookup(conn)
        s = lookup_summary(conn)
        print(f'Rebuilt lookup: {n} spots from {s["n_decisions"]} decisions')
        return

    files = sorted(Path(args.hh_dir).rglob('*.txt'))
    if args.max_files > 0:
        files = files[:args.max_files]
    train_files, holdout_files = _split_holdout(
        files, args.holdout_frac, args.holdout_seed)

    holdout_path = Path(args.holdout_out)
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    with open(holdout_path, 'w') as hf:
        hf.write('\n'.join(str(p) for p in holdout_files) + '\n')
    print(f'HH files: {len(files)} total → {len(train_files)} train, '
          f'{len(holdout_files)} holdout (seed={args.holdout_seed}); '
          f'holdout list → {holdout_path}')

    t0 = time.time()
    batch: List[Tuple] = []
    BATCH_SIZE = 50_000
    n_inserted = 0
    n_files_done = 0
    n_files_skipped = 0
    for fi, path in enumerate(train_files):
        path_str = str(path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue

        if not args.no_cache and is_file_processed(conn, path_str, mtime):
            n_files_skipped += 1
            continue

        try:
            hands = parse_file(path_str)
        except Exception:
            continue

        hands_in_file = 0
        for h in hands:
            if abs(h.big_blind_dollars - BB_USD) > 0.001:
                continue
            if h.num_active_players != 6:
                continue
            hands_in_file += 1
            for row in _walk_hand(h, cohort, path_str):
                batch.append(row)
                if len(batch) >= BATCH_SIZE:
                    n_inserted += insert_decisions_bulk(conn, batch)
                    batch.clear()

        mark_file_processed(conn, path_str, mtime, hands_in_file)
        n_files_done += 1

        if not args.quiet and (fi + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            eta = (len(train_files) - fi - 1) / rate if rate > 0 else 0
            print(f'  [{fi+1}/{len(train_files)}] '
                  f'{n_inserted} decisions inserted, {n_files_skipped} skipped '
                  f'({elapsed:.0f}s, ~{eta:.0f}s remaining)')

    if batch:
        n_inserted += insert_decisions_bulk(conn, batch)
        batch.clear()

    print(f'Ingestion done in {time.time()-t0:.0f}s. '
          f'{n_files_done} files processed, {n_files_skipped} skipped, '
          f'{n_inserted} decisions inserted this run.')

    print('Rebuilding lookup table...')
    n_spots = rebuild_lookup(conn)
    s = lookup_summary(conn)
    print(f'Lookup: {n_spots} unique spots from '
          f'{s["n_decisions"]} total decisions across '
          f'{s["n_files_processed"]} files.')

    print('\nTop 10 spots by sample count:')
    for hole, pos, hist, sb, nta, n, fp, cp, rp, p50 in s['top_spots_by_samples']:
        p50_str = f'{p50:.1f}' if p50 is not None else '-'
        print(f"  {hole:<4} {pos:<4} {hist:<13} {sb:<7} nta={nta}  "
              f"n={n:>6}  F/C/R={fp}/{cp}/{rp}  p50_size={p50_str}")


if __name__ == '__main__':
    main()
