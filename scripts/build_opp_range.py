"""Build the empirical opponent-range lookup from hand histories.

For each postflop decision by a player who reaches showdown (with known
hole cards), record:
  - the spot key from THAT PLAYER's POV (their position / SPR /
    action_context)
  - their hand class (169-class: 'AA', 'AKs', 'T9o', etc.)

Aggregate into `opp_range_lookup` with per-(spot, hand_class) counts.
Resumable via `opp_range_processed_files`.

Showdown-only data caveat
-------------------------
Like the postflop sizing lookup, only spots reaching showdown are
captured. Folded hands are invisible. The bias is gentler in aggressive
spots (the ones the threat-range feature targets) because committed
opponents tend to show down. We accept this for v1 — see CLAUDE.md
postflop_sizing notes for the same trade-off.

Unlike the postflop sizing build, no cohort filter — uses every player
(matches the "all players for now" framing).

Usage:
    python3 scripts/build_opp_range.py
    python3 scripts/build_opp_range.py --rebuild
    python3 scripts/build_opp_range.py --max-files 100  # smoke test
    python3 scripts/build_opp_range.py --workers 32     # parallel ingest
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hand_history.parser import parse_file
from src.hand_history.models import ParsedHand
from src.hand_history.obs_builder import ObservationBuilder
from src.opp_range_spot_key import (
    build_opp_range_spot_key, hand_class_from_hole,
)
from src.opp_range_db import (
    connect, insert_observations_bulk, rebuild_lookup,
    is_file_processed, mark_file_processed, summary,
)


BB_USD = 0.05
DOLLARS_TO_CHIPS = 40.0   # 1 chip = $0.025 = 0.5 BB → 1 BB = 2 chips


def _walk_hand(hand: ParsedHand, obs_builder: ObservationBuilder,
                source_file: str) -> Iterable[Tuple]:
    """Yield observation rows for every postflop decision by a showdown player.

    Each row: (street, board_texture, position, spr_bucket, action_context,
              hand_class, player_name, hand_id, source_file).
    """
    if hand.num_active_players != 6:
        return
    if abs(hand.big_blind_dollars - BB_USD) > 0.001:
        return

    try:
        dps = obs_builder.build_decision_points(
            hand, source_file=source_file, allow_unknown_cards=False)
    except Exception:
        return
    dp_by_idx = {(dp.player_name, dp.action_index): dp for dp in dps}

    pot_chips = 0
    current_street = 'preflop'
    player_street_bet: Dict[str, int] = defaultdict(int)
    preflop_raiser: Optional[str] = None
    starting_stack: Dict[str, int] = {p.name: p.stack_chips
                                       for p in hand.players}
    folded: set = set()
    counter: Dict[str, int] = defaultdict(int)

    for action in hand.actions:
        if action.action_type in ('blind_sb', 'blind_bb'):
            chips = action.amount_chips
            pot_chips += chips
            player_street_bet[action.player_name] = chips
            continue

        actor = action.player_name
        if actor in folded:
            continue

        if action.street != current_street:
            current_street = action.street
            player_street_bet = defaultdict(int)

        action_idx = counter[actor]
        counter[actor] += 1

        # Only postflop decisions, and only if this actor's cards are known
        # (i.e. they later reached showdown).
        is_postflop = current_street in ('flop', 'turn', 'river')
        if is_postflop:
            dp = dp_by_idx.get((actor, action_idx))
            if dp is not None and dp.hole_cards and len(dp.hole_cards) >= 2:
                obs = dp.observation
                hand_class = hand_class_from_hole(dp.hole_cards)
                if hand_class is None:
                    pass
                else:
                    opp_bet = max(
                        (b for n, b in player_street_bet.items() if n != actor),
                        default=0,
                    )
                    hero_bet = player_street_bet.get(actor, 0)
                    facing_bet_size_pct = None
                    if opp_bet > hero_bet and pot_chips > 0:
                        facing_bet_size_pct = (opp_bet - hero_bet) / pot_chips

                    was_pf_raiser = (actor == preflop_raiser)
                    pos_label = dp.position_label
                    pos_map = {'BTN': 0, 'SB': 1, 'BB': 2,
                               'LJ': 3, 'HJ': 4, 'CO': 5,
                               'UTG': 3, 'MP': 4}
                    rel_pos = pos_map.get(pos_label, 0)

                    eff_chips = max(starting_stack.get(actor, 200)
                                     - hero_bet, 1)
                    eff_bb = eff_chips / 2.0
                    pot_bb = pot_chips / 2.0

                    try:
                        spot = build_opp_range_spot_key(
                            street=current_street, obs=obs,
                            rel_pos=rel_pos,
                            num_active=hand.num_active_players,
                            effective_stack_bb=eff_bb,
                            pot_bb=pot_bb,
                            was_preflop_raiser=was_pf_raiser,
                            facing_bet_size_pct=facing_bet_size_pct,
                        )
                    except Exception:
                        spot = None

                    if spot is not None:
                        yield (
                            spot.street, spot.board_texture, spot.position,
                            spot.spr_bucket, spot.action_context,
                            hand_class, actor, hand.hand_id, source_file,
                        )

        # ── APPLY ACTION TO STATE ──
        if action.action_type == 'fold':
            folded.add(actor)
            continue
        if action.action_type == 'check':
            continue
        chips = action.amount_chips
        if action.action_type == 'raise':
            increment = max(0, chips - player_street_bet[actor])
            player_street_bet[actor] = chips
            pot_chips += increment
            if current_street == 'preflop':
                preflop_raiser = actor
        elif action.action_type in ('call', 'bet'):
            player_street_bet[actor] += chips


# ── Worker-process state ──
_W_OBS_BUILDER: Optional[ObservationBuilder] = None


def _worker_init() -> None:
    global _W_OBS_BUILDER
    _W_OBS_BUILDER = ObservationBuilder(num_players=6)


def _process_file(arg: Tuple[str, float]) -> Tuple[str, float, List[Tuple], int]:
    path_str, mtime = arg
    obs_builder = _W_OBS_BUILDER
    rows: List[Tuple] = []
    hands_in_file = 0
    try:
        hands = parse_file(path_str)
    except Exception:
        return path_str, mtime, rows, 0

    for h in hands:
        if h.num_active_players != 6:
            continue
        if abs(h.big_blind_dollars - BB_USD) > 0.001:
            continue
        hands_in_file += 1
        try:
            for row in _walk_hand(h, obs_builder, path_str):
                rows.append(row)
        except Exception:
            continue

    return path_str, mtime, rows, hands_in_file


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--hh-dir', default='data/poker_archive')
    ap.add_argument('--db-path', default='data/opp_range.sqlite')
    ap.add_argument('--rebuild', action='store_true',
                    help='Drop+rebuild lookup from existing observations')
    ap.add_argument('--no-cache', action='store_true',
                    help='Re-ingest every HH file even if previously processed')
    ap.add_argument('--max-files', type=int, default=0)
    ap.add_argument('--workers', type=int,
                    default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    conn = connect(args.db_path)

    if args.rebuild:
        n = rebuild_lookup(conn)
        s = summary(conn)
        print(f'Rebuilt opp_range_lookup: {n} (spot,hand) rows '
              f'across {s["n_spots"]} spots from '
              f'{s["n_observations"]:,} observations')
        return

    files = sorted(Path(args.hh_dir).rglob('*.txt'))
    if args.max_files > 0:
        files = files[:args.max_files]
    print(f'HH files: {len(files)}')

    # Pre-filter: skip files already processed
    processed_set = set()
    if not args.no_cache:
        for row in conn.execute(
                "SELECT path, mtime FROM opp_range_processed_files"):
            processed_set.add((row[0], row[1]))

    todo: List[Tuple[str, float]] = []
    for p in files:
        st = p.stat()
        key = (str(p), st.st_mtime)
        if key in processed_set:
            continue
        if not args.no_cache and is_file_processed(conn, str(p), st.st_mtime):
            continue
        todo.append((str(p), st.st_mtime))

    print(f'Files to process: {len(todo)} '
          f'(skipped {len(files) - len(todo)} cached)')

    if not todo:
        print('Nothing to ingest — rebuilding lookup from existing observations.')
        n = rebuild_lookup(conn)
        s = summary(conn)
        print(f'Lookup: {n} rows | {s["n_spots"]} spots | '
              f'{s["n_observations"]:,} observations')
        return

    t_start = time.time()
    total_rows = 0
    total_hands = 0
    files_done = 0
    batch_rows: List[Tuple] = []
    BATCH = 5000

    if args.workers <= 1:
        _worker_init()
        for arg in todo:
            path_str, mtime, rows, hand_count = _process_file(arg)
            batch_rows.extend(rows)
            total_rows += len(rows)
            total_hands += hand_count
            files_done += 1
            mark_file_processed(conn, path_str, mtime, hand_count)
            if len(batch_rows) >= BATCH:
                insert_observations_bulk(conn, batch_rows)
                batch_rows = []
            if files_done % 100 == 0 and not args.quiet:
                _print_progress(files_done, len(todo), total_rows,
                                 total_hands, t_start)
    else:
        with mp.Pool(processes=args.workers, initializer=_worker_init) as pool:
            for path_str, mtime, rows, hand_count in pool.imap_unordered(
                    _process_file, todo, chunksize=4):
                batch_rows.extend(rows)
                total_rows += len(rows)
                total_hands += hand_count
                files_done += 1
                mark_file_processed(conn, path_str, mtime, hand_count)
                if len(batch_rows) >= BATCH:
                    insert_observations_bulk(conn, batch_rows)
                    batch_rows = []
                if files_done % 100 == 0 and not args.quiet:
                    _print_progress(files_done, len(todo), total_rows,
                                     total_hands, t_start)

    if batch_rows:
        insert_observations_bulk(conn, batch_rows)
    print(f'\nIngested {total_rows:,} observations from '
          f'{total_hands:,} hands in {files_done} files '
          f'({time.time() - t_start:.0f}s)')

    print('Rebuilding aggregated lookup...')
    n = rebuild_lookup(conn)
    s = summary(conn)
    print(f'Lookup: {n:,} (spot,hand) rows | {s["n_spots"]} distinct spots')


def _print_progress(done, total, rows, hands, t_start):
    elapsed = time.time() - t_start
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    print(f'  {done}/{total} files | {rows:,} rows | {hands:,} hands | '
          f'{rate:.1f} files/s | ETA {eta:.0f}s')


if __name__ == '__main__':
    main()
