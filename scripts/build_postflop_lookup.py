"""Build the postflop sizing lookup table from cohort players' hand histories.

For each cohort player's postflop bet/raise/all-in decision (with known
hole cards from showdown), record:
  - the spot key (hand bucket / draw / texture / position / SPR / action_ctx)
  - the size as fraction of pot at decision time + absolute BB
  - whether it was an all-in

After ingesting the archive, aggregate into `postflop_lookup` with
percentile sizing per spot. Resumable via `postflop_processed_files`.

Showdown-only data caveat
-------------------------
Like the preflop lookup, only spots reaching showdown are captured.
Postflop-bet bias is gentler than preflop-fold bias — most aggressive
postflop actions go to showdown — but missing folded-after-bet hands
still under-represent failed-bluff sizings. Aggregation still produces
useful "what size do winners actually use here" signal.

Usage:
    python3 scripts/build_postflop_lookup.py
    python3 scripts/build_postflop_lookup.py --rebuild
    python3 scripts/build_postflop_lookup.py --max-files 100  # smoke
    python3 scripts/build_postflop_lookup.py --workers 32     # parallel
    python3 scripts/build_postflop_lookup.py --export-json data/postflop.json
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
from src.hand_history.models import ParsedHand, ParsedAction
from src.hand_history.obs_builder import ObservationBuilder
from src.postflop_spot_key import build_spot_key
from src.postflop_db import (
    connect, insert_decisions_bulk, rebuild_lookup,
    is_file_processed, mark_file_processed, lookup_summary,
    export_lookup_to_json,
)


BB_USD = 0.05
DOLLARS_TO_CHIPS = 40.0   # 1 chip = $0.025 = 0.5 BB → 1 BB = 2 chips


def _action_to_chips(amount_dollars: float) -> int:
    return int(round(amount_dollars * DOLLARS_TO_CHIPS))


def _walk_hand(hand: ParsedHand, cohort: set,
               obs_builder: ObservationBuilder,
               source_file: str) -> Iterable[Tuple]:
    """Yield decision rows for every cohort postflop bet/raise/all-in.

    Each row is the tuple expected by `postflop_db.insert_decisions_bulk`.
    """
    if hand.num_active_players != 6:
        return
    if abs(hand.big_blind_dollars - BB_USD) > 0.001:
        return

    # Pre-compute decision points to grab the obs vector per action.
    try:
        dps = obs_builder.build_decision_points(
            hand, source_file=source_file, allow_unknown_cards=False)
    except Exception:
        return
    # Index DPs by (player_name, action_index_for_player)
    dp_by_idx = {(dp.player_name, dp.action_index): dp for dp in dps}

    # State tracker
    pot_chips = 0
    current_street = 'preflop'
    player_street_bet: Dict[str, int] = defaultdict(int)
    street_max_bet = 0
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
            street_max_bet = max(street_max_bet, chips)
            continue

        actor = action.player_name
        if actor in folded:
            continue

        # Street transition — reset per-street state
        if action.street != current_street:
            current_street = action.street
            player_street_bet = defaultdict(int)
            street_max_bet = 0

        action_idx = counter[actor]
        counter[actor] += 1

        # ── BUILD ROW (only for cohort postflop bets/raises with known cards) ──
        is_postflop = current_street in ('flop', 'turn', 'river')
        is_aggressive = (action.action_type in ('bet', 'raise')
                         or action.is_all_in)

        if (is_postflop and is_aggressive and actor in cohort):
            dp = dp_by_idx.get((actor, action_idx))
            # dp.hole_cards is None when we don't know cards (no showdown)
            if dp is not None and dp.hole_cards and len(dp.hole_cards) >= 2:
                obs = dp.observation
                pot_before = pot_chips   # in chips, before this action
                size_chips = action.amount_chips

                # size_pct_pot = total bet amount / pot before action
                # (matches runtime convention)
                size_pct_pot = size_chips / pot_before if pot_before > 0 else None
                size_bb = size_chips / 2.0    # 2 chips = 1 BB

                # facing_bet_size_pct: opponent's max bet on this street so far
                # (excluding hero's own bet) divided by pot_before_action
                opp_bet = max(
                    (b for n, b in player_street_bet.items() if n != actor),
                    default=0,
                )
                hero_bet = player_street_bet.get(actor, 0)
                facing_bet_size_pct = None
                if opp_bet > hero_bet and pot_before > 0:
                    facing_bet_size_pct = (opp_bet - hero_bet) / pot_before

                was_pf_raiser = (actor == preflop_raiser)

                # Position: rel_pos from active+sorted_seats — DP has it
                rel_pos = 0
                pos_label = dp.position_label
                # Map BTN→0, SB→1, BB→2, LJ→3, HJ→4, CO→5
                pos_map = {'BTN': 0, 'SB': 1, 'BB': 2,
                           'LJ': 3, 'HJ': 4, 'CO': 5,
                           'UTG': 3, 'MP': 4}
                rel_pos = pos_map.get(pos_label, 0)

                # Effective stack: hero remaining stack at this decision
                eff_chips = max(starting_stack.get(actor, 200)
                                - hero_bet, 1)
                eff_bb = eff_chips / 2.0
                pot_bb = pot_before / 2.0

                action_label = ('all_in' if action.is_all_in
                                else action.action_type)
                is_all_in = 1 if action.is_all_in else 0

                try:
                    spot = build_spot_key(
                        street=current_street, obs=obs,
                        hole_cards=dp.hole_cards,
                        community_cards=dp.community_cards,
                        rel_pos=rel_pos,
                        num_active=hand.num_active_players,
                        effective_stack_bb=eff_bb,
                        pot_bb=pot_bb,
                        was_preflop_raiser=was_pf_raiser,
                        facing_bet_size_pct=facing_bet_size_pct,
                        facing_check_raise=False,
                    )
                except Exception:
                    spot = None

                if spot is not None:
                    yield (
                        spot.street, spot.hand_bucket, spot.draw_type,
                        spot.board_texture, spot.position, spot.spr_bucket,
                        spot.action_context, action_label,
                        size_pct_pot, size_bb, is_all_in,
                        pot_bb, eff_bb / pot_bb if pot_bb > 0 else None,
                        actor, hand.hand_id, source_file,
                    )

        # ── APPLY ACTION TO STATE ──
        if action.action_type == 'fold':
            folded.add(actor)
            continue
        if action.action_type == 'check':
            continue
        chips = action.amount_chips
        if action.action_type == 'raise':
            # Cumulative on this street
            increment = max(0, chips - player_street_bet[actor])
            player_street_bet[actor] = chips
            pot_chips += increment
            street_max_bet = chips
            if current_street == 'preflop':
                preflop_raiser = actor
        elif action.action_type in ('call', 'bet'):
            # Incremental
            player_street_bet[actor] += chips
            pot_chips += chips
            if action.action_type == 'bet':
                street_max_bet = player_street_bet[actor]


# ── Worker-process state (one ObservationBuilder + cohort per worker) ──
_W_OBS_BUILDER: Optional[ObservationBuilder] = None
_W_COHORT: Optional[frozenset] = None


def _worker_init(cohort_set: frozenset) -> None:
    """Pool initializer — runs once per worker process. Building the
    ObservationBuilder is non-trivial so we do it once per worker, not
    per file."""
    global _W_OBS_BUILDER, _W_COHORT
    _W_OBS_BUILDER = ObservationBuilder(num_players=6)
    _W_COHORT = cohort_set


def _process_file(arg: Tuple[str, float]) -> Tuple[str, float, List[Tuple], int]:
    """Worker: parse one HH file, return all decision rows + bookkeeping.

    Returns (path, mtime, rows, hands_in_file). Errors are swallowed —
    file is still marked processed so we don't retry it forever.
    """
    path_str, mtime = arg
    obs_builder = _W_OBS_BUILDER
    cohort = _W_COHORT
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
            for row in _walk_hand(h, cohort, obs_builder, path_str):
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
    ap.add_argument('--good-players',
                    default='data/good_players_winrate10.txt')
    ap.add_argument('--db-path', default='data/postflop_sizing.sqlite')
    ap.add_argument('--rebuild', action='store_true',
                    help='Drop+rebuild lookup from existing decisions')
    ap.add_argument('--no-cache', action='store_true',
                    help='Re-ingest every HH file even if previously processed')
    ap.add_argument('--max-files', type=int, default=0)
    ap.add_argument('--workers', type=int,
                    default=max(1, (os.cpu_count() or 4) - 1),
                    help='Parallel worker processes (default: cpu_count-1). '
                         'Use 1 to run serially.')
    ap.add_argument('--export-json', default='',
                    help='If set, dump aggregated lookup to this JSON path')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    # Cohort
    cohort = set()
    if Path(args.good_players).exists():
        with open(args.good_players) as f:
            cohort = {line.strip() for line in f if line.strip()}
    print(f'Cohort: {len(cohort)} players from {args.good_players}')
    if not cohort and not args.rebuild:
        sys.exit('Cohort empty — run scripts/compute_player_winrates.py first.')

    conn = connect(args.db_path)

    if args.rebuild and not cohort:
        n = rebuild_lookup(conn)
        s = lookup_summary(conn)
        print(f'Rebuilt postflop lookup: {n} spots from '
              f'{s["n_decisions"]} decisions')
        if args.export_json:
            n_exp = export_lookup_to_json(conn, args.export_json)
            print(f'Exported {n_exp} spots to {args.export_json}')
        return

    files = sorted(Path(args.hh_dir).rglob('*.txt'))
    if args.max_files > 0:
        files = files[:args.max_files]
    print(f'HH files: {len(files)}')

    # Pre-filter: skip files already processed (single bulk load of the
    # processed table is much faster than per-file SELECT).
    processed_set = set()
    if not args.no_cache:
        rows = conn.execute(
            'SELECT path, mtime FROM postflop_processed_files').fetchall()
        processed_set = {(p, round(m, 3)) for p, m in rows}

    to_process: List[Tuple[str, float]] = []
    n_files_skipped = 0
    for path in files:
        path_str = str(path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if (path_str, round(mtime, 3)) in processed_set:
            n_files_skipped += 1
            continue
        to_process.append((path_str, mtime))

    print(f'To process: {len(to_process)} files '
          f'(skipping {n_files_skipped} already processed) '
          f'with {args.workers} worker(s)')

    t0 = time.time()
    batch: List[Tuple] = []
    BATCH_SIZE = 5_000   # smaller batch = more frequent progress, still fast
    n_inserted = 0
    n_files_done = 0

    # Single-process fast path (avoid Pool overhead for tiny runs)
    if args.workers <= 1:
        obs_builder = ObservationBuilder(num_players=6)
        for fi, (path_str, mtime) in enumerate(to_process):
            try:
                hands = parse_file(path_str)
            except Exception:
                continue
            hands_in_file = 0
            for h in hands:
                if h.num_active_players != 6:
                    continue
                if abs(h.big_blind_dollars - BB_USD) > 0.001:
                    continue
                hands_in_file += 1
                for row in _walk_hand(h, cohort, obs_builder, path_str):
                    batch.append(row)
                    if len(batch) >= BATCH_SIZE:
                        n_inserted += insert_decisions_bulk(conn, batch)
                        batch.clear()
            mark_file_processed(conn, path_str, mtime, hands_in_file)
            n_files_done += 1
            if not args.quiet and n_files_done % 200 == 0:
                _print_progress(fi + 1, len(to_process), n_inserted,
                                len(batch), n_files_skipped, t0)
    else:
        # Parallel path: workers parse, master writes to SQLite.
        ctx = mp.get_context('fork')   # fork = no re-import overhead
        with ctx.Pool(processes=args.workers,
                      initializer=_worker_init,
                      initargs=(frozenset(cohort),)) as pool:
            # chunksize tuning: larger chunks reduce IPC overhead, smaller
            # chunks balance load better. 8 is a reasonable middle.
            for path_str, mtime, rows, hands_in_file in pool.imap_unordered(
                    _process_file, to_process, chunksize=8):
                if rows:
                    batch.extend(rows)
                    while len(batch) >= BATCH_SIZE:
                        n_inserted += insert_decisions_bulk(
                            conn, batch[:BATCH_SIZE])
                        del batch[:BATCH_SIZE]
                mark_file_processed(conn, path_str, mtime, hands_in_file)
                n_files_done += 1
                if not args.quiet and n_files_done % 200 == 0:
                    _print_progress(n_files_done, len(to_process), n_inserted,
                                    len(batch), n_files_skipped, t0)

    if batch:
        n_inserted += insert_decisions_bulk(conn, batch)
        batch.clear()

    elapsed = time.time() - t0
    print(f'Ingestion done in {elapsed:.0f}s. '
          f'{n_files_done} processed, {n_files_skipped} skipped, '
          f'{n_inserted} decisions inserted this run.')

    print('Rebuilding lookup table...')
    n_spots = rebuild_lookup(conn)
    s = lookup_summary(conn)
    print(f'Lookup: {n_spots} unique spots from '
          f'{s["n_decisions"]} total decisions across '
          f'{s["n_files_processed"]} files.')

    print('\nTop 15 spots by sample count:')
    print(f'  {"street":<6} {"hand":<14} {"draw":<11} {"tex":<10} {"pos":<3} '
          f'{"spr":<5} {"ctx":<14} {"n":>5} {"shove%":>6} {"p50%":>5} {"p50bb":>6}')
    for (street, hb, dt, bt, pos, spr, ac, n, ai, p50_pct, p50_bb) in s['top_spots_by_samples']:
        p50_pct = p50_pct if p50_pct is not None else 0
        p50_bb = p50_bb if p50_bb is not None else 0
        print(f'  {street:<6} {hb:<14} {dt:<11} {bt:<10} {pos:<3} '
              f'{spr:<5} {ac:<14} {n:>5} {ai:>5.0f}% {p50_pct:>4.0f}% {p50_bb:>5.1f}')

    if args.export_json:
        n_exp = export_lookup_to_json(conn, args.export_json)
        print(f'\nExported {n_exp} spots to {args.export_json}')


def _print_progress(done: int, total: int, n_inserted: int,
                    buffered: int, n_skipped: int, t0: float) -> None:
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    print(f'  [{done}/{total}] {n_inserted} inserted (+{buffered} buffered), '
          f'{n_skipped} skipped — {elapsed:.0f}s elapsed, '
          f'~{eta:.0f}s remaining ({rate:.1f} files/s)')


if __name__ == '__main__':
    main()
