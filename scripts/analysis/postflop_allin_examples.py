"""Pull real cohort hand-history examples for top postflop ALL-IN spots.

For each of the top-N promoting spots in ``postflop_lookup``, query the raw
``postflop_decisions`` table and emit a few representative all-in rows
alongside a few representative non-all-in rows so the user can spot-check
that the cohort behaviour looks valid.

Usage:
    python3 scripts/analysis/postflop_allin_examples.py
    python3 scripts/analysis/postflop_allin_examples.py --top 20 --min-samples 50
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.postflop_db import connect  # noqa: E402

DEFAULT_DB = str(ROOT / 'data' / 'postflop_sizing.sqlite')


def fetch_top_spots(conn, min_samples, threshold, limit):
    return conn.execute(
        """SELECT street, hand_bucket, draw_type, board_texture,
                  position, spr_bucket, action_context,
                  n_samples, n_all_in, all_in_pct,
                  bet_size_bb_p25, bet_size_bb_p50, bet_size_bb_p75
           FROM postflop_lookup
           WHERE n_samples >= ? AND all_in_pct >= ?
           ORDER BY n_samples DESC LIMIT ?""",
        (min_samples, threshold * 100, limit),
    ).fetchall()


def fetch_examples(conn, spot_key, is_all_in, limit=3):
    """Pull raw rows for the given spot key."""
    street, hb, dt, bt, pos, spr, ac = spot_key
    return conn.execute(
        """SELECT player_name, hand_id, pot_bb_at_decision,
                  spr_at_decision, size_bb, size_pct_pot, action,
                  source_file
           FROM postflop_decisions
           WHERE street=? AND hand_bucket=? AND draw_type=? AND
                 board_texture=? AND position=? AND spr_bucket=? AND
                 action_context=? AND is_all_in=?
           ORDER BY RANDOM() LIMIT ?""",
        (street, hb, dt, bt, pos, spr, ac, is_all_in, limit),
    ).fetchall()


def fmt_spot_header(spot, idx):
    (street, hb, dt, bt, pos, spr, ac,
     n, n_ai, pct, bb25, bb50, bb75) = spot
    bb50_s = f"{bb50:.1f}" if bb50 is not None else "?"
    bb75_s = f"{bb75:.1f}" if bb75 is not None else "?"
    return (
        f"\n{'='*100}\n"
        f"#{idx} SPOT: street={street} hand={hb} draw={dt} "
        f"board={bt} pos={pos} SPR={spr} ctx={ac}\n"
        f"  cohort: n={n}, all_in={n_ai} ({pct:.1f}%), "
        f"non-shove sizing p50={bb50_s}bb p75={bb75_s}bb\n"
        f"{'-'*100}"
    )


def fmt_row(row):
    (player, hand_id, pot_bb, spr, size_bb, size_pct, action, src) = row
    pot_s = f"{pot_bb:.1f}" if pot_bb is not None else "?"
    spr_s = f"{spr:.2f}" if spr is not None else "?"
    size_s = f"{size_bb:.1f}" if size_bb is not None else "?"
    pct_s = f"{size_pct*100:.0f}%" if size_pct is not None else "?"
    src_short = Path(src).name if src else '?'
    return (f"    {action:<6s} player={player[:15]:<15s} "
            f"pot={pot_s:>6s}bb spr={spr_s:>5s} "
            f"size={size_s:>6s}bb ({pct_s:>5s} of pot) "
            f"hand={hand_id} [{src_short}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--min-samples', type=int, default=20)
    ap.add_argument('--threshold', type=float, default=0.35)
    ap.add_argument('--examples-per-spot', type=int, default=3)
    args = ap.parse_args()

    if not Path(args.db).exists():
        sys.exit(f"DB not found: {args.db}")

    conn = connect(args.db)
    try:
        spots = fetch_top_spots(conn, args.min_samples,
                                args.threshold, args.top)
        if not spots:
            print("No spots match filter.")
            return

        print(f"Top {len(spots)} promoting spots (n>={args.min_samples}, "
              f"all_in_pct>={args.threshold*100:.0f}%)\n")

        for idx, spot in enumerate(spots, 1):
            spot_key = spot[:7]
            print(fmt_spot_header(spot, idx))

            ai_rows = fetch_examples(conn, spot_key, is_all_in=1,
                                     limit=args.examples_per_spot)
            print(f"  ALL-IN examples ({len(ai_rows)}):")
            for r in ai_rows:
                print(fmt_row(r))

            other_rows = fetch_examples(conn, spot_key, is_all_in=0,
                                        limit=args.examples_per_spot)
            print(f"  NON-ALL-IN examples ({len(other_rows)}):")
            for r in other_rows:
                print(fmt_row(r))
    finally:
        conn.close()


if __name__ == '__main__':
    main()
