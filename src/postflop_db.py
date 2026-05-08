"""SQLite I/O for the postflop sizing lookup.

Two-table design:
  - `postflop_decisions` (raw): one row per cohort player's postflop bet/raise.
    Durable source of truth — re-bucketing the lookup never requires
    re-parsing 20M HH files.
  - `postflop_lookup` (aggregated): one row per spot key with action
    distribution + raise-size statistics. Rebuilt from `postflop_decisions`.

Plus a small `postflop_processed_files` table so the build script can skip
already-ingested HH files when run incrementally.

Mirrors `src/preflop_db.py` patterns. All public functions take or return
a `sqlite3.Connection` so callers manage transactions explicitly.
"""
import json as _json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.postflop_spot_key import PostflopSpotKey

# Bucket edges for the per-spot bet-size histogram (% of pot).
# Buckets are [edges[i], edges[i+1]) with the last open-ended at +inf.
# 7 buckets: 0-33 / 33-50 / 50-75 / 75-100 / 100-150 / 150-200 / 200+.
BET_SIZE_BUCKET_EDGES: Tuple[float, ...] = (0.0, 0.33, 0.50, 0.75, 1.00, 1.50, 2.00)
NUM_BET_SIZE_BUCKETS: int = len(BET_SIZE_BUCKET_EDGES)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS postflop_decisions (
    id INTEGER PRIMARY KEY,
    street          TEXT NOT NULL,
    hand_bucket     TEXT NOT NULL,
    draw_type       TEXT NOT NULL,
    board_texture   TEXT NOT NULL,
    position        TEXT NOT NULL,
    spr_bucket      TEXT NOT NULL,
    action_context  TEXT NOT NULL,
    action          TEXT NOT NULL,    -- 'bet' | 'raise' | 'all_in'
    size_pct_pot    REAL,             -- size as fraction of pot at decision
    size_bb         REAL,             -- absolute size in BB
    is_all_in       INTEGER NOT NULL,
    pot_bb_at_decision REAL,
    spr_at_decision REAL,
    player_name     TEXT,
    hand_id         TEXT,
    source_file     TEXT
);
CREATE INDEX IF NOT EXISTS idx_postflop_spot ON postflop_decisions(
    street, hand_bucket, draw_type, board_texture, position,
    spr_bucket, action_context);

CREATE TABLE IF NOT EXISTS postflop_lookup (
    street          TEXT NOT NULL,
    hand_bucket     TEXT NOT NULL,
    draw_type       TEXT NOT NULL,
    board_texture   TEXT NOT NULL,
    position        TEXT NOT NULL,
    spr_bucket      TEXT NOT NULL,
    action_context  TEXT NOT NULL,
    n_samples       INTEGER NOT NULL,
    n_all_in        INTEGER NOT NULL,
    all_in_pct      REAL NOT NULL,
    bet_size_pct_p25 REAL,
    bet_size_pct_p50 REAL,
    bet_size_pct_p75 REAL,
    bet_size_bb_p25  REAL,
    bet_size_bb_p50  REAL,
    bet_size_bb_p75  REAL,
    size_hist_json   TEXT,
    last_built_at   TEXT,
    PRIMARY KEY (street, hand_bucket, draw_type, board_texture,
                 position, spr_bucket, action_context)
);

CREATE TABLE IF NOT EXISTS postflop_processed_files (
    path         TEXT PRIMARY KEY,
    mtime        REAL NOT NULL,
    hand_count   INTEGER,
    processed_at TEXT
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    """Open (and lazily create) a SQLite DB; ensure schema exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    # Forward-compat: add columns added after initial schema. CREATE TABLE
    # IF NOT EXISTS won't add columns to existing tables.
    cols = {row[1] for row in conn.execute(
        "PRAGMA table_info(postflop_lookup)").fetchall()}
    if 'size_hist_json' not in cols:
        conn.execute(
            "ALTER TABLE postflop_lookup ADD COLUMN size_hist_json TEXT")
    conn.commit()
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    return conn


def insert_decisions_bulk(conn: sqlite3.Connection,
                          rows: Iterable[Tuple]) -> int:
    """Bulk insert rows into `postflop_decisions`.

    Each row: (street, hand_bucket, draw_type, board_texture, position,
              spr_bucket, action_context, action, size_pct_pot, size_bb,
              is_all_in, pot_bb_at_decision, spr_at_decision, player_name,
              hand_id, source_file).
    Returns number inserted.
    """
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        """INSERT INTO postflop_decisions
           (street, hand_bucket, draw_type, board_texture, position,
            spr_bucket, action_context, action, size_pct_pot, size_bb,
            is_all_in, pot_bb_at_decision, spr_at_decision,
            player_name, hand_id, source_file)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return len(rows)


def _percentiles(sizes: List[float]) -> Tuple[Optional[float], ...]:
    """Return (p25, p50, p75) from a sorted list."""
    if not sizes:
        return (None, None, None)
    s = sorted(sizes)
    n = len(s)
    p25 = s[max(0, int(0.25 * n) - 1)]
    p50 = s[n // 2]
    p75 = s[min(n - 1, int(0.75 * n))]
    return (p25, p50, p75)


def _bucket_index(pct: float) -> int:
    """Return the bucket index for a bet-size pct using BET_SIZE_BUCKET_EDGES."""
    idx = 0
    for i, edge in enumerate(BET_SIZE_BUCKET_EDGES):
        if pct >= edge:
            idx = i
        else:
            break
    return idx


def _compute_size_histogram(pcts: List[float]) -> Optional[str]:
    """Bucket bet-size pcts and return JSON-encoded histogram.

    Output format:
        {"counts": [c0..c6], "medians": [m0..m6 or null]}

    `counts[i]` is the number of cohort decisions whose size_pct fell in
    bucket i (defined by `BET_SIZE_BUCKET_EDGES`).
    `medians[i]` is the median size_pct among decisions in bucket i, or
    None if the bucket is empty (used at runtime to pick a representative
    size in the chosen bucket).

    Returns None when `pcts` is empty so the column stays NULL — the
    runtime falls back to the existing percentile path in that case.
    """
    if not pcts:
        return None
    counts = [0] * NUM_BET_SIZE_BUCKETS
    bucket_vals: List[List[float]] = [[] for _ in range(NUM_BET_SIZE_BUCKETS)]
    for p in pcts:
        b = _bucket_index(float(p))
        counts[b] += 1
        bucket_vals[b].append(float(p))
    medians: List[Optional[float]] = []
    for vals in bucket_vals:
        if not vals:
            medians.append(None)
        else:
            vals.sort()
            medians.append(vals[len(vals) // 2])
    return _json.dumps({'counts': counts, 'medians': medians},
                       separators=(',', ':'))


def rebuild_lookup(conn: sqlite3.Connection) -> int:
    """Recompute the `postflop_lookup` table from `postflop_decisions`.

    Replaces all rows. Returns the number of distinct spots produced.
    """
    conn.execute('DELETE FROM postflop_lookup')

    # Aggregate counts per spot
    spots = conn.execute(
        """SELECT street, hand_bucket, draw_type, board_texture,
                  position, spr_bucket, action_context,
                  SUM(CASE WHEN is_all_in = 1 THEN 1 ELSE 0 END) AS n_all_in,
                  COUNT(*) AS n
           FROM postflop_decisions
           GROUP BY street, hand_bucket, draw_type, board_texture,
                    position, spr_bucket, action_context"""
    ).fetchall()

    now_iso = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    inserts = []
    for (street, hb, dt, bt, pos, spr, ac, n_all_in, n) in spots:
        if n == 0:
            continue
        # Pull raise sizes (excluding all-ins, since their sizes are
        # constrained by stack — we want to know cohort's chosen-bet sizes
        # for non-shove raises separately).
        size_rows = conn.execute(
            """SELECT size_pct_pot, size_bb FROM postflop_decisions
               WHERE street=? AND hand_bucket=? AND draw_type=? AND
                     board_texture=? AND position=? AND spr_bucket=? AND
                     action_context=? AND is_all_in = 0 AND
                     size_pct_pot IS NOT NULL""",
            (street, hb, dt, bt, pos, spr, ac),
        ).fetchall()
        pcts = [r[0] for r in size_rows if r[0] is not None]
        bbs = [r[1] for r in size_rows if r[1] is not None]
        pct_p25, pct_p50, pct_p75 = _percentiles(pcts)
        bb_p25, bb_p50, bb_p75 = _percentiles(bbs)
        size_hist_json = _compute_size_histogram(pcts)

        all_in_pct = 100.0 * n_all_in / n

        inserts.append((
            street, hb, dt, bt, pos, spr, ac,
            n, n_all_in, all_in_pct,
            pct_p25, pct_p50, pct_p75,
            bb_p25, bb_p50, bb_p75,
            size_hist_json,
            now_iso,
        ))

    conn.executemany(
        """INSERT INTO postflop_lookup
           (street, hand_bucket, draw_type, board_texture, position,
            spr_bucket, action_context, n_samples, n_all_in, all_in_pct,
            bet_size_pct_p25, bet_size_pct_p50, bet_size_pct_p75,
            bet_size_bb_p25, bet_size_bb_p50, bet_size_bb_p75,
            size_hist_json,
            last_built_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        inserts,
    )
    conn.commit()
    return len(inserts)


def _decode_size_hist(blob: Optional[str]) -> Optional[Dict]:
    """Parse size_hist_json into a dict, or None when absent/malformed."""
    if not blob:
        return None
    try:
        d = _json.loads(blob)
    except Exception:
        return None
    counts = d.get('counts')
    medians = d.get('medians')
    if (not isinstance(counts, list)
            or not isinstance(medians, list)
            or len(counts) != NUM_BET_SIZE_BUCKETS
            or len(medians) != NUM_BET_SIZE_BUCKETS):
        return None
    return {'counts': counts, 'medians': medians}


def query_spot(conn: sqlite3.Connection, key: PostflopSpotKey) -> Optional[Dict]:
    """Look up a single spot. Returns dict or None if not found."""
    row = conn.execute(
        """SELECT n_samples, n_all_in, all_in_pct,
                  bet_size_pct_p25, bet_size_pct_p50, bet_size_pct_p75,
                  bet_size_bb_p25, bet_size_bb_p50, bet_size_bb_p75,
                  size_hist_json
           FROM postflop_lookup
           WHERE street=? AND hand_bucket=? AND draw_type=? AND
                 board_texture=? AND position=? AND spr_bucket=? AND
                 action_context=?""",
        key.to_tuple(),
    ).fetchone()
    if row is None:
        return None
    n, n_ai, ai_pct, p25, p50, p75, bb25, bb50, bb75, hist_blob = row
    return {
        'n_samples': n, 'n_all_in': n_ai, 'all_in_pct': ai_pct,
        'bet_size_pct_p25': p25, 'bet_size_pct_p50': p50, 'bet_size_pct_p75': p75,
        'bet_size_bb_p25': bb25, 'bet_size_bb_p50': bb50, 'bet_size_bb_p75': bb75,
        'size_hist': _decode_size_hist(hist_blob),
    }


def load_lookup_into_memory(conn: sqlite3.Connection) -> Dict[Tuple, Dict]:
    """Fetch the entire postflop_lookup into a dict keyed by spot tuple.

    For runtime use — avoids per-decision SQLite roundtrips.
    """
    out: Dict[Tuple, Dict] = {}
    for row in conn.execute(
        """SELECT street, hand_bucket, draw_type, board_texture, position,
                  spr_bucket, action_context, n_samples, n_all_in, all_in_pct,
                  bet_size_pct_p25, bet_size_pct_p50, bet_size_pct_p75,
                  bet_size_bb_p25, bet_size_bb_p50, bet_size_bb_p75,
                  size_hist_json
           FROM postflop_lookup"""
    ):
        key = tuple(row[:7])
        out[key] = {
            'n_samples': row[7], 'n_all_in': row[8], 'all_in_pct': row[9],
            'bet_size_pct_p25': row[10], 'bet_size_pct_p50': row[11],
            'bet_size_pct_p75': row[12],
            'bet_size_bb_p25': row[13], 'bet_size_bb_p50': row[14],
            'bet_size_bb_p75': row[15],
            'size_hist': _decode_size_hist(row[16]),
        }
    return out


def is_file_processed(conn: sqlite3.Connection, path: str, mtime: float) -> bool:
    row = conn.execute(
        'SELECT mtime FROM postflop_processed_files WHERE path=?', (path,),
    ).fetchone()
    return row is not None and abs(row[0] - mtime) < 1e-3


def mark_file_processed(conn: sqlite3.Connection, path: str,
                        mtime: float, hand_count: int) -> None:
    conn.execute(
        """INSERT INTO postflop_processed_files (path, mtime, hand_count, processed_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(path) DO UPDATE SET
             mtime=excluded.mtime,
             hand_count=excluded.hand_count,
             processed_at=excluded.processed_at""",
        (path, mtime, hand_count,
         datetime.utcnow().isoformat(timespec='seconds') + 'Z'),
    )
    conn.commit()


def lookup_summary(conn: sqlite3.Connection) -> Dict:
    """Diagnostics for `--rebuild` runs and tests."""
    n_decisions = conn.execute('SELECT COUNT(*) FROM postflop_decisions').fetchone()[0]
    n_spots = conn.execute('SELECT COUNT(*) FROM postflop_lookup').fetchone()[0]
    n_files = conn.execute('SELECT COUNT(*) FROM postflop_processed_files').fetchone()[0]
    top = conn.execute(
        """SELECT street, hand_bucket, draw_type, board_texture, position,
                  spr_bucket, action_context, n_samples,
                  ROUND(all_in_pct, 1),
                  ROUND(bet_size_pct_p50 * 100, 0) AS p50_pct,
                  ROUND(bet_size_bb_p50, 1)
           FROM postflop_lookup ORDER BY n_samples DESC LIMIT 15"""
    ).fetchall()
    return {
        'n_decisions': n_decisions,
        'n_spots': n_spots,
        'n_files_processed': n_files,
        'top_spots_by_samples': top,
    }


def export_lookup_to_json(conn: sqlite3.Connection, out_path: str) -> int:
    """Dump the aggregated `postflop_lookup` to a JSON snapshot for portability."""
    import json as _json
    rows = []
    for row in conn.execute(
        """SELECT street, hand_bucket, draw_type, board_texture, position,
                  spr_bucket, action_context, n_samples, n_all_in, all_in_pct,
                  bet_size_pct_p25, bet_size_pct_p50, bet_size_pct_p75,
                  bet_size_bb_p25, bet_size_bb_p50, bet_size_bb_p75
           FROM postflop_lookup ORDER BY n_samples DESC"""
    ):
        rows.append({
            'spot': {
                'street': row[0], 'hand_bucket': row[1], 'draw_type': row[2],
                'board_texture': row[3], 'position': row[4],
                'spr_bucket': row[5], 'action_context': row[6],
            },
            'n_samples': row[7], 'n_all_in': row[8], 'all_in_pct': row[9],
            'bet_size_pct_p25': row[10], 'bet_size_pct_p50': row[11],
            'bet_size_pct_p75': row[12],
            'bet_size_bb_p25': row[13], 'bet_size_bb_p50': row[14],
            'bet_size_bb_p75': row[15],
        })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        _json.dump({'spots': rows, 'n_total': len(rows)}, f, indent=2)
    return len(rows)
