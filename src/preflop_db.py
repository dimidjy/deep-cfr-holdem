"""SQLite I/O for the preflop lookup table.

Two-table design:
  - `decisions` (raw): one row per cohort player's preflop decision.
    Durable source of truth — re-bucketing the lookup never requires
    re-parsing HH files.
  - `lookup` (aggregated): one row per spot key, carrying the action
    distribution + raise-size statistics. Rebuilt from `decisions` on
    demand.

Plus a small `processed_files` table so the build script can skip
already-ingested HH files when run incrementally.

All public functions take or return a `sqlite3.Connection` so callers
can manage transactions and connection lifetime explicitly.
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.preflop_spot_key import SpotKey

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    hole_class     TEXT NOT NULL,
    position       TEXT NOT NULL,
    action_history TEXT NOT NULL,
    stack_bucket   TEXT NOT NULL,
    num_to_act     INTEGER NOT NULL,
    player_name    TEXT NOT NULL,
    action         TEXT NOT NULL,    -- 'fold' | 'call' | 'raise'
    size_bb        REAL,             -- NULL for fold/check; raise total in BB
    pot_bb_at_decision REAL,
    hand_id        TEXT,
    source_file    TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_spot ON decisions(
    hole_class, position, action_history, stack_bucket, num_to_act);

CREATE TABLE IF NOT EXISTS lookup (
    hole_class     TEXT NOT NULL,
    position       TEXT NOT NULL,
    action_history TEXT NOT NULL,
    stack_bucket   TEXT NOT NULL,
    num_to_act     INTEGER NOT NULL,
    n_samples      INTEGER NOT NULL,
    fold_pct       REAL NOT NULL,
    call_pct       REAL NOT NULL,
    raise_pct      REAL NOT NULL,
    raise_size_bb_p25  REAL,
    raise_size_bb_p50  REAL,
    raise_size_bb_p75  REAL,
    raise_size_bb_mean REAL,
    last_built_at TEXT,
    PRIMARY KEY (hole_class, position, action_history, stack_bucket, num_to_act)
);

CREATE TABLE IF NOT EXISTS processed_files (
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
    conn.commit()
    # Mild perf tweaks; fine for a write-once / read-many workload.
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    return conn


def insert_decisions_bulk(conn: sqlite3.Connection,
                          rows: Iterable[Tuple]) -> int:
    """Bulk insert rows into `decisions`.

    Each row: (hole_class, position, action_history, stack_bucket,
              num_to_act, player_name, action, size_bb, pot_bb_at_decision,
              hand_id, source_file).
    Returns number inserted.
    """
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        """INSERT INTO decisions
           (hole_class, position, action_history, stack_bucket, num_to_act,
            player_name, action, size_bb, pot_bb_at_decision,
            hand_id, source_file)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return len(rows)


def rebuild_lookup(conn: sqlite3.Connection) -> int:
    """Recompute the `lookup` table from `decisions`.

    Replaces all rows. Returns the number of distinct spots produced.
    """
    conn.execute('DELETE FROM lookup')

    # Pull aggregated counts per spot
    spots = conn.execute(
        """SELECT hole_class, position, action_history, stack_bucket, num_to_act,
                  SUM(CASE WHEN action='fold'  THEN 1 ELSE 0 END) AS folds,
                  SUM(CASE WHEN action='call'  THEN 1 ELSE 0 END) AS calls,
                  SUM(CASE WHEN action='raise' THEN 1 ELSE 0 END) AS raises,
                  COUNT(*) AS n
           FROM decisions
           GROUP BY hole_class, position, action_history, stack_bucket, num_to_act"""
    ).fetchall()

    now_iso = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    inserts = []
    for hole, pos, hist, sb, nta, folds, calls, raises, n in spots:
        if n == 0:
            continue
        # Pull raise sizes for this spot to compute mean + percentiles
        sizes = [r[0] for r in conn.execute(
            """SELECT size_bb FROM decisions
               WHERE hole_class=? AND position=? AND action_history=?
                 AND stack_bucket=? AND num_to_act=?
                 AND action='raise' AND size_bb IS NOT NULL""",
            (hole, pos, hist, sb, nta),
        ).fetchall()]
        if sizes:
            sizes.sort()
            mean = sum(sizes) / len(sizes)
            p25 = sizes[max(0, int(0.25 * len(sizes)) - 1)]
            p50 = sizes[len(sizes) // 2]
            p75 = sizes[min(len(sizes) - 1, int(0.75 * len(sizes)))]
        else:
            mean = p25 = p50 = p75 = None

        inserts.append((
            hole, pos, hist, sb, nta, n,
            100.0 * folds / n, 100.0 * calls / n, 100.0 * raises / n,
            p25, p50, p75, mean, now_iso,
        ))

    conn.executemany(
        """INSERT INTO lookup
           (hole_class, position, action_history, stack_bucket, num_to_act,
            n_samples, fold_pct, call_pct, raise_pct,
            raise_size_bb_p25, raise_size_bb_p50, raise_size_bb_p75,
            raise_size_bb_mean, last_built_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        inserts,
    )
    conn.commit()
    return len(inserts)


def query_spot(conn: sqlite3.Connection, key: SpotKey) -> Optional[Dict]:
    """Look up a single spot. Returns dict or None if not found."""
    row = conn.execute(
        """SELECT n_samples, fold_pct, call_pct, raise_pct,
                  raise_size_bb_p25, raise_size_bb_p50,
                  raise_size_bb_p75, raise_size_bb_mean
           FROM lookup
           WHERE hole_class=? AND position=? AND action_history=?
             AND stack_bucket=? AND num_to_act=?""",
        key.to_tuple(),
    ).fetchone()
    if row is None:
        return None
    n, fp, cp, rp, p25, p50, p75, mean = row
    return {
        'n_samples': n,
        'fold_pct': fp, 'call_pct': cp, 'raise_pct': rp,
        'raise_size_bb_p25': p25, 'raise_size_bb_p50': p50,
        'raise_size_bb_p75': p75, 'raise_size_bb_mean': mean,
    }


def load_lookup_into_memory(conn: sqlite3.Connection) -> Dict[Tuple, Dict]:
    """Fetch the entire lookup table into a dict keyed by spot tuple.

    For runtime use — avoids per-decision SQLite roundtrips.
    """
    out: Dict[Tuple, Dict] = {}
    for row in conn.execute(
        """SELECT hole_class, position, action_history, stack_bucket, num_to_act,
                  n_samples, fold_pct, call_pct, raise_pct,
                  raise_size_bb_p25, raise_size_bb_p50,
                  raise_size_bb_p75, raise_size_bb_mean
           FROM lookup"""
    ):
        hole, pos, hist, sb, nta = row[0:5]
        out[(hole, pos, hist, sb, nta)] = {
            'n_samples': row[5],
            'fold_pct': row[6], 'call_pct': row[7], 'raise_pct': row[8],
            'raise_size_bb_p25': row[9], 'raise_size_bb_p50': row[10],
            'raise_size_bb_p75': row[11], 'raise_size_bb_mean': row[12],
        }
    return out


def is_file_processed(conn: sqlite3.Connection, path: str, mtime: float) -> bool:
    """True if file at `path` was processed at the same mtime."""
    row = conn.execute(
        'SELECT mtime FROM processed_files WHERE path=?', (path,),
    ).fetchone()
    return row is not None and abs(row[0] - mtime) < 1e-3


def mark_file_processed(conn: sqlite3.Connection, path: str,
                        mtime: float, hand_count: int) -> None:
    conn.execute(
        """INSERT INTO processed_files (path, mtime, hand_count, processed_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(path) DO UPDATE SET
             mtime=excluded.mtime,
             hand_count=excluded.hand_count,
             processed_at=excluded.processed_at""",
        (path, mtime, hand_count, datetime.utcnow().isoformat(timespec='seconds') + 'Z'),
    )
    conn.commit()


def lookup_summary(conn: sqlite3.Connection) -> Dict:
    """Quick-and-dirty diagnostics for `--rebuild` runs and tests."""
    n_decisions = conn.execute('SELECT COUNT(*) FROM decisions').fetchone()[0]
    n_spots = conn.execute('SELECT COUNT(*) FROM lookup').fetchone()[0]
    n_files = conn.execute('SELECT COUNT(*) FROM processed_files').fetchone()[0]
    top = conn.execute(
        """SELECT hole_class, position, action_history, stack_bucket, num_to_act,
                  n_samples, ROUND(fold_pct,1), ROUND(call_pct,1), ROUND(raise_pct,1),
                  ROUND(raise_size_bb_p50,2)
           FROM lookup ORDER BY n_samples DESC LIMIT 10"""
    ).fetchall()
    return {
        'n_decisions': n_decisions,
        'n_spots': n_spots,
        'n_files_processed': n_files,
        'top_spots_by_samples': top,
    }
