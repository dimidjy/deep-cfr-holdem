"""SQLite I/O for the empirical opponent-range lookup.

Two-table design mirroring `src/postflop_db.py`:
  - `opp_range_observations` (raw): one row per opponent showdown decision
    at a postflop spot. Append-only durable source of truth.
  - `opp_range_lookup` (aggregated): per-spot hand-class counts. Rebuilt
    from observations whenever you re-bucket or add new HH.

Plus a `opp_range_processed_files` table for resumable HH ingestion.
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


_SCHEMA = """
CREATE TABLE IF NOT EXISTS opp_range_observations (
    id INTEGER PRIMARY KEY,
    street          TEXT NOT NULL,
    board_texture   TEXT NOT NULL,
    position        TEXT NOT NULL,
    spr_bucket      TEXT NOT NULL,
    action_context  TEXT NOT NULL,
    hand_class      TEXT NOT NULL,       -- '169-class', e.g. 'AKs', 'T9o', 'QQ'
    player_name     TEXT,
    hand_id         TEXT,
    source_file     TEXT
);
CREATE INDEX IF NOT EXISTS idx_opp_range_spot ON opp_range_observations(
    street, board_texture, position, spr_bucket, action_context);

CREATE TABLE IF NOT EXISTS opp_range_lookup (
    street          TEXT NOT NULL,
    board_texture   TEXT NOT NULL,
    position        TEXT NOT NULL,
    spr_bucket      TEXT NOT NULL,
    action_context  TEXT NOT NULL,
    hand_class      TEXT NOT NULL,
    count           INTEGER NOT NULL,
    last_built_at   TEXT,
    PRIMARY KEY (street, board_texture, position, spr_bucket,
                 action_context, hand_class)
);

CREATE TABLE IF NOT EXISTS opp_range_processed_files (
    path         TEXT PRIMARY KEY,
    mtime        REAL NOT NULL,
    hand_count   INTEGER,
    processed_at TEXT
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    return conn


def insert_observations_bulk(conn: sqlite3.Connection,
                              rows: Iterable[Tuple]) -> int:
    """Bulk insert rows into `opp_range_observations`.

    Each row: (street, board_texture, position, spr_bucket, action_context,
              hand_class, player_name, hand_id, source_file).
    """
    cur = conn.cursor()
    rows = list(rows)
    cur.executemany(
        "INSERT INTO opp_range_observations "
        "(street, board_texture, position, spr_bucket, action_context, "
        " hand_class, player_name, hand_id, source_file) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def rebuild_lookup(conn: sqlite3.Connection) -> int:
    """Aggregate observations → opp_range_lookup. Returns rows written."""
    now = datetime.utcnow().isoformat(timespec='seconds')
    cur = conn.cursor()
    cur.execute("DELETE FROM opp_range_lookup")
    cur.execute(f"""
        INSERT INTO opp_range_lookup
            (street, board_texture, position, spr_bucket, action_context,
             hand_class, count, last_built_at)
        SELECT street, board_texture, position, spr_bucket, action_context,
               hand_class, COUNT(*) AS count, '{now}'
        FROM opp_range_observations
        GROUP BY street, board_texture, position, spr_bucket,
                 action_context, hand_class
    """)
    n = cur.rowcount
    conn.commit()
    return n


def query_spot(conn: sqlite3.Connection,
               street: str, board_texture: str, position: str,
               spr_bucket: str, action_context: str
               ) -> List[Tuple[str, int]]:
    """Return [(hand_class, count), ...] for a spot. Empty if none."""
    rows = conn.execute(
        "SELECT hand_class, count FROM opp_range_lookup "
        "WHERE street=? AND board_texture=? AND position=? "
        "AND spr_bucket=? AND action_context=?",
        (street, board_texture, position, spr_bucket, action_context),
    ).fetchall()
    return rows


def is_file_processed(conn: sqlite3.Connection,
                      path: str, mtime: float) -> bool:
    row = conn.execute(
        "SELECT mtime FROM opp_range_processed_files WHERE path=?",
        (path,)).fetchone()
    return row is not None and abs(row[0] - mtime) < 1e-6


def mark_file_processed(conn: sqlite3.Connection,
                        path: str, mtime: float, hand_count: int):
    conn.execute(
        "INSERT OR REPLACE INTO opp_range_processed_files "
        "(path, mtime, hand_count, processed_at) VALUES (?,?,?,?)",
        (path, mtime, hand_count,
         datetime.utcnow().isoformat(timespec='seconds')))
    conn.commit()


def summary(conn: sqlite3.Connection) -> dict:
    n_obs = conn.execute(
        "SELECT COUNT(*) FROM opp_range_observations").fetchone()[0]
    n_spots = conn.execute(
        "SELECT COUNT(DISTINCT street||board_texture||position||spr_bucket||"
        "action_context) FROM opp_range_lookup").fetchone()[0]
    n_lookup = conn.execute(
        "SELECT COUNT(*) FROM opp_range_lookup").fetchone()[0]
    n_files = conn.execute(
        "SELECT COUNT(*) FROM opp_range_processed_files").fetchone()[0]
    return {
        'n_observations': n_obs,
        'n_spots': n_spots,
        'n_lookup_rows': n_lookup,
        'n_files_processed': n_files,
    }
