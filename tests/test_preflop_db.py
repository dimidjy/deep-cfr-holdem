"""Tests for src/preflop_db.py — SQLite I/O round-trip + aggregation."""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preflop_db import (
    connect, insert_decisions_bulk, rebuild_lookup, query_spot,
    load_lookup_into_memory, is_file_processed, mark_file_processed,
    lookup_summary,
)
from src.preflop_spot_key import SpotKey, HIST_OPEN, HIST_VS_RAISE


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix='.sqlite')
    os.close(fd)
    conn = connect(path)
    yield conn, path
    conn.close()
    os.unlink(path)


def _make_rows(spot_tuple, *, folds=0, calls=0, raises_with_sizes=()):
    """Build raw decision rows for one spot."""
    hc, pos, hist, sb, nta = spot_tuple
    rows = []
    for i in range(folds):
        rows.append((hc, pos, hist, sb, nta, f'p_f{i}', 'fold', None,
                     1.5, f'h{i}', 'f.txt'))
    for i in range(calls):
        rows.append((hc, pos, hist, sb, nta, f'p_c{i}', 'call', None,
                     1.5, f'h{i}', 'f.txt'))
    for i, size in enumerate(raises_with_sizes):
        rows.append((hc, pos, hist, sb, nta, f'p_r{i}', 'raise', size,
                     1.5, f'h{i}', 'f.txt'))
    return rows


def test_insert_and_query_roundtrip(tmp_db):
    conn, _ = tmp_db
    spot = ('AA', 'BTN', HIST_OPEN, '70-150', 0)
    rows = _make_rows(spot, folds=1, calls=4, raises_with_sizes=(2.5, 3.0, 2.5, 2.5, 2.5))
    n = insert_decisions_bulk(conn, rows)
    assert n == 10
    assert rebuild_lookup(conn) == 1

    res = query_spot(conn, SpotKey(*spot))
    assert res is not None
    assert res['n_samples'] == 10
    assert abs(res['fold_pct'] - 10.0) < 0.01
    assert abs(res['call_pct'] - 40.0) < 0.01
    assert abs(res['raise_pct'] - 50.0) < 0.01
    assert abs(res['raise_size_bb_p50'] - 2.5) < 0.01
    # mean of (2.5, 3.0, 2.5, 2.5, 2.5) = 2.6
    assert abs(res['raise_size_bb_mean'] - 2.6) < 0.01


def test_query_spot_missing_returns_none(tmp_db):
    conn, _ = tmp_db
    rebuild_lookup(conn)  # empty rebuild
    assert query_spot(conn, SpotKey('AA', 'BTN', HIST_OPEN, '70-150', 0)) is None


def test_load_lookup_into_memory(tmp_db):
    conn, _ = tmp_db
    spot1 = ('AA', 'BTN', HIST_OPEN, '70-150', 0)
    spot2 = ('AKS', 'CO', HIST_VS_RAISE, '70-150', 1)
    insert_decisions_bulk(conn, _make_rows(spot1, raises_with_sizes=(3.0, 3.0)))
    insert_decisions_bulk(conn, _make_rows(spot2, calls=3, raises_with_sizes=(8.0,)))
    rebuild_lookup(conn)

    cache = load_lookup_into_memory(conn)
    assert spot1 in cache
    assert spot2 in cache
    assert cache[spot1]['raise_pct'] == 100.0
    assert cache[spot2]['n_samples'] == 4


def test_rebuild_replaces_old_data(tmp_db):
    conn, _ = tmp_db
    spot = ('AA', 'BTN', HIST_OPEN, '70-150', 0)
    insert_decisions_bulk(conn, _make_rows(spot, raises_with_sizes=(3.0,)))
    rebuild_lookup(conn)
    assert query_spot(conn, SpotKey(*spot))['n_samples'] == 1

    # Add more decisions, rebuild — old aggregate should be replaced
    insert_decisions_bulk(conn, _make_rows(spot, raises_with_sizes=(3.0, 3.0, 3.0)))
    rebuild_lookup(conn)
    assert query_spot(conn, SpotKey(*spot))['n_samples'] == 4


def test_processed_files_tracking(tmp_db):
    conn, _ = tmp_db
    assert not is_file_processed(conn, '/tmp/x.txt', 100.0)
    mark_file_processed(conn, '/tmp/x.txt', 100.0, hand_count=42)
    assert is_file_processed(conn, '/tmp/x.txt', 100.0)
    # Different mtime → not processed
    assert not is_file_processed(conn, '/tmp/x.txt', 200.0)


def test_percentages_sum_to_100(tmp_db):
    conn, _ = tmp_db
    spot = ('AA', 'BTN', HIST_OPEN, '70-150', 0)
    insert_decisions_bulk(conn, _make_rows(spot, folds=3, calls=5, raises_with_sizes=(3.0, 3.0)))
    rebuild_lookup(conn)
    r = query_spot(conn, SpotKey(*spot))
    assert abs(r['fold_pct'] + r['call_pct'] + r['raise_pct'] - 100.0) < 0.01


def test_lookup_summary_smoke(tmp_db):
    conn, _ = tmp_db
    insert_decisions_bulk(conn, _make_rows(
        ('AA', 'BTN', HIST_OPEN, '70-150', 0), raises_with_sizes=(3.0,)))
    rebuild_lookup(conn)
    s = lookup_summary(conn)
    assert s['n_decisions'] == 1
    assert s['n_spots'] == 1
    assert len(s['top_spots_by_samples']) == 1


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
