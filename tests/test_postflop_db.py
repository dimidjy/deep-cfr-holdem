"""Tests for src/postflop_db.py — SQLite I/O round-trip + aggregation."""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.postflop_db import (
    connect, insert_decisions_bulk, rebuild_lookup, query_spot,
    load_lookup_into_memory, is_file_processed, mark_file_processed,
    lookup_summary, export_lookup_to_json,
)
from src.postflop_spot_key import (
    PostflopSpotKey, HB_TOP_PAIR_HIGH, HB_TRIPS_SET, DT_NONE,
    BT_DRY, BT_TWO_TONE, AC_CBET, AC_FACE_BET_MED,
)


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix='.sqlite')
    os.close(fd)
    conn = connect(path)
    yield conn, path
    conn.close()
    os.unlink(path)


def _row(spot, action='bet', size_pct=0.5, size_bb=2.5, is_all_in=0,
         pot_bb=5.0, spr=20.0):
    """Build a `postflop_decisions` row tuple."""
    s, hb, dt, bt, pos, spr_b, ac = spot
    return (s, hb, dt, bt, pos, spr_b, ac, action, size_pct, size_bb,
            is_all_in, pot_bb, spr, 'p_x', 'h_x', 'f.txt')


def test_insert_query_roundtrip(tmp_db):
    conn, _ = tmp_db
    spot = ('flop', HB_TOP_PAIR_HIGH, DT_NONE, BT_DRY, 'IP', '5-15', AC_CBET)
    rows = []
    # 8 cohort cbets at p50=50% pot
    for size in (0.33, 0.50, 0.50, 0.50, 0.66, 0.66, 0.75, 1.00):
        rows.append(_row(spot, action='bet', size_pct=size, size_bb=size * 5))
    n = insert_decisions_bulk(conn, rows)
    assert n == 8
    assert rebuild_lookup(conn) == 1

    res = query_spot(conn, PostflopSpotKey(*spot))
    assert res is not None
    assert res['n_samples'] == 8
    assert res['all_in_pct'] == 0.0
    # Median size is 50% pot
    assert abs(res['bet_size_pct_p50'] - 0.66) < 0.05 or abs(res['bet_size_pct_p50'] - 0.50) < 0.05


def test_all_in_percentage(tmp_db):
    conn, _ = tmp_db
    spot = ('river', HB_TRIPS_SET, DT_NONE, BT_DRY, 'IP', '<2', AC_CBET)
    rows = []
    for _ in range(7):
        rows.append(_row(spot, action='all_in', size_pct=None,
                         size_bb=100.0, is_all_in=1))
    for _ in range(3):
        rows.append(_row(spot, action='bet', size_pct=0.50, size_bb=2.5))
    insert_decisions_bulk(conn, rows)
    rebuild_lookup(conn)

    res = query_spot(conn, PostflopSpotKey(*spot))
    assert res['n_samples'] == 10
    assert res['n_all_in'] == 7
    assert abs(res['all_in_pct'] - 70.0) < 0.01


def test_query_missing_spot_returns_none(tmp_db):
    conn, _ = tmp_db
    rebuild_lookup(conn)
    spot = ('flop', HB_TOP_PAIR_HIGH, DT_NONE, BT_DRY, 'IP', '5-15', AC_CBET)
    assert query_spot(conn, PostflopSpotKey(*spot)) is None


def test_load_lookup_into_memory(tmp_db):
    conn, _ = tmp_db
    spot1 = ('flop', HB_TOP_PAIR_HIGH, DT_NONE, BT_DRY, 'IP', '5-15', AC_CBET)
    spot2 = ('turn', HB_TRIPS_SET, DT_NONE, BT_TWO_TONE, 'OOP', '2-5',
             AC_FACE_BET_MED)
    insert_decisions_bulk(conn, [
        _row(spot1, size_pct=0.5, size_bb=2.5),
        _row(spot1, size_pct=0.5, size_bb=2.5),
        _row(spot2, action='raise', size_pct=2.5, size_bb=12.5),
    ])
    rebuild_lookup(conn)
    cache = load_lookup_into_memory(conn)
    assert spot1 in cache and spot2 in cache
    assert cache[spot1]['n_samples'] == 2
    assert cache[spot2]['n_samples'] == 1


def test_processed_files_tracking(tmp_db):
    conn, _ = tmp_db
    assert not is_file_processed(conn, '/tmp/x.txt', 100.0)
    mark_file_processed(conn, '/tmp/x.txt', 100.0, hand_count=42)
    assert is_file_processed(conn, '/tmp/x.txt', 100.0)
    assert not is_file_processed(conn, '/tmp/x.txt', 200.0)


def test_rebuild_replaces_data(tmp_db):
    conn, _ = tmp_db
    spot = ('flop', HB_TOP_PAIR_HIGH, DT_NONE, BT_DRY, 'IP', '5-15', AC_CBET)
    insert_decisions_bulk(conn, [_row(spot)])
    rebuild_lookup(conn)
    assert query_spot(conn, PostflopSpotKey(*spot))['n_samples'] == 1
    insert_decisions_bulk(conn, [_row(spot), _row(spot), _row(spot)])
    rebuild_lookup(conn)
    assert query_spot(conn, PostflopSpotKey(*spot))['n_samples'] == 4


def test_lookup_summary(tmp_db):
    conn, _ = tmp_db
    spot = ('flop', HB_TOP_PAIR_HIGH, DT_NONE, BT_DRY, 'IP', '5-15', AC_CBET)
    insert_decisions_bulk(conn, [_row(spot, size_pct=0.5, size_bb=2.5)])
    rebuild_lookup(conn)
    s = lookup_summary(conn)
    assert s['n_decisions'] == 1
    assert s['n_spots'] == 1
    assert len(s['top_spots_by_samples']) == 1


def test_export_to_json(tmp_db):
    conn, _ = tmp_db
    spot = ('flop', HB_TOP_PAIR_HIGH, DT_NONE, BT_DRY, 'IP', '5-15', AC_CBET)
    insert_decisions_bulk(conn, [_row(spot)] * 5)
    rebuild_lookup(conn)
    fd, jpath = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    try:
        n = export_lookup_to_json(conn, jpath)
        assert n == 1
        import json
        with open(jpath) as f:
            data = json.load(f)
        assert data['n_total'] == 1
        assert data['spots'][0]['n_samples'] == 5
    finally:
        os.unlink(jpath)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
