"""Tests for src/postflop_sizing.py — runtime sizing + all-in promotion."""
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.postflop_db import connect, insert_decisions_bulk, rebuild_lookup
from src.postflop_sizing import (
    PostflopSizingLookup,
    FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN, RAISE_CUSTOM,
    _round_to_half_bb,
)
from src.postflop_spot_key import (
    HB_TOP_PAIR_HIGH, HB_TRIPS_SET, DT_NONE, BT_DRY,
    AC_CBET, AC_FACE_BET_MED,
    HAND_ONE_PAIR, HAND_TRIPS,
    IDX_MADE_HAND, IDX_TOP_PAIR, IDX_RANK_WITHIN_CAT,
)


def _obs(made_hand=0.0, top_pair=0.0, rank=0.5):
    o = np.zeros(53, dtype=np.float32)
    o[IDX_MADE_HAND] = made_hand
    o[IDX_TOP_PAIR] = top_pair
    o[IDX_RANK_WITHIN_CAT] = rank
    return o


def _row(spot_tuple, action='bet', size_pct=0.5, size_bb=2.5,
         is_all_in=0, pot_bb=5.0, spr=20.0):
    s, hb, dt, bt, pos, sb, ac = spot_tuple
    return (s, hb, dt, bt, pos, sb, ac, action, size_pct, size_bb,
            is_all_in, pot_bb, spr, 'p_x', 'h_x', 'f.txt')


@pytest.fixture
def seeded_db():
    """SQLite DB with three crafted spots covering common scenarios."""
    fd, path = tempfile.mkstemp(suffix='.sqlite')
    os.close(fd)
    conn = connect(path)

    rows = []

    # Spot A: top_pair_high, dry flop, IP, cbet — cohort bets ~50% pot
    spot_A = ('flop', HB_TOP_PAIR_HIGH, DT_NONE, BT_DRY, 'IP', '5-15', AC_CBET)
    for size in (0.33, 0.50, 0.50, 0.50, 0.50, 0.66, 0.66, 0.75):
        rows.append(_row(spot_A, action='bet', size_pct=size, size_bb=size * 5.0))

    # Spot B: trips, river, cohort SHOVES 70% of the time (all-in promotion test)
    spot_B = ('river', HB_TRIPS_SET, DT_NONE, BT_DRY, 'IP', '<2', AC_CBET)
    for _ in range(7):
        rows.append(_row(spot_B, action='all_in', size_pct=None, size_bb=80.0,
                         is_all_in=1, pot_bb=8.0, spr=1.5))
    for _ in range(3):
        rows.append(_row(spot_B, action='bet', size_pct=0.50, size_bb=4.0,
                         pot_bb=8.0, spr=1.5))

    # Spot C: only 2 samples — sparse, should fall through
    spot_sparse = ('flop', HB_TRIPS_SET, DT_NONE, BT_DRY, 'OOP', '15+', AC_CBET)
    for _ in range(2):
        rows.append(_row(spot_sparse, size_pct=0.5, size_bb=2.5))

    insert_decisions_bulk(conn, rows)
    rebuild_lookup(conn)
    conn.close()
    yield path
    os.unlink(path)


# ─── Sizing override path ────────────────────────────────────────────────

def test_sizing_override_picks_p25_for_raise_half(seeded_db):
    lookup = PostflopSizingLookup(seeded_db)
    inputs = dict(
        street='flop',
        obs=_obs(made_hand=HAND_ONE_PAIR, top_pair=1.0, rank=0.85),
        hole_cards=['As', 'Kh'], community_cards=['Kd', '7c', '2s'],
        rel_pos=0, num_active=2,
        effective_stack_bb=50, pot_bb=5,   # SPR=10 → '5-15' bucket (matches seed)
        was_preflop_raiser=True, facing_bet_size_pct=None,
        call_amount_bb=0.0,
    )
    new_action, size = lookup.adjust(RAISE_HALF, inputs)
    assert new_action == RAISE_HALF  # not promoted
    assert size is not None
    # p25 of cohort sizes (0.33 to 0.75) is around 0.33-0.50
    # Total bet = 0 + 0.33-0.5 * (5+0) = 1.65 to 2.5bb. Floor at min_raise=2.0.
    assert 2.0 <= size <= 3.5


def test_sizing_override_picks_p75_for_raise_pot(seeded_db):
    lookup = PostflopSizingLookup(seeded_db)
    inputs = dict(
        street='flop',
        obs=_obs(made_hand=HAND_ONE_PAIR, top_pair=1.0, rank=0.85),
        hole_cards=['As', 'Kh'], community_cards=['Kd', '7c', '2s'],
        rel_pos=0, num_active=2,
        effective_stack_bb=50, pot_bb=5,   # SPR=10 → '5-15' bucket (matches seed)
        was_preflop_raiser=True, facing_bet_size_pct=None,
        call_amount_bb=0.0,
    )
    new_action, size = lookup.adjust(RAISE_POT, inputs)
    assert new_action == RAISE_POT
    # p75 around 0.66-0.75 → 3.3-3.75bb (after rounding to 0.5)
    assert 3.0 <= size <= 4.5


# ─── All-in promotion ────────────────────────────────────────────────────

def test_all_in_promotion_when_cohort_shoves(seeded_db):
    """Spot B has 70% shove rate ≥ default threshold of 50% → promote."""
    lookup = PostflopSizingLookup(seeded_db, all_in_promotion_threshold=0.50)
    inputs = dict(
        street='river',
        obs=_obs(made_hand=HAND_TRIPS),
        hole_cards=['7s', '7h'], community_cards=['7d', 'Js', '2c', 'Qh', '4d'],
        rel_pos=0, num_active=2,
        effective_stack_bb=12, pot_bb=8,
        was_preflop_raiser=True, facing_bet_size_pct=None,
        call_amount_bb=0.0,
    )
    new_action, size = lookup.adjust(RAISE_HALF, inputs)
    # Promoted to ALL_IN with size = effective_stack_bb (=12, rounded)
    assert new_action == ALL_IN
    assert size == 12.0


def test_no_promotion_when_threshold_high(seeded_db):
    """Threshold 0.90 means 70% shove rate doesn't trigger promotion."""
    lookup = PostflopSizingLookup(seeded_db, all_in_promotion_threshold=0.90)
    inputs = dict(
        street='river',
        obs=_obs(made_hand=HAND_TRIPS),
        hole_cards=['7s', '7h'], community_cards=['7d', 'Js', '2c', 'Qh', '4d'],
        rel_pos=0, num_active=2,
        effective_stack_bb=12, pot_bb=8,
        was_preflop_raiser=True, facing_bet_size_pct=None,
        call_amount_bb=0.0,
    )
    new_action, size = lookup.adjust(RAISE_POT, inputs)
    assert new_action == RAISE_POT
    # Sizing override fires (not promoted); size from p75 of non-shove samples


# ─── Sparse / no data path ───────────────────────────────────────────────

def test_sparse_spot_returns_unchanged(seeded_db):
    lookup = PostflopSizingLookup(seeded_db)
    inputs = dict(
        street='flop',
        obs=_obs(made_hand=HAND_TRIPS),
        hole_cards=['7s', '7h'], community_cards=['7d', 'Js', '2c'],
        rel_pos=2, num_active=6,
        effective_stack_bb=100, pot_bb=2,
        was_preflop_raiser=True, facing_bet_size_pct=None,
        call_amount_bb=0.0,
    )
    # Spot maps to (flop, trips_set, none, dry, OOP, 15+, cbet) — only 2 samples
    new_action, size = lookup.adjust(RAISE_POT, inputs)
    assert new_action == RAISE_POT
    assert size is None   # below threshold


def test_no_data_at_all_returns_unchanged(seeded_db):
    lookup = PostflopSizingLookup(seeded_db)
    inputs = dict(
        street='turn',
        obs=_obs(made_hand=0.0),
        hole_cards=['9d', '4s'], community_cards=['Kd', '7c', '2s', 'Jh'],
        rel_pos=2, num_active=6,
        effective_stack_bb=100, pot_bb=5,
        was_preflop_raiser=False, facing_bet_size_pct=None,
        call_amount_bb=0.0,
    )
    new_action, size = lookup.adjust(RAISE_HALF, inputs)
    assert new_action == RAISE_HALF
    assert size is None


# ─── Non-applicable cases ────────────────────────────────────────────────

def test_fold_returns_unchanged(seeded_db):
    lookup = PostflopSizingLookup(seeded_db)
    inputs = dict(
        street='flop', obs=_obs(), hole_cards=None, community_cards=None,
        rel_pos=0, num_active=2, effective_stack_bb=100, pot_bb=5,
        was_preflop_raiser=True, facing_bet_size_pct=None,
        call_amount_bb=0.0,
    )
    new_action, size = lookup.adjust(FOLD, inputs)
    assert new_action == FOLD
    assert size is None


def test_call_returns_unchanged(seeded_db):
    lookup = PostflopSizingLookup(seeded_db)
    inputs = dict(
        street='flop', obs=_obs(), hole_cards=None, community_cards=None,
        rel_pos=0, num_active=2, effective_stack_bb=100, pot_bb=5,
        was_preflop_raiser=True, facing_bet_size_pct=None,
        call_amount_bb=0.0,
    )
    new_action, size = lookup.adjust(CALL, inputs)
    assert new_action == CALL
    assert size is None


def test_preflop_returns_unchanged(seeded_db):
    """Module is postflop-only; preflop streets are skipped."""
    lookup = PostflopSizingLookup(seeded_db)
    inputs = dict(
        street='preflop', obs=_obs(), hole_cards=['As', 'Ah'],
        community_cards=[],
        rel_pos=0, num_active=2, effective_stack_bb=100, pot_bb=1.5,
        was_preflop_raiser=False, facing_bet_size_pct=None,
        call_amount_bb=1.0,
    )
    new_action, size = lookup.adjust(RAISE_POT, inputs)
    assert new_action == RAISE_POT
    assert size is None


# ─── Rounding helper ─────────────────────────────────────────────────────

def test_round_to_half_bb():
    assert _round_to_half_bb(2.42) == 2.5
    assert _round_to_half_bb(11.43) == 11.5
    assert _round_to_half_bb(23.83) == 24.0
    assert _round_to_half_bb(2.0) == 2.0
    assert _round_to_half_bb(1.5) == 2.0   # min floor


# ─── Counters ────────────────────────────────────────────────────────────

def test_counters_track_correctly(seeded_db):
    lookup = PostflopSizingLookup(seeded_db)
    base = dict(
        street='flop',
        obs=_obs(made_hand=HAND_ONE_PAIR, top_pair=1.0, rank=0.85),
        hole_cards=['As', 'Kh'], community_cards=['Kd', '7c', '2s'],
        rel_pos=0, num_active=2,
        effective_stack_bb=50, pot_bb=5,   # SPR=10 → '5-15' bucket (matches seed)
        was_preflop_raiser=True, facing_bet_size_pct=None,
        call_amount_bb=0.0,
    )
    lookup.adjust(RAISE_POT, base)             # sizing override
    lookup.adjust(FOLD, base)                  # non-raise
    lookup.adjust(RAISE_HALF, {**base, 'street': 'preflop'})  # non-postflop
    c = lookup.counters
    assert c['sizing_override'] >= 1
    assert c['non_raise'] >= 1
    assert c['non_postflop'] >= 1


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
