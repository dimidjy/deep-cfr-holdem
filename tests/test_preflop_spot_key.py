"""Tests for src/preflop_spot_key.py — locks the spot-keying contract.

This module is the single source of truth used by both the offline build
script and the runtime lookup. Any divergence between build and runtime
will silently produce wrong strategies. Every enum value and edge case is
covered here.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preflop_spot_key import (
    cards_to_class, position_label, stack_bucket, num_to_act_bucket,
    derive_action_history, build_spot_key, SpotKey,
    HIST_OPEN, HIST_LIMPED_POT, HIST_VS_RAISE, HIST_VS_3BET, HIST_VS_4BET,
)


# ─── cards_to_class ──────────────────────────────────────────────────────

def test_cards_to_class_pair():
    assert cards_to_class('As', 'Ah') == 'AA'
    assert cards_to_class('2c', '2d') == '22'


def test_cards_to_class_suited():
    assert cards_to_class('As', 'Ks') == 'AKS'
    assert cards_to_class('Th', '7h') == 'T7S'
    # Order of inputs doesn't matter
    assert cards_to_class('Ks', 'As') == 'AKS'


def test_cards_to_class_offsuit():
    assert cards_to_class('As', 'Kh') == 'AKO'
    assert cards_to_class('7d', '2c') == '72O'


# ─── position_label ──────────────────────────────────────────────────────

def test_position_label_6max():
    # Existing range JSON uses LJ/HJ/CO, NOT UTG/MP/CO. Lock that here.
    assert position_label(0, 6) == 'BTN'
    assert position_label(1, 6) == 'SB'
    assert position_label(2, 6) == 'BB'
    assert position_label(3, 6) == 'LJ'
    assert position_label(4, 6) == 'HJ'
    assert position_label(5, 6) == 'CO'


def test_position_label_heads_up():
    assert position_label(0, 2) == 'BTN'
    assert position_label(1, 2) == 'BB'


def test_position_label_3max():
    assert position_label(0, 3) == 'BTN'
    assert position_label(1, 3) == 'SB'
    assert position_label(2, 3) == 'BB'


def test_position_label_unknown_falls_back():
    # Anything outside the documented buckets should still produce SOMETHING
    assert position_label(99, 6).startswith('P')


# ─── stack_bucket ────────────────────────────────────────────────────────

def test_stack_bucket_short():
    assert stack_bucket(20.0) == '<40'
    assert stack_bucket(39.9) == '<40'


def test_stack_bucket_mid():
    assert stack_bucket(40.0) == '40-70'
    assert stack_bucket(60.0) == '40-70'


def test_stack_bucket_standard():
    assert stack_bucket(100.0) == '70-150'
    assert stack_bucket(149.9) == '70-150'


def test_stack_bucket_deep():
    assert stack_bucket(150.0) == '150+'
    assert stack_bucket(500.0) == '150+'


# ─── num_to_act_bucket ───────────────────────────────────────────────────

def test_num_to_act_bucket_clamp():
    assert num_to_act_bucket(0) == 0
    assert num_to_act_bucket(3) == 3
    assert num_to_act_bucket(4) == 4
    assert num_to_act_bucket(5) == 4   # clamped
    assert num_to_act_bucket(99) == 4  # clamped
    assert num_to_act_bucket(-1) == 0  # safety


# ─── derive_action_history ───────────────────────────────────────────────

def test_action_history_open():
    """No prior voluntary action → 'open'."""
    assert derive_action_history([]) == HIST_OPEN


def test_action_history_open_after_folds():
    """Folds don't change the spot category — still 'open'."""
    history = [('fold', None), ('fold', None), ('fold', None)]
    assert derive_action_history(history) == HIST_OPEN


def test_action_history_limped_pot():
    """One limp before us, no raises → 'limped_pot'."""
    history = [('fold', None), ('call', 1.0), ('fold', None)]
    assert derive_action_history(history) == HIST_LIMPED_POT


def test_action_history_multi_limp():
    history = [('call', 1.0), ('call', 1.0), ('fold', None)]
    assert derive_action_history(history) == HIST_LIMPED_POT


def test_action_history_vs_raise():
    history = [('fold', None), ('raise', 2.5), ('fold', None)]
    assert derive_action_history(history) == HIST_VS_RAISE


def test_action_history_vs_raise_with_caller():
    """An open + a cold-call before us is still 'vs_raise' (not limped)."""
    history = [('raise', 2.5), ('call', 2.5), ('fold', None)]
    assert derive_action_history(history) == HIST_VS_RAISE


def test_action_history_vs_3bet():
    history = [('raise', 2.5), ('raise', 9.0)]
    assert derive_action_history(history) == HIST_VS_3BET


def test_action_history_vs_4bet():
    history = [('raise', 2.5), ('raise', 9.0), ('raise', 22.0)]
    assert derive_action_history(history) == HIST_VS_4BET


def test_action_history_5bet_collapses_to_4bet():
    history = [('raise', 2.5), ('raise', 9.0), ('raise', 22.0), ('raise', 55.0)]
    assert derive_action_history(history) == HIST_VS_4BET


def test_action_history_limp_then_raise_is_vs_raise():
    """If someone limps and someone behind raises, the new spot is vs_raise,
    not limped_pot — the raise is the dominant signal."""
    history = [('call', 1.0), ('raise', 4.0)]
    assert derive_action_history(history) == HIST_VS_RAISE


def test_action_history_call_after_raise_is_not_limp():
    """A 'call' after a raise is a cold-call, not a limp — so this is still
    vs_raise (1 raise count), not limped_pot."""
    history = [('raise', 2.5), ('call', 2.5)]
    assert derive_action_history(history) == HIST_VS_RAISE


# ─── build_spot_key end-to-end ───────────────────────────────────────────

def test_build_spot_key_btn_open_aa_100bb():
    key = build_spot_key(
        hole_cards=['As', 'Ah'],
        rel_pos=0, num_players=6,
        prior_actions=[('fold', None)] * 5,
        effective_bb=100.0,
        num_to_act=0,
    )
    assert key == SpotKey(
        hole_class='AA',
        position='BTN',
        action_history=HIST_OPEN,
        stack_bucket='70-150',
        num_to_act=0,
    )


def test_build_spot_key_co_vs_3bet_aks_short():
    key = build_spot_key(
        hole_cards=['As', 'Ks'],
        rel_pos=5, num_players=6,
        prior_actions=[('raise', 2.5), ('raise', 9.0)],
        effective_bb=35.0,
        num_to_act=2,
    )
    assert key.hole_class == 'AKS'
    assert key.position == 'CO'
    assert key.action_history == HIST_VS_3BET
    assert key.stack_bucket == '<40'
    assert key.num_to_act == 2


def test_spot_key_to_tuple_is_hashable():
    """Required for SQLite primary key / dict lookup."""
    key = build_spot_key(
        hole_cards=['As', 'Ah'],
        rel_pos=0, num_players=6,
        prior_actions=[],
        effective_bb=100.0,
        num_to_act=5,  # clamps to 4
    )
    t = key.to_tuple()
    assert t == ('AA', 'BTN', HIST_OPEN, '70-150', 4)
    # Hashable for use as dict key
    assert hash(t) == hash(('AA', 'BTN', HIST_OPEN, '70-150', 4))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
