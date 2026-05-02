"""Tests for src/postflop_spot_key.py — locks the postflop spot-keying contract.

Build/runtime divergence is the #1 production-bug source for this kind of
lookup, so every enum value should be reachable from a synthetic obs vector
or a hand-crafted card sequence.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.postflop_spot_key import (
    classify_hand_bucket, classify_draw_type, classify_board_texture,
    classify_position, classify_spr, classify_action_context, build_spot_key,
    PostflopSpotKey,
    HB_HIGH_CARD, HB_LOW_PAIR, HB_MID_PAIR, HB_TOP_PAIR_LOW, HB_TOP_PAIR_HIGH,
    HB_OVERPAIR, HB_TWO_PAIR, HB_TRIPS_SET, HB_STRAIGHT, HB_FLUSH, HB_FULL_PLUS,
    DT_NONE, DT_FLUSH_DRAW, DT_OESD, DT_GUTSHOT, DT_COMBO_DRAW, DT_MADE_HAND,
    BT_DRY, BT_TWO_TONE, BT_PAIRED, BT_MONOTONE, BT_WET_PAIRED,
    AC_CBET, AC_PROBE, AC_DONK,
    AC_FACE_BET_SMALL, AC_FACE_BET_MED, AC_FACE_BET_LARGE, AC_FACE_CHECK_RAISE,
    HAND_HIGH_CARD, HAND_ONE_PAIR, HAND_TWO_PAIR, HAND_TRIPS,
    HAND_STRAIGHT, HAND_FLUSH, HAND_FULL_HOUSE, HAND_QUADS,
    IDX_MADE_HAND, IDX_FLUSH_DRAW, IDX_STRAIGHT_DRAW, IDX_TOP_PAIR,
    IDX_BOARD_PAIR, IDX_BOARD_FLUSH, IDX_BOARD_STRAIGHT, IDX_RANK_WITHIN_CAT,
)


def _obs(made_hand=0.0, flush_draw=0.0, straight_draw=0.0, top_pair=0.0,
         board_pair=0.0, board_flush=0.0, board_straight=0.0,
         rank_within_cat=0.5):
    """Build a synthetic obs vector with only the equity-block features set."""
    o = np.zeros(53, dtype=np.float32)
    o[IDX_MADE_HAND] = made_hand
    o[IDX_FLUSH_DRAW] = flush_draw
    o[IDX_STRAIGHT_DRAW] = straight_draw
    o[IDX_TOP_PAIR] = top_pair
    o[IDX_BOARD_PAIR] = board_pair
    o[IDX_BOARD_FLUSH] = board_flush
    o[IDX_BOARD_STRAIGHT] = board_straight
    o[IDX_RANK_WITHIN_CAT] = rank_within_cat
    return o


# ─── classify_hand_bucket ───────────────────────────────────────────────

def test_hand_bucket_high_card():
    assert classify_hand_bucket(_obs(made_hand=0.0)) == HB_HIGH_CARD


def test_hand_bucket_top_pair_high():
    assert classify_hand_bucket(_obs(made_hand=HAND_ONE_PAIR,
                                     top_pair=1.0, rank_within_cat=0.85)) == HB_TOP_PAIR_HIGH


def test_hand_bucket_top_pair_low():
    assert classify_hand_bucket(_obs(made_hand=HAND_ONE_PAIR,
                                     top_pair=1.0, rank_within_cat=0.20)) == HB_TOP_PAIR_LOW


def test_hand_bucket_overpair():
    assert classify_hand_bucket(_obs(made_hand=HAND_ONE_PAIR,
                                     top_pair=0.0, rank_within_cat=0.85)) == HB_OVERPAIR


def test_hand_bucket_mid_pair():
    assert classify_hand_bucket(_obs(made_hand=HAND_ONE_PAIR,
                                     top_pair=0.0, rank_within_cat=0.45)) == HB_MID_PAIR


def test_hand_bucket_low_pair():
    assert classify_hand_bucket(_obs(made_hand=HAND_ONE_PAIR,
                                     top_pair=0.0, rank_within_cat=0.10)) == HB_LOW_PAIR


def test_hand_bucket_two_pair():
    assert classify_hand_bucket(_obs(made_hand=HAND_TWO_PAIR)) == HB_TWO_PAIR


def test_hand_bucket_trips_set():
    assert classify_hand_bucket(_obs(made_hand=HAND_TRIPS)) == HB_TRIPS_SET


def test_hand_bucket_straight():
    assert classify_hand_bucket(_obs(made_hand=HAND_STRAIGHT)) == HB_STRAIGHT


def test_hand_bucket_flush():
    assert classify_hand_bucket(_obs(made_hand=HAND_FLUSH)) == HB_FLUSH


def test_hand_bucket_full_plus():
    assert classify_hand_bucket(_obs(made_hand=HAND_FULL_HOUSE)) == HB_FULL_PLUS
    assert classify_hand_bucket(_obs(made_hand=HAND_QUADS)) == HB_FULL_PLUS


# ─── classify_draw_type ─────────────────────────────────────────────────

def test_draw_type_none():
    obs = _obs(made_hand=HAND_HIGH_CARD)
    assert classify_draw_type(obs, ['Ks', '7d'], ['2c', '5h', '9s']) == DT_NONE


def test_draw_type_made_hand_short_circuits():
    obs = _obs(made_hand=HAND_FLUSH)
    # Even with cards suggesting drawing structure, made hand wins
    assert classify_draw_type(obs, ['Ah', 'Kh'], ['Qh', '7h', '2h']) == DT_MADE_HAND


def test_draw_type_flush_draw():
    obs = _obs(made_hand=HAND_HIGH_CARD)
    # Hero Ah2h, board QhJs7h → 4 hearts, no straight draw (A-2 don't connect with Q-J-7)
    assert classify_draw_type(obs, ['Ah', '2h'], ['Qh', 'Js', '7h']) == DT_FLUSH_DRAW


def test_draw_type_oesd():
    obs = _obs(made_hand=HAND_HIGH_CARD)
    # Hero JT, board 9-8-2 rainbow → 4 outs to 7 (low end), 4 outs to Q (high end) = OESD
    assert classify_draw_type(obs, ['Jh', 'Td'], ['9c', '8s', '2d']) == DT_OESD


def test_draw_type_gutshot():
    obs = _obs(made_hand=HAND_HIGH_CARD)
    # Hero JT, board 8-7-2 → only 9 fills (J-T-9-8-7), 4 outs = gutshot
    assert classify_draw_type(obs, ['Jh', 'Td'], ['8c', '7s', '2d']) == DT_GUTSHOT


def test_draw_type_combo():
    obs = _obs(made_hand=HAND_HIGH_CARD)
    # Hero JhTh, board 9h8s2c → flush draw (4 hearts) + OESD (J-T-9-8-?)
    # Check: hearts = JhTh + 9h = 3 in our cards + need 4 total. Board: 9h, 8s, 2c.
    # Total hearts: Jh, Th, 9h = 3. Not 4-flush. Let me adjust.
    # Hero JhTh, board 9h-7h-2c → hearts = Jh, Th, 9h, 7h = 4 → flush draw, plus
    # straight: J-T-9-?-7 needs 8 = OESD-like (one inner gap = gutshot)
    # Better: hero JhTh, board Qh-9h-8s → hearts: 3 (Jh, Th, Qh) — no flush draw.
    # Use: hero 9h8h, board 7h-6c-Qs. Hearts: 9h+8h+7h = 3. NOT flush draw.
    # Tricky to construct combo with only 3 community cards. Try hero KhQh, board JhTd9h:
    # Hearts: Kh, Qh, Jh, 9h = 4! Flush draw. Straight: K-Q-J-T-? → need 9 or A. 9 is here too. So already a straight!
    # Hmm — that becomes made hand. Try hero Th9h, board Jh-8c-Qs:
    # Hearts: Th, 9h, Jh = 3 → NOT flush draw.
    # Combo on flop is genuinely rare. Use turn (4 community cards):
    # Hero KhQh, board Jh-Td-3h-2c → hearts 3 (Kh,Qh,Jh,3h)=4 → flush draw. Straight: KQJT? need 9 or A → OESD
    obs = _obs(made_hand=HAND_HIGH_CARD)
    # Note: turn version, 4 community cards
    assert classify_draw_type(obs, ['Kh', 'Qh'], ['Jh', 'Td', '3h', '2c']) == DT_COMBO_DRAW


def test_draw_type_no_data_fallback_to_features():
    """Without raw cards, gutshot collapses into oesd via the feature-based path."""
    obs = _obs(made_hand=HAND_HIGH_CARD, straight_draw=0.5)
    # No hole_cards/community_cards → feature fallback. Treats both gutshot and OESD as 'oesd'.
    assert classify_draw_type(obs, None, None) == DT_OESD


def test_draw_type_river_returns_none():
    obs = _obs(made_hand=HAND_HIGH_CARD)
    assert classify_draw_type(obs, ['Ks', '7d'],
                              ['2c', '5h', '9s', 'Jd', '3c']) == DT_NONE


# ─── classify_board_texture ────────────────────────────────────────────

def test_board_texture_dry():
    assert classify_board_texture(_obs()) == BT_DRY


def test_board_texture_two_tone():
    assert classify_board_texture(_obs(board_flush=0.5)) == BT_TWO_TONE


def test_board_texture_paired():
    assert classify_board_texture(_obs(board_pair=0.33)) == BT_PAIRED


def test_board_texture_monotone():
    assert classify_board_texture(_obs(board_flush=1.0)) == BT_MONOTONE


def test_board_texture_wet_paired():
    assert classify_board_texture(_obs(board_pair=0.33, board_flush=0.5)) == BT_WET_PAIRED


def test_board_texture_wet_via_straight_texture():
    # Straight-heavy board with no pair counts as two_tone
    assert classify_board_texture(_obs(board_straight=0.7)) == BT_TWO_TONE


# ─── classify_position ─────────────────────────────────────────────────

def test_position_btn_is_ip():
    assert classify_position(rel_pos=0, num_active=6) == 'IP'


def test_position_other_is_oop():
    assert classify_position(rel_pos=1, num_active=6) == 'OOP'   # SB
    assert classify_position(rel_pos=2, num_active=6) == 'OOP'   # BB
    assert classify_position(rel_pos=5, num_active=6) == 'OOP'   # CO


def test_position_with_explicit_to_act():
    """When num_to_act_after_hero is given, it overrides rel_pos heuristic."""
    assert classify_position(rel_pos=5, num_active=6,
                             num_to_act_after_hero_postflop=0) == 'IP'
    assert classify_position(rel_pos=0, num_active=2,
                             num_to_act_after_hero_postflop=1) == 'OOP'


def test_position_hu():
    """In HU dead-button: BTN/SB still IP, BB still OOP."""
    assert classify_position(rel_pos=0, num_active=2) == 'IP'
    assert classify_position(rel_pos=1, num_active=2) == 'OOP'


# ─── classify_spr ──────────────────────────────────────────────────────

def test_spr_short():
    assert classify_spr(effective_stack_bb=10, pot_bb=8) == '<2'


def test_spr_med():
    assert classify_spr(effective_stack_bb=20, pot_bb=5) == '<2' or \
           classify_spr(effective_stack_bb=20, pot_bb=5) == '2-5'
    # 20/5 = 4 → 2-5
    assert classify_spr(effective_stack_bb=20, pot_bb=5) == '2-5'


def test_spr_standard():
    assert classify_spr(effective_stack_bb=100, pot_bb=10) == '5-15'


def test_spr_deep():
    assert classify_spr(effective_stack_bb=150, pot_bb=5) == '15+'


def test_spr_zero_pot():
    """Pot of 0 (degenerate) clamps to 15+."""
    assert classify_spr(effective_stack_bb=100, pot_bb=0) == '15+'


# ─── classify_action_context ───────────────────────────────────────────

def test_action_context_cbet():
    assert classify_action_context(was_preflop_raiser=True, position_is_oop=False,
                                    facing_bet_size_pct=None,
                                    facing_check_raise=False) == AC_CBET


def test_action_context_probe():
    assert classify_action_context(was_preflop_raiser=False, position_is_oop=False,
                                    facing_bet_size_pct=None,
                                    facing_check_raise=False) == AC_PROBE


def test_action_context_donk():
    assert classify_action_context(was_preflop_raiser=False, position_is_oop=True,
                                    facing_bet_size_pct=None,
                                    facing_check_raise=False) == AC_DONK


def test_action_context_face_bet_small():
    assert classify_action_context(was_preflop_raiser=False, position_is_oop=True,
                                    facing_bet_size_pct=0.30,
                                    facing_check_raise=False) == AC_FACE_BET_SMALL


def test_action_context_face_bet_med():
    assert classify_action_context(was_preflop_raiser=False, position_is_oop=False,
                                    facing_bet_size_pct=0.60,
                                    facing_check_raise=False) == AC_FACE_BET_MED


def test_action_context_face_bet_large():
    assert classify_action_context(was_preflop_raiser=False, position_is_oop=False,
                                    facing_bet_size_pct=1.20,
                                    facing_check_raise=False) == AC_FACE_BET_LARGE


def test_action_context_face_check_raise():
    assert classify_action_context(was_preflop_raiser=True, position_is_oop=False,
                                    facing_bet_size_pct=2.0,
                                    facing_check_raise=True) == AC_FACE_CHECK_RAISE


# ─── build_spot_key end-to-end ─────────────────────────────────────────

def test_build_spot_key_top_pair_high_dry_cbet():
    obs = _obs(made_hand=HAND_ONE_PAIR, top_pair=1.0, rank_within_cat=0.85)
    key = build_spot_key(
        street='flop', obs=obs,
        hole_cards=['As', 'Kh'], community_cards=['Kd', '7c', '2s'],
        rel_pos=0, num_active=2,
        effective_stack_bb=95, pot_bb=5,
        was_preflop_raiser=True, facing_bet_size_pct=None,
    )
    assert key == PostflopSpotKey(
        street='flop',
        hand_bucket=HB_TOP_PAIR_HIGH,
        draw_type=DT_NONE,
        board_texture=BT_DRY,
        position='IP',
        spr_bucket='15+',
        action_context=AC_CBET,
    )


def test_build_spot_key_combo_draw_facing_pot_bet():
    obs = _obs(made_hand=HAND_HIGH_CARD)
    key = build_spot_key(
        street='turn', obs=obs,
        hole_cards=['Kh', 'Qh'], community_cards=['Jh', 'Td', '3h', '2c'],
        rel_pos=2, num_active=6,
        effective_stack_bb=80, pot_bb=12,
        was_preflop_raiser=False, facing_bet_size_pct=1.0,
    )
    assert key.draw_type == DT_COMBO_DRAW
    assert key.position == 'OOP'
    assert key.action_context == AC_FACE_BET_LARGE


def test_spot_key_to_tuple_is_hashable():
    """Required for SQLite primary key / dict lookup."""
    obs = _obs(made_hand=HAND_TRIPS)
    key = build_spot_key(
        street='flop', obs=obs,
        hole_cards=['7s', '7h'], community_cards=['7d', 'Js', '2c'],
        rel_pos=0, num_active=2,
        effective_stack_bb=100, pot_bb=4,
        was_preflop_raiser=True, facing_bet_size_pct=None,
    )
    t = key.to_tuple()
    # Hashable for use as dict key
    assert hash(t) == hash(t)
    assert t[0] == 'flop'
    assert t[1] == HB_TRIPS_SET


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
