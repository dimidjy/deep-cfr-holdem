"""Position-calculation tests covering the dead-button edge cases.

Repro of the bug found in logs/action_debug/20260425_145107_381_call.json:
HU spot where the seat marked as button is busted (is_active=False). The
old obs_builder fell back to button_idx=0 when btn_seat wasn't in
active_players, mislabeling hero as BTN even though hero was the BB.
That caused obs/preflop-range disagreement.

These tests pin the dead-button mapping and the obs/preflop-range
agreement so the bug can't regress.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.game_state import CardInfo, GameState, PlayerInfo
from src.agent.obs_builder import ObservationBuilder


def _make_state(*, button_seat, my_seat, seat_specs, street='preflop',
                pot_total=1.5, call_amount=0.0):
    """seat_specs: list of (seat, is_active, is_sitting_out, bet, stack)."""
    players = []
    for seat, active, sitout, bet, stack in seat_specs:
        cards = None
        if seat == my_seat:
            cards = [CardInfo('A', 's'), CardInfo('K', 's')]
        players.append(PlayerInfo(
            seat=seat, name=f'p{seat}', stack=stack, is_active=active,
            is_sitting_out=sitout, bet_current_round=bet, hole_cards=cards,
            has_button=(seat == button_seat)))
    return GameState(
        players=players, my_seat=my_seat, community_cards=[],
        pot_total=pot_total, street=street, button_seat=button_seat,
        legal_actions=[0, 1, 2, 3, 4, 5],
        num_active_players=sum(1 for s in seat_specs if s[1] and not s[2]),
        call_amount=call_amount,
    )


class TestComputeRelPos:
    def test_hu_button_active_btn_is_zero(self):
        # Hero on BTN, opponent BB
        gs = _make_state(button_seat=0, my_seat=0, seat_specs=[
            (0, True, False, 0.5, 100.0),
            (1, True, False, 1.0, 100.0),
        ])
        assert gs.compute_rel_pos() == 0  # BTN

    def test_hu_button_active_bb_is_one(self):
        gs = _make_state(button_seat=0, my_seat=1, seat_specs=[
            (0, True, False, 0.5, 100.0),
            (1, True, False, 1.0, 100.0),
        ])
        assert gs.compute_rel_pos() == 1  # BB

    def test_hu_dead_button_hero_is_bb(self):
        """Reproduces 20260425_145107_381_call: btn seat busted, 2 active.

        Pre-fix obs_builder labeled hero as BTN; correct answer is BB.
        """
        gs = _make_state(button_seat=4, my_seat=0, seat_specs=[
            (0, True,  False, 4.0, 97.5),  # hero
            (1, False, False, 0.0,  0.0),
            (2, False, False, 0.0,  0.0),
            (3, False, False, 0.0,  0.0),
            (4, False, False, 0.0, 45.5),  # busted BTN
            (5, True,  False, 5.0, 219.0),
        ])
        # Closest active seat clockwise from seat 4 is 5 → BTN.
        # Hero (seat 0) is the next one → BB (rel_pos 1).
        assert gs.compute_rel_pos() == 1
        # And the opponent at seat 5 is BTN.
        assert gs.compute_rel_pos(seat=5) == 0

    def test_three_handed_normal(self):
        # btn=0, sb=1, bb=2 — hero=2 should be BB (rel_pos 2)
        gs = _make_state(button_seat=0, my_seat=2, seat_specs=[
            (0, True, False, 0.0, 100.0),
            (1, True, False, 0.5, 100.0),
            (2, True, False, 1.0, 100.0),
        ])
        assert gs.compute_rel_pos(seat=0) == 0
        assert gs.compute_rel_pos(seat=1) == 1
        assert gs.compute_rel_pos(seat=2) == 2

    def test_three_handed_dead_button_t8o_spot(self):
        """Reproduces 20260425_144909_379_raise_min: BTN active, 3 active,
        sparse layout. Hero at seat 0 with btn_seat=0 should be BTN.
        """
        gs = _make_state(button_seat=0, my_seat=0, seat_specs=[
            (0, True,  False, 0.0, 101.0),  # hero, BTN
            (1, True,  False, 0.5,  41.0),  # SB
            (2, False, False, 0.0,   0.0),
            (3, False, False, 0.0,   0.0),
            (4, False, False, 0.0,   0.0),
            (5, True,  False, 1.0,  49.0),  # BB
        ])
        assert gs.compute_rel_pos(seat=0) == 0  # BTN
        assert gs.compute_rel_pos(seat=1) == 1  # SB
        assert gs.compute_rel_pos(seat=5) == 2  # BB

    def test_six_max_normal(self):
        # btn=2 → rel_pos walks clockwise: 2,3,4,5,0,1.
        # Project's rel_pos→pos mapping is 0=BTN, 1=SB, 2=BB, 3=UTG/LJ,
        # 4=MP/HJ, 5=CO. So seats clockwise from btn map to:
        #   2→BTN, 3→SB, 4→BB, 5→LJ(first to act), 0→HJ, 1→CO.
        gs = _make_state(button_seat=2, my_seat=5, seat_specs=[
            (0, True, False, 0.0, 100.0),
            (1, True, False, 0.0, 100.0),
            (2, True, False, 0.0, 100.0),
            (3, True, False, 0.5, 100.0),
            (4, True, False, 1.0, 100.0),
            (5, True, False, 0.0, 100.0),
        ])
        assert gs.compute_rel_pos(seat=2) == 0  # BTN
        assert gs.compute_rel_pos(seat=3) == 1  # SB
        assert gs.compute_rel_pos(seat=4) == 2  # BB
        assert gs.compute_rel_pos(seat=5) == 3  # LJ
        assert gs.compute_rel_pos(seat=0) == 4  # HJ
        assert gs.compute_rel_pos(seat=1) == 5  # CO

    def test_inactive_seat_returns_zero(self):
        gs = _make_state(button_seat=0, my_seat=2, seat_specs=[
            (0, True,  False, 0.0, 100.0),
            (1, True,  False, 0.5, 100.0),
            (2, False, False, 0.0,   0.0),
        ])
        assert gs.compute_rel_pos() == 0

    def test_no_players(self):
        gs = GameState(players=[], my_seat=0, button_seat=0)
        assert gs.compute_rel_pos() == 0


class TestObsBuilderPositionAgreesWithRelPos:
    """Pin: obs's pos one-hot must match compute_rel_pos in dead-button HU."""

    def _pos_one_hot_idx(self, obs):
        # Position block is the first 8 dims (6 one-hot + norm + is_button).
        return int(np.argmax(obs[:6]))

    def _is_button_flag(self, obs):
        return float(obs[7])

    def test_dead_button_hu_obs_says_bb_not_btn(self):
        gs = _make_state(button_seat=4, my_seat=0, seat_specs=[
            (0, True,  False, 4.0, 97.5),
            (1, False, False, 0.0,  0.0),
            (2, False, False, 0.0,  0.0),
            (3, False, False, 0.0,  0.0),
            (4, False, False, 0.0, 45.5),  # busted BTN
            (5, True,  False, 5.0, 219.0),
        ])
        builder = ObservationBuilder(num_players=6, behavioral_stats=True,
                                     equity_vs_odds=True)
        obs = builder.build(gs)
        assert obs is not None
        # In HU, rel_pos 1 maps to BB in 6p position (index 2 via _POS_MAP_TO_6P[2]).
        assert self._pos_one_hot_idx(obs) == 2  # BB
        assert self._is_button_flag(obs) == 0.0

    def test_three_handed_dead_button_btn_active_obs_says_btn(self):
        gs = _make_state(button_seat=0, my_seat=0, seat_specs=[
            (0, True,  False, 0.0, 101.0),
            (1, True,  False, 0.5,  41.0),
            (2, False, False, 0.0,   0.0),
            (3, False, False, 0.0,   0.0),
            (4, False, False, 0.0,   0.0),
            (5, True,  False, 1.0,  49.0),
        ])
        builder = ObservationBuilder(num_players=6, behavioral_stats=True,
                                     equity_vs_odds=True)
        obs = builder.build(gs)
        assert obs is not None
        assert self._pos_one_hot_idx(obs) == 0  # BTN
        assert self._is_button_flag(obs) == 1.0

    def test_position_label_dead_button_hu(self):
        gs = _make_state(button_seat=4, my_seat=0, seat_specs=[
            (0, True,  False, 4.0, 97.5),
            (4, False, False, 0.0, 45.5),
            (5, True,  False, 5.0, 219.0),
        ])
        builder = ObservationBuilder(num_players=6, behavioral_stats=True,
                                     equity_vs_odds=True)
        assert builder.get_position_label(gs) == 'BB'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
