"""Unit tests for the GTO integration (src/gto/).

Covers the parts that don't need the solver binary: the solver↔project action
mapping, range-string building/expansion, and spot-input derivation on a
synthetic first-to-act and a synthetic facing-bet river spot.

Run: python3 -m pytest tests/test_gto.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gto.action_mapping import (
    map_node_actions, fold_strategy_to_vector, parse_solver_action,
    bet_size_lines, FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN,
)
from src.gto.range_builder import (
    expand_range_to_explicit, build_range_string_from_combos,
    _ensure_class_weight_floor, tier_range_string,
)
from src.gto.spot_inputs import build_gto_spot_inputs


# ── action_mapping ──────────────────────────────────────────────────────

def test_parse_solver_action():
    assert parse_solver_action('BET 75.000000') == ('BET', 75.0)
    assert parse_solver_action('CHECK') == ('CHECK', None)
    assert parse_solver_action('RAISE 200') == ('RAISE', 200.0)


def test_map_bet_node():
    # Betting node (CHECK available): smallest bet → Raise½, larger → RaisePot,
    # bet ≥ 0.9*stack → AllIn.
    actions = ['CHECK', 'BET 100.000000', 'BET 200.000000', 'BET 3350.000000']
    m = map_node_actions(actions, eff_stack_chips=3350)
    assert m['CHECK'] == CALL
    assert m['BET 100.000000'] == RAISE_HALF
    assert m['BET 200.000000'] == RAISE_POT
    assert m['BET 3350.000000'] == ALL_IN


def test_map_facing_bet_node():
    actions = ['CALL', 'RAISE 1800.000000', 'RAISE 5650.000000', 'FOLD']
    m = map_node_actions(actions, eff_stack_chips=5650)
    assert m['FOLD'] == FOLD
    assert m['CALL'] == CALL
    assert m['RAISE 5650.000000'] == ALL_IN      # ≥ 0.9 * stack
    assert m['RAISE 1800.000000'] == RAISE_POT   # raising node → pot bucket


def test_fold_strategy_to_vector():
    actions = ['CHECK', 'BET 100.000000', 'BET 200.000000']
    vec = fold_strategy_to_vector(
        {actions[0]: 0.5, actions[1]: 0.3, actions[2]: 0.2}, eff_stack_chips=3350)
    assert len(vec) == 6
    assert abs(vec[CALL] - 0.5) < 1e-6
    assert abs(vec[RAISE_HALF] - 0.3) < 1e-6
    assert abs(vec[RAISE_POT] - 0.2) < 1e-6
    assert abs(sum(vec) - 1.0) < 1e-6


def test_bet_size_lines():
    lines = bet_size_lines(('river',))
    assert 'set_bet_sizes oop,river,bet,50,100' in lines
    assert 'set_bet_sizes ip,river,raise,100' in lines
    assert any(l.startswith('set_allin_threshold') for l in lines)


def test_bet_size_lines_injected():
    # An injected exact bet size is merged (sorted) into that cell only.
    lines = bet_size_lines(('river',), extra={('oop', 'river', 'bet'): [81]})
    assert 'set_bet_sizes oop,river,bet,50,81,100' in lines
    assert 'set_bet_sizes ip,river,bet,50,100' in lines   # other cell untouched


# ── range_builder ───────────────────────────────────────────────────────

def test_expand_range_no_plus():
    assert expand_range_to_explicit('22+') == \
        '22,33,44,55,66,77,88,99,TT,JJ,QQ,KK,AA'
    assert expand_range_to_explicit('A2s+') == \
        'A2s,A3s,A4s,A5s,A6s,A7s,A8s,A9s,ATs,AJs,AQs,AKs'
    assert expand_range_to_explicit('K9o+') == 'K9o,KTo,KJo,KQo'
    assert '+' not in tier_range_string(1)


def test_build_range_weights():
    s = build_range_string_from_combos([('AA', 100), ('AKo', 100)])
    # AA (6 combos) should carry MORE weight per class string than AKo (12),
    # so that per-combo frequency matches the equal class counts.
    parts = dict(p.split(':') for p in s.split(','))
    assert float(parts['AA']) > float(parts['AKo'])


def test_ensure_class_weight_floor():
    # Absent class → added at the floor weight.
    assert _ensure_class_weight_floor('AA,KK', 'T4o').endswith('T4o:0.1000')
    # Present below floor → boosted; never mixes class + specific combo
    # (which crashes the solver).
    out = _ensure_class_weight_floor('AA,T4o:0.0018,KK', 'T4o')
    assert 'T4o:0.1000' in out and 'T4o:0.0018' not in out
    # Present above floor → left as-is.
    assert _ensure_class_weight_floor('AA,T4o:0.5', 'T4o') == 'AA,T4o:0.5000'


# ── spot_inputs ─────────────────────────────────────────────────────────

def _obs_two_tone():
    """Minimal obs long enough for the board-texture classifier (idx 23-25)."""
    obs = [0.0] * 54
    obs[24] = 0.5   # board_flush_texture in the two_tone band
    return obs


def _river_spot(hero_bet, villain_bet, call_amount, hero_button):
    """Synthetic HU river spot JSON. Hero seat 0, villain seat 1."""
    return {
        'action_idx': 1,
        'obs_vector': _obs_two_tone(),
        'game_state': {
            'my_seat': 0,
            'button_seat': 0 if hero_button else 1,
            'street': 'river',
            'pot_total': 20.0,
            'call_amount': call_amount,
            'num_active_players': 2,
            'big_blind': 1.0,
            'community_cards': [
                {'rank': '8', 'suit': 'c'}, {'rank': 'Q', 'suit': 'd'},
                {'rank': '7', 'suit': 's'}, {'rank': '9', 'suit': 'c'},
                {'rank': '2', 'suit': 'h'}],
            'players': [
                {'seat': 0, 'stack': 80.0, 'is_active': True,
                 'bet_current_round': hero_bet, 'total_invested': 4.0,
                 'has_button': hero_button,
                 'hole_cards': [{'rank': 'J', 'suit': 'h'},
                                {'rank': '5', 'suit': 'h'}]},
                {'seat': 1, 'stack': 80.0, 'is_active': True,
                 'bet_current_round': villain_bet, 'total_invested': 4.0,
                 'has_button': not hero_button}],
        },
    }


def test_spot_inputs_first_to_act():
    inp = build_gto_spot_inputs(_river_spot(0.0, 0.0, 0.0, hero_button=False))
    assert inp is not None
    assert inp.street == 'river'
    assert inp.hero_is_oop is True          # hero not on button → OOP → first
    assert inp.nav_type == 'root_hero'
    assert inp.hero_hand_combo == 'Jh5h'
    assert inp.hero_hand_class == 'J5s'
    assert inp.pot_chips == 2000            # 20 BB * 100
    assert inp.facing_bet_size_pct is None


def test_spot_inputs_facing_bet_ip():
    # Hero on button (IP), villain (OOP) bet 10 into a 10-pot (pot_total=20).
    inp = build_gto_spot_inputs(_river_spot(0.0, 10.0, 10.0, hero_button=True))
    assert inp is not None
    assert inp.hero_is_oop is False
    assert inp.nav_type == 'ip_face_bet'
    assert inp.facing_bet_size_pct is not None
    # villain bet 10 into start pot 10 → frac ≈ 1.0
    assert abs(inp.facing_bet_frac_start_pot - 1.0) < 1e-6
    # villain range key flips position to OOP
    assert inp.villain_spot_key.position == 'OOP'
    assert inp.hero_spot_key.position == 'IP'


def test_bluff_reduction():
    from types import SimpleNamespace
    from src.gto.extract import _apply_bluff_reduction
    # High-card hand betting first-to-act: [fold, check, R½, RPot, allin, _]
    air = SimpleNamespace(hero_made_hand_cat=0.0, nav_type='root_hero')
    vec = _apply_bluff_reduction([0.0, 0.4, 0.3, 0.3, 0.0, 0.0], air, 1.0)
    assert vec == [0.0, 0.4, 0.3, 0.3, 0.0, 0.0]          # factor 1.0 = no change
    vec = _apply_bluff_reduction([0.0, 0.4, 0.3, 0.3, 0.0, 0.0], air, 0.0)
    assert abs(vec[1] - 1.0) < 1e-9 and vec[2] == 0.0 and vec[3] == 0.0  # all → check
    vec = _apply_bluff_reduction([0.0, 0.4, 0.3, 0.3, 0.0, 0.0], air, 0.5)
    assert abs(vec[2] - 0.15) < 1e-9 and abs(vec[1] - 0.7) < 1e-9         # half trimmed

    # Air FACING a bet: removed mass goes to FOLD, not check.
    air_fb = SimpleNamespace(hero_made_hand_cat=0.0, nav_type='ip_face_bet')
    vec = _apply_bluff_reduction([0.2, 0.5, 0.0, 0.3, 0.0, 0.0], air_fb, 0.0)
    assert abs(vec[0] - 0.5) < 1e-9 and vec[3] == 0.0                     # raise → fold

    # Value hand (two pair) is never trimmed.
    value = SimpleNamespace(hero_made_hand_cat=0.25, nav_type='root_hero')
    vec = _apply_bluff_reduction([0.0, 0.2, 0.4, 0.4, 0.0, 0.0], value, 0.0)
    assert vec == [0.0, 0.2, 0.4, 0.4, 0.0, 0.0]


def test_spot_inputs_skips_preflop_and_multiway():
    pre = _river_spot(0.0, 0.0, 0.0, hero_button=False)
    pre['game_state']['street'] = 'preflop'
    assert build_gto_spot_inputs(pre) is None
    multi = _river_spot(0.0, 0.0, 0.0, hero_button=False)
    multi['game_state']['num_active_players'] = 3
    assert build_gto_spot_inputs(multi) is None


if __name__ == '__main__':
    import subprocess
    raise SystemExit(subprocess.call(
        [sys.executable, '-m', 'pytest', __file__, '-q']))
