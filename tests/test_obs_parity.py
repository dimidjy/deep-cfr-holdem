"""Parity guards for the observation layout.

Three code paths build observations — the training engine
(`src/engine/fast_enhanced_engine.py`), the runtime agent
(`src/agent/obs_builder.py`), and the hand-history generator
(`src/hand_history/obs_builder.py`). When they disagree, a model is trained
on one set of semantics and served another, which is silent and expensive.

Two real bugs motivated these tests:

  * ``obs[18]`` was equity-vs-one-random-hand in the engine but true multiway
    equity in the agent, so a 6-way top pair read 0.70 in training and 0.19 at
    runtime. The engine's opponent counter also referenced two attribute names
    that do not exist (`_state.player_states` / `.has_folded`), so it silently
    fell back to heads-up in every path.
  * The spot annotator's made-hand tier table drifted one full step from the
    cutoffs it claimed to mirror, labelling high card as "one pair".
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import obs_schema as schema
from src.engine.factory import make_engine
from src.engine.fast_enhanced_engine import FastEnhancedPokerEngine
from src.agent.obs_builder import ObservationBuilder, feature_names

try:
    import eval7 as _eval7
    _EVAL7 = True
except ImportError:  # pragma: no cover - eval7 is a hard dep in practice
    _EVAL7 = False


# ─────────────────────────────────────────────────────────────────────────────
# Layout: every builder must agree on width and ordering
# ─────────────────────────────────────────────────────────────────────────────
class TestLayoutParity:
    def test_engine_obs_width_matches_schema(self):
        eng = make_engine(num_players=6)
        eng.reset()
        assert eng.get_observation_space_size() == schema.STANDARD_OBS_DIM
        assert len(eng.get_observation()) == schema.STANDARD_OBS_DIM

    def test_agent_feature_names_match_schema(self):
        names = feature_names(num_players=6, pad_to_6p=True,
                              behavioral_stats=True, equity_vs_odds=False)
        assert names == schema.feature_names()
        assert len(names) == schema.STANDARD_OBS_DIM

    def test_agent_adds_exactly_one_dim_for_equity_vs_odds(self):
        with_evo = feature_names(equity_vs_odds=True)
        assert len(with_evo) == schema.STANDARD_OBS_DIM + schema.EQUITY_VS_ODDS_DIM
        assert with_evo[-1] == 'equity_vs_odds'

    def test_block_widths_shared_across_builders(self):
        from src.hand_history import obs_builder as hh
        assert hh.BEHAVIORAL_STATS_DIM == schema.BEHAVIORAL_STATS_DIM
        assert hh.BETTING_LINE_DIM == schema.BETTING_LINE_DIM
        assert FastEnhancedPokerEngine.BEHAVIORAL_STATS_DIM == schema.BEHAVIORAL_STATS_DIM
        assert FastEnhancedPokerEngine.BETTING_LINE_DIM == schema.BETTING_LINE_DIM
        assert FastEnhancedPokerEngine.OPPONENT_PROFILE_DIM == schema.OPPONENT_PROFILE_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Indices: the magic numbers sprinkled through the codebase must still be right
# ─────────────────────────────────────────────────────────────────────────────
class TestIndexParity:
    def test_decision_engine_indices_come_from_schema(self):
        import src.decision_engine as de
        for name in ('IDX_POT_NORM', 'IDX_STACK_TO_POT', 'IDX_POT_ODDS',
                     'IDX_HIGH_CARD', 'IDX_EQUITY', 'IDX_MADE_HAND',
                     'IDX_FLUSH_DRAW', 'IDX_STRAIGHT_DRAW', 'IDX_TOP_PAIR',
                     'IDX_BOARD_PAIR', 'IDX_BOARD_FLUSH', 'IDX_BOARD_STRAIGHT',
                     'IDX_RANK_WITHIN_CAT', 'IDX_IS_RIVER'):
            assert getattr(de, name) == getattr(schema, name), name

    @pytest.mark.parametrize('idx,expected', [
        (schema.IDX_POT_ODDS, 'pot_odds'),
        (schema.IDX_EQUITY, 'hand_strength'),
        (schema.IDX_MADE_HAND, 'made_hand_cat'),
        (schema.IDX_FLUSH_DRAW, 'hero_flush_participation'),
        (schema.IDX_STRAIGHT_DRAW, 'hero_straight_participation'),
        (schema.IDX_TOP_PAIR, 'top_pair'),
    ])
    def test_index_points_at_expected_feature(self, idx, expected):
        assert schema.feature_names()[idx] == expected

    def test_train_gpu_shaper_indices(self):
        """train_gpu.py's barrel-fold shaper reads these three raw indices."""
        names = schema.feature_names()
        assert names[39] == 'facing_3bet_plus'
        assert names[41] == 'min_opp_stack_norm'
        assert names[49] == 'pot_escalation'


# ─────────────────────────────────────────────────────────────────────────────
# Equity semantics: the actual train/serve mismatch that shipped
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not _EVAL7, reason='eval7 not installed')
class TestEquityParity:
    CASES = [
        (['3s', 'Qh'], ['Qc', '2c', 'Tc']),   # top pair, monotone board
        (['Ah', 'Kd'], ['7c', '2d', '9s']),   # overcards, dry board
        (['7h', '7c'], ['7s', 'Kd', '3c']),   # set
    ]

    @pytest.mark.parametrize('hole,board', CASES)
    @pytest.mark.parametrize('n_opp', [2, 3, 5])
    def test_engine_and_agent_multiway_equity_agree(self, hole, board, n_opp):
        """Same hand, same opponent count → same equity within MC noise.

        Before the fix the engine ignored n_opp entirely and these differed
        by up to 0.5.
        """
        e7_hole = [_eval7.Card(c) for c in hole]
        e7_board = [_eval7.Card(c) for c in board]
        agent = ObservationBuilder(num_players=6, behavioral_stats=True)

        engine_eq = np.mean([
            FastEnhancedPokerEngine._multiway_equity_mc(e7_hole, e7_board, n_opp, 400)
            for _ in range(3)])
        agent_eq = np.mean([
            agent._multiway_equity(e7_hole, e7_board, n_opp, 400)
            for _ in range(3)])

        assert abs(engine_eq - agent_eq) < 0.05, (
            f'{"".join(hole)} on {"".join(board)} vs {n_opp} opponents: '
            f'engine {engine_eq:.3f} vs agent {agent_eq:.3f}')

    @pytest.mark.parametrize('hole,board', CASES)
    def test_equity_decreases_with_more_opponents(self, hole, board):
        """A sanity property the broken engine violated: it returned the same
        heads-up number no matter how many players were in the pot."""
        e7_hole = [_eval7.Card(c) for c in hole]
        e7_board = [_eval7.Card(c) for c in board]
        eqs = [FastEnhancedPokerEngine._multiway_equity_mc(e7_hole, e7_board, n, 400)
               for n in (1, 3, 5)]
        assert eqs[0] > eqs[1] > eqs[2], f'equity did not fall with opponents: {eqs}'


class TestActiveOpponentCount:
    """`_n_active_opponents` used to read `_state.player_states[...].has_folded`
    — neither name exists, so it always threw and returned 1."""

    def test_counts_all_opponents_at_start(self):
        eng = make_engine(num_players=6)
        eng.reset()
        assert eng._n_active_opponents() == 5

    def test_count_drops_as_players_fold(self):
        eng = make_engine(num_players=6)
        eng.reset()
        before = eng._n_active_opponents()
        folded = 0
        for _ in range(2):
            if eng.is_done():
                break
            legal = eng.get_legal_actions()
            if 0 in legal:
                eng.step(0)
                folded += 1
            else:
                break
        if folded:
            assert eng._n_active_opponents() == before - folded

    def test_never_returns_zero(self):
        eng = make_engine(num_players=6)
        eng.reset()
        guard = 0
        while not eng.is_done() and guard < 40:
            assert eng._n_active_opponents() >= 1
            legal = eng.get_legal_actions()
            eng.step(legal[0])
            guard += 1


# ─────────────────────────────────────────────────────────────────────────────
# Made-hand labels
# ─────────────────────────────────────────────────────────────────────────────
class TestMadeHandLabels:
    EXPECTED = {
        schema.HAND_HIGH_CARD: 'high card',
        schema.HAND_ONE_PAIR: 'one pair',
        schema.HAND_TWO_PAIR: 'two pair',
        schema.HAND_TRIPS: 'trips',
        schema.HAND_STRAIGHT: 'straight',
        schema.HAND_FLUSH: 'flush',
        schema.HAND_FULL_HOUSE: 'full house',
        schema.HAND_QUADS: 'quads',
        schema.HAND_STRAIGHT_FLUSH: 'straight flush',
    }

    @pytest.mark.parametrize('val,label', sorted(EXPECTED.items()))
    def test_label_matches_cutoff(self, val, label):
        assert schema.made_hand_label(val) == label

    def test_cutoffs_are_one_eighth_apart(self):
        """made_hand_cat = (9 - treys_rank_class) / 8 — evenly spaced."""
        vals = sorted(self.EXPECTED)
        diffs = [round(b - a, 6) for a, b in zip(vals, vals[1:])]
        assert all(d == 0.125 for d in diffs), diffs
