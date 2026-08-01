"""Canonical observation layout — the single source of truth.

Three separate code paths build observation vectors:

  * ``src/engine/fast_enhanced_engine.py``  — Deep CFR training
  * ``src/agent/obs_builder.py``     — the runtime agent
  * ``src/hand_history/obs_builder.py``     — hand-history data generation

They must agree feature-for-feature, or a model trained on one set of
semantics is served a different one at runtime. That has already happened
twice: ``obs[18]`` was heads-up equity in training but multiway equity in
the agent, and the spot annotator's made-hand labels drifted a full tier
out of step with the cutoffs it claimed to mirror.

Everything that describes *where a feature lives* or *how wide a block is*
belongs here, and every consumer should import it rather than re-declaring
a literal. ``tests/test_obs_parity.py`` enforces that the builders agree.

Block layout (6 players)::

    Position(8) PotOdds(3) HandStrength(5) Stack(2) Equity(9)
    Street(4) GameState(6) BettingHistory(5)                 = 42 base
    + BehavioralStats(8) + BettingLine(3)                    = 53  <- current
    [+ OpponentProfile(5) when archetype mode is on]
    [+ EquityVsOdds(1) appended by the agent builder only]
"""
from typing import List

# ─────────────────────────────────────────────────────────────────────────────
# Block widths
# ─────────────────────────────────────────────────────────────────────────────
POSITION_DIM = 8          # 6 one-hot + normalized position + is_button
POT_ODDS_DIM = 3
HAND_STRENGTH_DIM = 5
STACK_DIM = 2
EQUITY_DIM = 9
STREET_DIM = 4
GAME_STATE_DIM = 6
BETTING_HISTORY_DIM = 5

BASE_DIM = (POSITION_DIM + POT_ODDS_DIM + HAND_STRENGTH_DIM + STACK_DIM
            + EQUITY_DIM + STREET_DIM + GAME_STATE_DIM + BETTING_HISTORY_DIM)

BEHAVIORAL_STATS_DIM = 8
BETTING_LINE_DIM = 3
OPPONENT_PROFILE_DIM = 5   # mirrors src/archetypes.OPPONENT_PROFILE_DIM
EQUITY_VS_ODDS_DIM = 1     # agent-only trailing feature

#: Observation width the current models are trained and served on.
STANDARD_OBS_DIM = BASE_DIM + BEHAVIORAL_STATS_DIM + BETTING_LINE_DIM  # 53

# ─────────────────────────────────────────────────────────────────────────────
# Feature names, in build order
# ─────────────────────────────────────────────────────────────────────────────
POSITION_NAMES = [
    'pos_btn', 'pos_sb', 'pos_bb', 'pos_utg', 'pos_mp', 'pos_co',
    'pos_norm', 'is_button',
]
POT_ODDS_NAMES = ['pot_normalized', 'stack_to_pot', 'pot_odds']
HAND_STRENGTH_NAMES = [
    'is_pair', 'is_suited', 'high_card_norm', 'low_card_norm', 'rank_gap_norm',
]
STACK_NAMES = ['m_ratio', 'relative_stack']
EQUITY_NAMES = [
    'hand_strength', 'made_hand_cat',
    'hero_flush_participation', 'hero_straight_participation',
    'top_pair', 'board_pairing_level',
    'board_flush_texture', 'board_straight_texture',
    'hand_rank_within_cat',
]
STREET_NAMES = ['is_preflop', 'is_flop', 'is_turn', 'is_river']
GAME_STATE_NAMES = [
    'players_alive_norm', 'call_amount_norm', 'aggression_level',
    'facing_bet', 'bet_to_pot_norm', 'can_raise',
]
BETTING_HISTORY_NAMES = [
    'my_pot_share', 'not_raise_fraction', 'facing_3bet_plus',
    'pot_was_raised', 'min_opp_stack_norm',
]
BEHAVIORAL_NAMES = [
    'opp_aggression_ratio', 'opp_fold_ratio',
    'avg_bet_sizing', 'max_bet_sizing',
    'num_raises_norm', 'street_aggression',
    'opp_check_ratio', 'pot_escalation',
]
BETTING_LINE_NAMES = [
    'pot_at_street_start_norm', 'hero_has_initiative',
    'agg_count_this_street_norm',
]
OPPONENT_PROFILE_NAMES = [
    'opp_vpip', 'opp_pfr', 'opp_af_norm', 'opp_fold_to_cbet', 'opp_wtsd',
]
EQUITY_VS_ODDS_NAMES = ['equity_vs_odds']


def feature_names(num_players: int = 6, pad_to_6p: bool = True,
                  behavioral_stats: bool = True,
                  equity_vs_odds: bool = False,
                  opp_profile: bool = False) -> List[str]:
    """Ordered feature names for the given builder flags.

    ``behavioral_stats`` covers both the behavioral block and the betting-line
    block — they are emitted together by every builder.
    """
    if pad_to_6p or num_players == 6:
        names = list(POSITION_NAMES)
    else:
        names = ([f'pos_seat_{i}' for i in range(num_players)]
                 + ['pos_norm', 'is_button'])
    names += POT_ODDS_NAMES
    names += HAND_STRENGTH_NAMES
    names += STACK_NAMES
    names += EQUITY_NAMES
    names += STREET_NAMES
    names += GAME_STATE_NAMES
    names += BETTING_HISTORY_NAMES
    if behavioral_stats:
        names += BEHAVIORAL_NAMES
        names += BETTING_LINE_NAMES
    if opp_profile:
        names += OPPONENT_PROFILE_NAMES
    if equity_vs_odds:
        names += EQUITY_VS_ODDS_NAMES
    return names


def index_of(name: str) -> int:
    """Index of a feature in the standard 53-dim layout."""
    return feature_names().index(name)


# ─────────────────────────────────────────────────────────────────────────────
# Named indices — derived from the layout above, never hand-written.
# src/decision_engine.py re-exports these as IDX_* for existing callers.
# ─────────────────────────────────────────────────────────────────────────────
_STD = feature_names()

IDX_POT_NORM = _STD.index('pot_normalized')            # 8
IDX_STACK_TO_POT = _STD.index('stack_to_pot')          # 9
IDX_POT_ODDS = _STD.index('pot_odds')                  # 10
IDX_HIGH_CARD = _STD.index('high_card_norm')           # 13
IDX_EQUITY = _STD.index('hand_strength')               # 18
IDX_MADE_HAND = _STD.index('made_hand_cat')            # 19
IDX_FLUSH_DRAW = _STD.index('hero_flush_participation')     # 20
IDX_STRAIGHT_DRAW = _STD.index('hero_straight_participation')  # 21
IDX_TOP_PAIR = _STD.index('top_pair')                  # 22
IDX_BOARD_PAIR = _STD.index('board_pairing_level')     # 23
IDX_BOARD_FLUSH = _STD.index('board_flush_texture')    # 24
IDX_BOARD_STRAIGHT = _STD.index('board_straight_texture')  # 25
IDX_RANK_WITHIN_CAT = _STD.index('hand_rank_within_cat')   # 26
IDX_IS_RIVER = _STD.index('is_river')                  # 30
IDX_FACING_3BET_PLUS = _STD.index('facing_3bet_plus')  # 39
IDX_MIN_OPP_STACK = _STD.index('min_opp_stack_norm')   # 41
IDX_POT_ESCALATION = _STD.index('pot_escalation')      # 49

# ─────────────────────────────────────────────────────────────────────────────
# Made-hand categories: made_hand_cat = (9 - treys_rank_class) / 8
# ─────────────────────────────────────────────────────────────────────────────
HAND_HIGH_CARD = 0.0
HAND_ONE_PAIR = 0.125
HAND_TWO_PAIR = 0.25
HAND_TRIPS = 0.375
HAND_STRAIGHT = 0.5
HAND_FLUSH = 0.625
HAND_FULL_HOUSE = 0.75
HAND_QUADS = 0.875
HAND_STRAIGHT_FLUSH = 1.0

#: Ordered strongest-first, for labelling a made_hand_cat value.
MADE_HAND_TIERS = [
    (HAND_STRAIGHT_FLUSH, 'straight flush'),
    (HAND_QUADS, 'quads'),
    (HAND_FULL_HOUSE, 'full house'),
    (HAND_FLUSH, 'flush'),
    (HAND_STRAIGHT, 'straight'),
    (HAND_TRIPS, 'trips'),
    (HAND_TWO_PAIR, 'two pair'),
    (HAND_ONE_PAIR, 'one pair'),
    (HAND_HIGH_CARD, 'high card'),
]


def made_hand_label(val: float) -> str:
    """Human-readable name for a made_hand_cat value."""
    if val < 0:
        return 'high card / unknown'
    for threshold, label in MADE_HAND_TIERS:
        if val >= threshold - 0.005:
            return label
    return 'high card'
