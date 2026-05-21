"""Build 38-dim (for 6-player) observation vectors from GameState.

Replicates the exact feature computation from FastEnhancedPokerEngine
(src/engine/fast_enhanced_engine.py) and ObservationBuilder
(src/hand_history/obs_builder.py), but takes GameState as input.
"""

import math
from typing import List, Optional

import numpy as np

from src.agent.game_state import CardInfo, PlayerInfo, GameState

try:
    from treys import Card as TreysCard, Evaluator as TreysEvaluator
    _TREYS_EVALUATOR = TreysEvaluator()
    _TREYS_AVAILABLE = True
except ImportError:
    _TREYS_AVAILABLE = False

try:
    import eval7 as _eval7
    _RANKS = '23456789TJQKA'
    _all_combos = []
    for _i, _r1 in enumerate(_RANKS):
        for _j, _r2 in enumerate(_RANKS):
            if _i < _j:
                _all_combos.append(f'{_r2}{_r1}s')
                _all_combos.append(f'{_r2}{_r1}o')
            elif _i == _j:
                _all_combos.append(f'{_r1}{_r2}')
    _EVAL7_RANDOM_RANGE = _eval7.HandRange(','.join(_all_combos))
    _EVAL7_AVAILABLE = True
except ImportError:
    _EVAL7_AVAILABLE = False


_RANK_MAP = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
}

# Treys rank_class → (best_rank_int, worst_rank_int) for within-category ranking
_TREYS_RANK_BOUNDS = {
    1: (1, 10),       # Straight Flush
    2: (11, 166),     # Four of a Kind
    3: (167, 322),    # Full House
    4: (323, 1599),   # Flush
    5: (1600, 1609),  # Straight
    6: (1610, 2467),  # Three of a Kind
    7: (2468, 3325),  # Two Pair
    8: (3326, 6185),  # One Pair
    9: (6186, 7462),  # High Card
}

_POS_LABELS = {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'UTG', 4: 'MP', 5: 'CO'}

BEHAVIORAL_STATS_DIM = 8

# Maps (num_players, rel_pos) → 6p rel_pos for observation padding
_POS_MAP_TO_6P = {
    6: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},  # identity
    5: {0: 0, 1: 1, 2: 2, 3: 4, 4: 5},          # HJ→4(MP), CO→5
    4: {0: 0, 1: 1, 2: 2, 3: 5},                 # CO→5
    3: {0: 0, 1: 1, 2: 2},                        # BTN, SB, BB only
    2: {0: 0, 1: 2},                              # BTN→0, BB→2
}


def _rank_val(rank_char: str) -> int:
    return _RANK_MAP.get(rank_char.upper(), 0)


# Feature names in the exact order produced by ObservationBuilder.build().
# When pad_to_6p=True, position block is 8 dims (6 one-hot + norm + is_button).
# When pad_to_6p=False, position block is num_players + 2.
_POSITION_6P_NAMES = [
    'pos_btn', 'pos_sb', 'pos_bb', 'pos_utg', 'pos_mp', 'pos_co',
    'pos_norm', 'is_button',
]
_POT_ODDS_NAMES = ['pot_normalized', 'stack_to_pot', 'pot_odds']
_HAND_STRENGTH_NAMES = [
    'is_pair', 'is_suited', 'high_card_norm', 'low_card_norm', 'rank_gap_norm',
]
_STACK_NAMES = ['m_ratio', 'relative_stack']
_EQUITY_NAMES = [
    'hand_strength', 'made_hand_cat',
    'hero_flush_participation', 'hero_straight_participation',
    'top_pair', 'board_pairing_level',
    'board_flush_texture', 'board_straight_texture',
    'hand_rank_within_cat',
]
_STREET_NAMES = ['is_preflop', 'is_flop', 'is_turn', 'is_river']
_GAME_STATE_NAMES = [
    'players_alive_norm', 'call_amount_norm', 'aggression_level',
    'facing_bet', 'bet_to_pot_norm', 'can_raise',
]
_BETTING_HIST_NAMES = [
    'my_pot_share', 'not_raise_fraction', 'facing_3bet_plus',
    'pot_was_raised', 'min_opp_stack_norm',
]
_BEHAVIORAL_NAMES = [
    'opp_aggression_ratio', 'opp_fold_ratio',
    'avg_bet_sizing', 'max_bet_sizing',
    'num_raises_norm', 'street_aggression',
    'opp_check_ratio', 'pot_escalation',
]
_BETTING_LINE_NAMES = [
    'pot_at_street_start_norm',
    'hero_has_initiative',
    'agg_count_this_street_norm',
]
BETTING_LINE_DIM = 3
_EQUITY_VS_ODDS_NAMES = ['equity_vs_odds']
_THREAT_RANGE_NAMES = ['equity_vs_threat_range']
_EQUITY_VS_RANDOM_NAMES = ['equity_vs_random']
_OPP_PROFILE_NAMES = [
    'opp_vpip', 'opp_pfr', 'opp_af_norm', 'opp_fold_to_cbet', 'opp_wtsd',
]


def feature_names(num_players: int = 6, pad_to_6p: bool = True,
                  behavioral_stats: bool = True,
                  equity_vs_odds: bool = True,
                  opp_profile: bool = False,
                  threat_range_equity: bool = False,
                  equity_vs_random: bool = False) -> List[str]:
    """Return ordered feature names matching ObservationBuilder.build() output.

    Pass the same constructor flags used for the builder to get matching names.
    Position block size depends on pad_to_6p:
    - True (or num_players=6): 8 dims (6 one-hot + norm + is_button)
    - False + num_players<6: num_players + 2 dims
    """
    names = []
    if pad_to_6p or num_players == 6:
        names += _POSITION_6P_NAMES
    else:
        names += [f'pos_seat_{i}' for i in range(num_players)] + ['pos_norm', 'is_button']
    names += _POT_ODDS_NAMES
    names += _HAND_STRENGTH_NAMES
    names += _STACK_NAMES
    names += _EQUITY_NAMES
    names += _STREET_NAMES
    names += _GAME_STATE_NAMES
    names += _BETTING_HIST_NAMES
    if behavioral_stats:
        names += _BEHAVIORAL_NAMES
        names += _BETTING_LINE_NAMES
    if opp_profile:
        names += _OPP_PROFILE_NAMES
    if equity_vs_odds:
        names += _EQUITY_VS_ODDS_NAMES
    # threat_range_equity / equity_vs_random are accepted for backward compat
    # but no longer wired into the obs (reverted to 53-dim; obs[18] = raw
    # equity-vs-random). Range-equity functions remain in src/range_equity.py.
    return names


class ObservationBuilder:
    """Build observation vectors from GameState for model inference.

    When pad_to_6p=True, always outputs 41-dim (6-player format) observations,
    mapping positions to their 6p equivalents. This allows using a single 6p model
    for all table sizes (2-6 players).
    """

    def __init__(self, num_players: int = 6, pad_to_6p: bool = False,
                 behavioral_stats: bool = False, equity_vs_odds: bool = False,
                 threat_range_equity: bool = False,
                 equity_vs_random: bool = False):
        self.num_players = num_players
        self.pad_to_6p = pad_to_6p and num_players < 6
        self.behavioral_stats = behavioral_stats
        self.equity_vs_odds = equity_vs_odds
        # NOTE: threat_range_equity / equity_vs_random are accepted for backward
        # compat but NO LONGER WIRED. obs is 53-dim with obs[18] = raw MC equity
        # vs a uniform-random range. The range-equity machinery
        # (_compute_obs18_threat_equity, src/range_equity.py) is retained for
        # future re-wiring.
        self.threat_range_equity = threat_range_equity
        self.equity_vs_random = equity_vs_random
        self._last_equity_vs_random: float = 0.0

    def build(self, game_state: GameState,
              opponent_profile: np.ndarray = None,
              hero_has_initiative_override: Optional[bool] = None,
              hand_raise_count_override: Optional[int] = None,
              preflop_raise_count_override: Optional[int] = None,
              opp_bet_sizings_override: Optional[List[float]] = None,
              opp_action_counts_override: Optional[dict] = None,
              street_action_counts_override: Optional[dict] = None
              ) -> Optional[np.ndarray]:
        """Build observation vector from GameState.

        Args:
            game_state: Current game state from parser.
            opponent_profile: Optional 5-dim cross-hand opponent stats
                (VPIP, PFR, AF_norm, Fold2CBet, WTSD). If None, skipped.
            hero_has_initiative_override: Cross-hand boolean (hero is the
                last aggressor in the hand). The caller tracks this per
                hand and passes it in. Without it, a degenerate snapshot
                proxy is used.
            hand_raise_count_override: Total raises across the hand (hero
                + opps, all streets). Drives `num_raises_norm`. Without it
                a current-street-only snapshot proxy is used (which
                collapses to 0 on every checked-to flop).
            opp_bet_sizings_override: List of opp bet/raise sizings as a
                ratio (bet / pot-before-bet) accumulated across the hand.
                Drives `avg_bet_sizing` and `max_bet_sizing`. Without it
                only the current snapshot's opp bets are used.

        Returns None if hole cards missing.
        """
        players = game_state.active_players
        if not players:
            return None

        # Reset per-build so a stale value from the previous build can't leak
        # into the appended equity_vs_random feature.
        self._last_equity_vs_random = 0.0

        my_player = game_state.my_player
        if not my_player or not my_player.hole_cards:
            return None

        # Find player index in active list (used for stack/pot lookups)
        player_idx = 0
        for i, p in enumerate(players):
            if p.seat == game_state.my_seat:
                player_idx = i
                break

        rel_pos = game_state.compute_rel_pos()

        hole_cards = [str(c) for c in my_player.hole_cards]
        community = [str(c) for c in game_state.community_cards]

        features = []
        features.append(self._position_features(rel_pos,
                                                 num_active=len(players)))
        features.append(self._pot_odds_features(game_state, players, player_idx))
        features.append(self._hand_strength_features(hole_cards))
        features.append(self._stack_features(game_state, players, player_idx))
        n_opponents = max(len([p for p in players if p.is_active]) - 1, 1)
        features.append(self._equity_features(hole_cards, community, n_opponents,
                                                game_state=game_state, players=players,
                                                hero_has_initiative=hero_has_initiative_override,
                                                hand_raise_count=hand_raise_count_override,
                                                preflop_raise_count=preflop_raise_count_override))
        features.append(self._street_features(game_state.street))
        features.append(self._game_state_features(game_state, players, player_idx))
        features.append(self._betting_history_features(game_state, players, player_idx))

        if self.behavioral_stats:
            features.append(self._behavioral_stats_features(
                game_state, players, player_idx,
                hand_raise_count_override=hand_raise_count_override,
                opp_bet_sizings_override=opp_bet_sizings_override,
                opp_action_counts_override=opp_action_counts_override,
                street_action_counts_override=street_action_counts_override))
            features.append(self._betting_line_features(
                game_state, players, player_idx,
                hero_has_initiative_override=hero_has_initiative_override))

        if opponent_profile is not None:
            features.append(opponent_profile.astype(np.float32))

        obs = np.concatenate(features).astype(np.float32)

        # Equity vs pot odds: positive = profitable call, negative = fold
        if self.equity_vs_odds and len(obs) > 18:
            pot_odds = obs[10]  # pot / (pot + call)
            equity = obs[18]   # hand_strength
            equity_needed = 1.0 - pot_odds
            evo = np.clip(equity - equity_needed, -1.0, 1.0)
            obs = np.append(obs, np.float32(evo))

        # NOTE: obs is 53-dim. obs[18] = raw MC equity vs a uniform-random range
        # (see _equity_features). The threat-range overlay and the appended
        # equity_vs_random dim were reverted; threat_range_equity/equity_vs_random
        # flags are accepted but no longer wired. The range-equity functions
        # (_compute_obs18_threat_equity, src/range_equity.py) remain for future
        # re-wiring.

        return obs

    def _compute_opp_range_spot_key(self, game_state, players, player_idx,
                                      obs, bh_hist_start, behavioral_start):
        """Build OppRangeSpotKey from current game state. Returns None on
        any failure or for preflop (DB is postflop-only).
        """
        if game_state.street == 'preflop':
            return None
        try:
            from src.opp_range_spot_key import build_opp_range_spot_key
            num_active = len(players)
            my_player = players[player_idx]
            effective_stack = my_player.stack
            pot_bb = max(game_state.pot_total / game_state.big_blind, 0.5)
            eff_bb = max(effective_stack / game_state.big_blind, 0.5)
            rel_pos = game_state.compute_rel_pos()
            # Use hero_has_initiative (betting_line[1]) as proxy for
            # was_preflop_raiser. betting_line starts at behavioral_start+8.
            betting_line_start = behavioral_start + 8
            if len(obs) > betting_line_start + 1:
                hero_has_initiative = float(obs[betting_line_start + 1])
            else:
                hero_has_initiative = 0.0
            was_pf_raiser = bool(hero_has_initiative >= 0.5)
            # facing_bet_size_pct: opp's max bet / pot
            opp_bets = [p.bet_current_round for i, p in enumerate(players)
                         if i != player_idx]
            max_opp_bet = max(opp_bets) if opp_bets else 0.0
            facing_bet_pct = None
            if max_opp_bet > my_player.bet_current_round:
                pot_before = max(game_state.pot_total - max_opp_bet, 1.0)
                facing_bet_pct = (max_opp_bet - my_player.bet_current_round) / pot_before
            return build_opp_range_spot_key(
                street=game_state.street, obs=obs,
                rel_pos=rel_pos, num_active=num_active,
                effective_stack_bb=eff_bb, pot_bb=pot_bb,
                was_preflop_raiser=was_pf_raiser,
                facing_bet_size_pct=facing_bet_pct,
            )
        except Exception:
            return None

    def get_position_label(self, game_state: GameState) -> str:
        n = max(len(game_state.active_players), 1)
        rel_pos_actual = game_state.compute_rel_pos() % n
        mapped_pos = _POS_MAP_TO_6P.get(n, {}).get(rel_pos_actual, rel_pos_actual)
        return _POS_LABELS.get(mapped_pos, 'MP')

    def _position_features(self, rel_pos: int,
                           num_active: Optional[int] = None) -> np.ndarray:
        # Modulo by actual active count, not configured num_players.
        # Runtime sees varying active counts (HU/3-handed); modulo by 6 when
        # only 3 are active gave wrong rel_pos (e.g. SB labeled as MP).
        n = num_active if num_active is not None else self.num_players
        rel_pos_actual = rel_pos % n if n > 0 else 0
        # Map short-handed rel_pos to its 6-handed equivalent so output stays
        # in the distribution the model was trained on (6-handed only).
        mapped_pos = _POS_MAP_TO_6P.get(n, {}).get(rel_pos_actual, rel_pos_actual)

        if self.pad_to_6p:
            one_hot = np.zeros(6, dtype=np.float32)
            one_hot[mapped_pos] = 1.0
            norm_pos = mapped_pos / 6.0
            is_button = 1.0 if rel_pos_actual == 0 else 0.0
            return np.array([*one_hot, norm_pos, is_button], dtype=np.float32)
        else:
            one_hot = np.zeros(self.num_players, dtype=np.float32)
            one_hot[mapped_pos] = 1.0
            norm_pos = mapped_pos / self.num_players
            is_button = 1.0 if rel_pos_actual == 0 else 0.0
            return np.array([*one_hot, norm_pos, is_button], dtype=np.float32)

    def _pot_odds_features(self, gs: GameState, players: List[PlayerInfo],
                           player_idx: int) -> np.ndarray:
        my_stack = players[player_idx].stack  # in BBs
        pot = gs.pot_total                     # in BBs
        total_chips = self.num_players * 100   # default 100BB stacks

        pot_normalized = float(np.clip(pot / max(total_chips, 1), 0.0, 1.0))
        raw_stp = my_stack / max(pot, 1)
        stack_to_pot = float(np.clip(
            math.log1p(raw_stp) / math.log1p(50), 0.0, 1.0))

        call_amount = gs.call_amount
        pot_odds = float(pot / max(pot + call_amount, 1)) if call_amount > 0 else 0.0

        return np.array([pot_normalized, stack_to_pot, pot_odds], dtype=np.float32)

    def _hand_strength_features(self, hole_cards: List[str]) -> np.ndarray:
        if len(hole_cards) < 2:
            return np.zeros(5, dtype=np.float32)
        r0 = _rank_val(hole_cards[0][0])
        r1 = _rank_val(hole_cards[1][0])
        s0 = hole_cards[0][1]
        s1 = hole_cards[1][1]

        is_pair = 1.0 if r0 == r1 else 0.0
        is_suited = 1.0 if s0 == s1 else 0.0
        hi, lo = max(r0, r1), min(r0, r1)
        high_card_norm = (hi - 2) / 12.0
        low_card_norm = (lo - 2) / 12.0
        rank_gap_norm = (hi - lo) / 12.0

        return np.array(
            [is_pair, is_suited, high_card_norm, low_card_norm, rank_gap_norm],
            dtype=np.float32)

    def _stack_features(self, gs: GameState, players: List[PlayerInfo],
                        player_idx: int) -> np.ndarray:
        my_stack = players[player_idx].stack
        total_chips = sum(p.stack + p.total_invested for p in players)
        pot = gs.pot_total

        m_ratio = min(my_stack / max(pot + 2, 1), 5.0) / 5.0
        relative_stack = my_stack / max(total_chips, 1)

        return np.array([m_ratio, relative_stack], dtype=np.float32)

    def _equity_features(self, hole_cards: List[str],
                         community_cards: List[str],
                         n_opponents: int = 1,
                         game_state=None,
                         players=None,
                         hero_has_initiative=None,
                         hand_raise_count=None,
                         preflop_raise_count=None) -> np.ndarray:
        if len(hole_cards) < 2:
            return np.zeros(9, dtype=np.float32)

        r1 = _rank_val(hole_cards[0][0])
        r2 = _rank_val(hole_cards[1][0])
        hi, lo = max(r1, r2), min(r1, r2)

        # Try GPU equity model first (fast path)
        try:
            from src.engine.fast_enhanced_engine import _EQUITY_MODEL, _equity_model_predict, _SUIT_MAP_EQ
            if _EQUITY_MODEL is not None:
                hole_ranks = [r1, r2]
                hole_suits = [_SUIT_MAP_EQ.get(hole_cards[0][1].lower(), 0),
                              _SUIT_MAP_EQ.get(hole_cards[1][1].lower(), 0)]
                board_ranks = [_rank_val(c[0]) for c in community_cards]
                board_suits = [_SUIT_MAP_EQ.get(c[1].lower(), 0) for c in community_cards]
                pred = _equity_model_predict(hole_ranks, hole_suits,
                                              board_ranks, board_suits, None, None,
                                              n_opponents=n_opponents)
                if pred is not None:
                    # pred[0] = raw equity vs random → obs[18]. No threat
                    # overlay (reverted to 53-dim, obs[18]=equity-vs-random).
                    return pred
        except Exception:
            pass

        # obs[18] = raw MC equity vs a uniform-random opponent range. The
        # threat-range overlay was reverted; _compute_obs18_threat_equity and
        # src/range_equity.py remain available for future re-wiring.
        if _EVAL7_AVAILABLE:
            try:
                e7_hole = [_eval7.Card(c) for c in hole_cards[:2]]
                e7_board = [_eval7.Card(c) for c in community_cards] if community_cards else []
                if n_opponents <= 1:
                    n_samples = 1000 if not community_cards else 300
                    hand_strength = _eval7.py_hand_vs_range_monte_carlo(
                        e7_hole, _EVAL7_RANDOM_RANGE, e7_board, n_samples)
                else:
                    hand_strength = self._multiway_equity(
                        e7_hole, e7_board, n_opponents, 500)
            except Exception:
                hand_strength = self._approx_preflop_equity(hole_cards)
        else:
            hand_strength = self._approx_preflop_equity(hole_cards)

        if not community_cards:
            return np.zeros(9, dtype=np.float32)

        if not _TREYS_AVAILABLE:
            return np.array([hand_strength, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        try:
            treys_hole = [TreysCard.new(c[0] + c[1].lower()) for c in hole_cards[:2]]
            treys_board = [TreysCard.new(c[0] + c[1].lower()) for c in community_cards]
        except Exception:
            return np.array([hand_strength, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        # Treys needs 5 or 7 total cards (2 hole + 3/4/5 board)
        total_cards = len(treys_hole) + len(treys_board)
        if total_cards not in (5, 6, 7):
            return np.array([hand_strength, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        try:
            rank_int = _TREYS_EVALUATOR.evaluate(treys_board, treys_hole)
            rank_class = _TREYS_EVALUATOR.get_rank_class(rank_int)
            made_hand_cat = (9 - rank_class) / 8.0
        except (KeyError, Exception):
            # Duplicate/invalid cards in the snapshot — no valid observation
            return None

        # Within-category rank: how good is this hand within its category?
        # 1.0 = best in category (e.g. nut flush), 0.0 = worst (e.g. bottom flush)
        lower, upper = _TREYS_RANK_BOUNDS.get(rank_class, (1, 7462))
        if upper > lower:
            hand_rank_within_cat = float((upper - rank_int) / (upper - lower))
        else:
            hand_rank_within_cat = 1.0

        board_ranks = [_rank_val(c[0]) for c in community_cards]
        hole_ranks = [r1, r2]
        max_board = max(board_ranks)
        top_pair = 1.0 if max_board in hole_ranks else 0.0

        # Hero flush participation: how many hero cards match the board's
        # dominant suit? Replaces has_flush_draw which falsely triggered on
        # 4-flush boards where hero had no matching cards.
        board_suit_counts = {}
        for c in community_cards:
            s = c[1]
            board_suit_counts[s] = board_suit_counts.get(s, 0) + 1
        dominant_suit = max(board_suit_counts, key=board_suit_counts.get)
        max_board_suited = board_suit_counts[dominant_suit]
        if max_board_suited >= 2:
            hero_match = sum(1 for c in hole_cards[:2] if c[1] == dominant_suit)
            hero_flush_part = hero_match / 2.0
        else:
            hero_flush_part = 0.0

        # Hero straight participation: does hero's card contribute to a
        # straight? Replaces has_straight_draw which triggered for ALL hands
        # on 4-straight boards (e.g., 6789K → every hand showed draw=1.0).
        board_ranks_set = set(board_ranks)
        if 14 in board_ranks_set:
            board_ranks_set.add(1)
        hero_ranks_set = set(hole_ranks)
        if 14 in hero_ranks_set:
            hero_ranks_set.add(1)
        hero_straight_part = 0.0
        for start in range(1, 11):
            window = set(range(start, start + 5))
            combined_in = len(window & (board_ranks_set | hero_ranks_set))
            hero_adds = combined_in - len(window & board_ranks_set)
            if combined_in >= 4 and hero_adds > 0:
                if combined_in == 5:
                    hero_straight_part = 1.0
                    break
                else:
                    hero_straight_part = max(hero_straight_part, 0.5)

        # Board pairing level: 0=unpaired, 0.33=single pair, 0.67=double
        # paired, 1.0=trips. Replaces binary board_is_paired which couldn't
        # distinguish paired boards from trips boards (the original bug).
        from collections import Counter
        board_rank_counts = Counter(board_ranks)
        max_rank_count = max(board_rank_counts.values())
        num_pairs = sum(1 for c in board_rank_counts.values() if c >= 2)
        if max_rank_count >= 3:
            board_pairing_level = 1.0
        elif num_pairs >= 2:
            board_pairing_level = 0.67
        elif num_pairs == 1:
            board_pairing_level = 0.33
        else:
            board_pairing_level = 0.0

        max_bs = max(board_suit_counts.values()) if board_suit_counts else 1
        board_flush_texture = float(np.clip((max_bs - 1) / 3.0, 0.0, 1.0))

        board_ranks_unique = sorted(set(board_ranks))
        if 14 in board_ranks_unique:
            board_ranks_unique = sorted(set([1] + board_ranks_unique))
        consecutive = sum(
            1 for i in range(len(board_ranks_unique) - 1)
            if board_ranks_unique[i + 1] - board_ranks_unique[i] == 1)
        max_possible = max(len(board_ranks_unique) - 1, 1)
        board_straight_texture = float(consecutive / max_possible)

        return np.array(
            [hand_strength, made_hand_cat, hero_flush_part, hero_straight_part,
             top_pair, board_pairing_level, board_flush_texture,
             board_straight_texture, hand_rank_within_cat],
            dtype=np.float32)

    def _equity_vs_random(self, hole_cards: List[str],
                          community_cards: List[str],
                          n_opponents: int = 1) -> float:
        """Raw MC equity vs a uniform-random opponent range (the original
        pre-threat-overlay obs[18]). Used for the appended equity_vs_random
        feature. Returns 0.0 on failure / preflop with no eval7."""
        if len(hole_cards) < 2:
            return 0.0
        if not _EVAL7_AVAILABLE:
            return self._approx_preflop_equity(hole_cards)
        try:
            e7_hole = [_eval7.Card(c) for c in hole_cards[:2]]
            e7_board = [_eval7.Card(c) for c in community_cards] if community_cards else []
            if n_opponents <= 1:
                ns = 1000 if not community_cards else 300
                return float(_eval7.py_hand_vs_range_monte_carlo(
                    e7_hole, _EVAL7_RANDOM_RANGE, e7_board, ns))
            return float(self._multiway_equity(e7_hole, e7_board, n_opponents, 300))
        except Exception:
            return self._approx_preflop_equity(hole_cards)

    def _compute_obs18_threat_equity(self, hole_cards: List[str],
                                       community_cards: List[str],
                                       players,
                                       hero_has_initiative=None,
                                       hand_raise_count=None,
                                       preflop_raise_count=None) -> float:
        """Tier-aware equity for obs[18]. Mirrors the engine's helper.

        Reads per-street aggression from `players[i].bet_current_round` and
        board texture from `community_cards`. Returns 0.0 on any error.

        `hero_has_initiative` (bool, optional): if False AND
        `hand_raise_count` > 0, opp has aggressed at some point this hand
        — tier lifts from 0 → 1 even on checked-through streets.
        """
        try:
            if len(hole_cards) < 2 or not community_cards:
                return 0.0
            from src.range_equity import (
                select_threat_tier, get_threat_range_equity)
            # Per-street tier signals from agent player state
            big_blind = 1.0   # parser normalizes to BB units
            max_opp_bet = max((getattr(p, 'bet_current_round', 0) for p in players),
                                default=0)
            pot_was_raised = 1.0 if max_opp_bet > big_blind else 0.0
            facing_3bet_plus = 1.0 if max_opp_bet > 6 * big_blind else 0.0

            # Multi-street aggression: did opp aggress at any prior point?
            # Signal True when hand has had raises AND hero is not the
            # most recent aggressor (opp is). Default False if unknown.
            opp_aggressed_this_hand = 0.0
            if (hero_has_initiative is not None
                    and hand_raise_count is not None
                    and not hero_has_initiative
                    and hand_raise_count > 0):
                opp_aggressed_this_hand = 1.0

            # Board textures (same as obs[24]/obs[25] later)
            board_ranks = [_rank_val(c[0]) for c in community_cards]
            board_suit_counts = {}
            for c in community_cards:
                s = c[1]
                board_suit_counts[s] = board_suit_counts.get(s, 0) + 1
            max_bs = max(board_suit_counts.values()) if board_suit_counts else 1
            board_flush_texture = float(np.clip((max_bs - 1) / 3.0, 0.0, 1.0))

            board_ranks_unique = sorted(set(board_ranks))
            if 14 in board_ranks_unique:
                board_ranks_unique = sorted(set([1] + board_ranks_unique))
            consecutive = sum(
                1 for i in range(len(board_ranks_unique) - 1)
                if board_ranks_unique[i + 1] - board_ranks_unique[i] == 1)
            max_possible = max(len(board_ranks_unique) - 1, 1)
            board_straight_texture = float(consecutive / max_possible)

            tier = select_threat_tier(
                num_raises_norm=0.0, max_bet_sizing=0.0,
                facing_3bet_plus=facing_3bet_plus,
                pot_was_raised=pot_was_raised,
                board_flush_texture=board_flush_texture,
                board_straight_texture=board_straight_texture,
                opp_aggressed_this_hand=opp_aggressed_this_hand,
                hand_raise_count=int(hand_raise_count or 0),
                preflop_raise_count=int(preflop_raise_count or 0))

            return get_threat_range_equity(
                hole_cards, community_cards, tier,
                spot_key=None, db_lookup=None, n_samples=300)
        except Exception:
            return 0.0

    @staticmethod
    def _multiway_equity(hole: list, board: list, n_opponents: int,
                         n_samples: int = 500) -> float:
        """Monte Carlo multiway equity: win rate vs N random opponents."""
        import random as _rng
        deck = [_eval7.Card(r + s) for r in '23456789TJQKA' for s in 'shdc']
        used = set(str(c) for c in hole + board)
        available = [c for c in deck if str(c) not in used]
        cards_needed = n_opponents * 2 + (5 - len(board))  # opp cards + remaining board

        if len(available) < cards_needed:
            return 0.5

        wins = 0
        for _ in range(n_samples):
            _rng.shuffle(available)
            idx = 0
            # Deal remaining board cards
            full_board = list(board) + available[idx:idx + (5 - len(board))]
            idx += 5 - len(board)
            # Evaluate hero
            my_score = _eval7.evaluate(list(hole) + full_board)
            # Evaluate opponents
            beat_all = True
            for _ in range(n_opponents):
                opp = [available[idx], available[idx + 1]]
                idx += 2
                opp_score = _eval7.evaluate(opp + full_board)
                if opp_score > my_score:  # higher = better in eval7
                    beat_all = False
                    break
            if beat_all:
                wins += 1

        return wins / max(n_samples, 1)

    @staticmethod
    def _approx_preflop_equity(hole_cards: List[str]) -> float:
        r1 = _rank_val(hole_cards[0][0])
        r2 = _rank_val(hole_cards[1][0])
        hi, lo = max(r1, r2), min(r1, r2)
        is_pair = (r1 == r2)
        is_suited = (hole_cards[0][1] == hole_cards[1][1])
        if is_pair:
            hs = 0.53 + (hi - 2) / 12.0 * 0.32
        else:
            gap_penalty = max(0, hi - lo - 1) * 0.01
            suited_bonus = 0.03 if is_suited else 0.0
            hs = (0.36 + (hi - 2) / 12.0 * 0.15
                  + (lo - 2) / 12.0 * 0.08
                  + suited_bonus - gap_penalty)
        return float(np.clip(hs, 0.0, 1.0))

    def _street_features(self, street: str) -> np.ndarray:
        idx = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}.get(street, 0)
        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[idx] = 1.0
        return one_hot

    def _game_state_features(self, gs: GameState, players: List[PlayerInfo],
                             player_idx: int) -> np.ndarray:
        alive = sum(1 for p in players if p.is_active)
        players_alive_norm = alive / self.num_players

        pot = gs.pot_total
        call_amount = gs.call_amount
        call_amount_norm = min(call_amount / max(pot, 1), 1.0)

        max_bet = max((p.bet_current_round for p in players), default=0)
        big_blind = 1.0  # in BB units
        ratio = max(max_bet, big_blind) / big_blind
        aggression_level = float(np.clip(
            math.log(max(ratio, 1)) / math.log(20), 0.0, 1.0))

        facing_bet = 1.0 if call_amount > 0 else 0.0
        bet_to_pot = float(np.clip(
            call_amount / max(pot - call_amount, 1), 0.0, 2.0)) / 2.0

        my_stack = players[player_idx].stack
        min_raise_inc = big_blind
        can_raise = 1.0 if (my_stack > call_amount and
                            min_raise_inc <= my_stack - call_amount) else 0.0

        return np.array(
            [players_alive_norm, call_amount_norm, aggression_level,
             facing_bet, bet_to_pot, can_raise],
            dtype=np.float32)

    def _betting_history_features(self, gs: GameState, players: List[PlayerInfo],
                                  player_idx: int) -> np.ndarray:
        player = players[player_idx]
        pot = max(gs.pot_total, 1)
        max_bet = max((p.bet_current_round for p in players), default=0)
        big_blind = 1.0

        # Estimate total_invested: use explicit value if available,
        # otherwise approximate from pot size and current bet
        invested = player.total_invested
        if invested <= 0:
            active = max(sum(1 for p in players if p.is_active), 1)
            all_bets = sum(p.bet_current_round for p in players)
            prev_streets = max(pot - all_bets, 0)
            invested = prev_streets / active + player.bet_current_round
            # Preflop: hero posted blind but bet_current_round may be 0
            # (parser reads it as 0 when blind is already in pot)
            if gs.street == 'preflop' and invested <= 0:
                invested = max(gs.small_blind, pot / active)

        my_pot_share = float(np.clip(invested / pot, 0.0, 1.0))

        active_count = max(sum(1 for p in players if p.is_active), 1)
        not_raise_fraction = float(np.clip(
            (active_count - 1) / max(self.num_players, 1), 0.0, 1.0))

        facing_3bet_plus = 1.0 if max_bet > 6 * big_blind else 0.0
        pot_was_raised = 1.0 if max_bet > big_blind else 0.0

        opp_stacks = [
            p.stack for i, p in enumerate(players)
            if i != player_idx and p.is_active
        ]
        min_opp_stack = min(opp_stacks) if opp_stacks else 0
        min_opp_stack_norm = float(np.clip(min_opp_stack / 100.0, 0.0, 1.0))

        return np.array(
            [my_pot_share, not_raise_fraction, facing_3bet_plus,
             pot_was_raised, min_opp_stack_norm],
            dtype=np.float32)

    def _behavioral_stats_features(
            self, gs: GameState, players: List[PlayerInfo],
            player_idx: int,
            hand_raise_count_override: Optional[int] = None,
            opp_bet_sizings_override: Optional[List[float]] = None,
            opp_action_counts_override: Optional[dict] = None,
            street_action_counts_override: Optional[dict] = None
            ) -> np.ndarray:
        """Compute 8-dim behavioral stats from current GameState snapshot.

        Mirrors src/hand_history/obs_builder._bh_compute() features:
          0. opp_aggression_ratio  — how aggressively opponents are betting
          1. opp_fold_ratio        — how many opponents have folded
          2. avg_bet_sizing        — average opponent bet size / pot
          3. max_bet_sizing        — max opponent bet size / pot
          4. num_raises_norm       — estimated raise count / 6
          5. street_aggression     — current street aggression level
          6. opp_check_ratio       — opponents checking on current street
          7. pot_escalation        — pot size / avg starting stack

        Overrides 2/3/4 are hand-level accumulators tracked by the caller
        across decisions — without them these slots collapse to 0 on every
        checked-to flop, mismatching how training computed the same dims
        from the full hand history.
        """
        feats = np.zeros(BEHAVIORAL_STATS_DIM, dtype=np.float32)

        # Use ALL players (not just active) to detect folds
        all_players = gs.players
        num_total = len(all_players)
        all_opps = [p for p in all_players if p.seat != gs.my_seat]
        active_opps = [p for p in all_opps if p.is_active and not p.is_sitting_out]
        folded_opps = [p for p in all_opps
                       if not p.is_active and not p.is_sitting_out and p.stack > 0]

        if not all_opps:
            return feats

        pot = max(gs.pot_total, 1.0)

        # 0 + 1: opp_aggression_ratio + opp_fold_ratio — prefer the hand-
        # level action counts tracked across decisions; snapshot fallback
        # collapses to a per-snapshot proxy that mismatches training (which
        # divides by total opp actions across the full hand).
        if opp_action_counts_override is not None:
            opp_total = max(int(opp_action_counts_override.get('total', 0)), 1)
            opp_agg = int(opp_action_counts_override.get('agg', 0))
            opp_fold = int(opp_action_counts_override.get('fold', 0))
            feats[0] = min(opp_agg / opp_total, 1.0)
            feats[1] = min(opp_fold / opp_total, 1.0)
        else:
            opp_with_bets = sum(1 for p in active_opps
                                if p.bet_current_round > 1.0 or p.is_all_in)
            opp_actions_est = max(len(active_opps), 1)
            feats[0] = opp_with_bets / opp_actions_est
            total_opps_in_hand = len(active_opps) + len(folded_opps)
            feats[1] = len(folded_opps) / max(total_opps_in_hand, 1)

        # 2 + 3: avg/max bet sizing — prefer the hand-level accumulator
        # (each entry already a bet/pot ratio captured at the time of the
        # bet), fall back to current snapshot if no accumulator supplied.
        if opp_bet_sizings_override:
            ratios = list(opp_bet_sizings_override)
            feats[2] = min(sum(ratios) / len(ratios) / 3.0, 1.0)
            feats[3] = min(max(ratios) / 3.0, 1.0)
        else:
            opp_bets = [p.bet_current_round for p in active_opps
                        if p.bet_current_round > 0]
            if opp_bets:
                pot_for_sizing = max(pot - sum(opp_bets), 1.0)
                avg_ratio = sum(b / pot_for_sizing
                                for b in opp_bets) / len(opp_bets)
                feats[2] = min(avg_ratio / 3.0, 1.0)
                max_bet = max(opp_bets)
                pot_for_max = max(pot - max_bet, 1.0)
                feats[3] = min((max_bet / pot_for_max) / 3.0, 1.0)

        # 4: num_raises_norm — prefer the hand-level counter tracked across
        # decisions; the snapshot fallback collapses to 0 on every
        # checked-to flop and mismatches training's hand-history derivation.
        if hand_raise_count_override is not None:
            feats[4] = min(max(0, int(hand_raise_count_override)) / 6.0, 1.0)
        else:
            max_bet_all = max(
                (p.bet_current_round for p in players), default=0)
            if max_bet_all > 1.0:
                est_raises = math.log2(max(max_bet_all, 1))
                feats[4] = min(est_raises / 6.0, 1.0)

        # 5 + 6: street_aggression + opp_check_ratio — prefer per-street
        # action counts. Training: agg_actions / all_actions on this street
        # (slot 5), opp_checks / opp_actions on this street (slot 6). The
        # snapshot fallback uses an opp-centric proxy that's directionally
        # similar but uses different denominators.
        street = gs.street or 'preflop'
        if street_action_counts_override is not None:
            sc = street_action_counts_override.get(street, {})
            total_st = int(sc.get('total', 0))
            agg_st = int(sc.get('agg', 0))
            opp_total_st = int(sc.get('opp_total', 0))
            opp_check_st = int(sc.get('opp_check', 0))
            feats[5] = min(agg_st / total_st, 1.0) if total_st > 0 else 0.0
            feats[6] = (min(opp_check_st / opp_total_st, 1.0)
                        if opp_total_st > 0 else 0.0)
        else:
            if active_opps:
                agg_count = sum(1 for p in active_opps
                                if p.bet_current_round > 0 or p.is_all_in)
                feats[5] = agg_count / len(active_opps)
                check_count = sum(1 for p in active_opps
                                  if p.bet_current_round == 0
                                  and not p.is_all_in)
                feats[6] = check_count / len(active_opps)

        # 7: pot_escalation — pot / avg starting stack
        # Approximate starting stack: current stack + pot share (since total_invested may be 0)
        avg_stack = max(sum(p.stack for p in all_players) / max(num_total, 1) + pot / max(num_total, 1), 1.0)
        feats[7] = min(pot / avg_stack, 1.0)

        return feats

    def _betting_line_features(self, gs: GameState, players: List[PlayerInfo],
                               player_idx: int,
                               hero_has_initiative_override: Optional[bool] = None
                               ) -> np.ndarray:
        """Compute 3-dim betting-line features.

        Mirrors src/engine/fast_enhanced_engine._get_betting_line_features
        and src/hand_history/obs_builder._bl_compute. Most fields are derived
        from the snapshot, but `hero_has_initiative` requires cross-street
        state — callers pass `hero_has_initiative_override`
        from a per-hand tracker. Without an override we fall back to a
        snapshot proxy that's only valid AFTER hero has bet this street.

        0. pot_at_street_start_norm — (pot - sum bet_current_round) / avg_stack
        1. hero_has_initiative — 1 if hero is the last aggressor in the hand
        2. agg_count_this_street_norm — log2(max_bet/BB) / 4 (proxy for raise
           count this street)
        """
        feats = np.zeros(BETTING_LINE_DIM, dtype=np.float32)

        all_players = gs.players
        num_total = max(len(all_players), 1)
        pot = max(gs.pot_total, 1.0)
        big_blind = 1.0  # gs uses BB-normalized chips

        # Pot at start of current street ≈ total pot minus what's been bet
        # this round so far.
        round_bets = sum(p.bet_current_round for p in all_players)
        prior_streets_pot = max(pot - round_bets, 0.0)
        avg_stack = max(
            sum(p.stack for p in all_players) / num_total + pot / num_total,
            1.0)
        feats[0] = float(np.clip(prior_streets_pot / avg_stack, 0.0, 1.0))

        if hero_has_initiative_override is not None:
            feats[1] = 1.0 if hero_has_initiative_override else 0.0
            max_bet = max(
                (p.bet_current_round for p in all_players), default=0)
        else:
            # Fallback snapshot proxy: degenerate before hero bets this
            # street (always 0 on a freshly-dealt flop) — known limitation,
            # caller should supply the override whenever possible.
            my_bet = players[player_idx].bet_current_round
            max_bet = max(
                (p.bet_current_round for p in all_players), default=0)
            feats[1] = 1.0 if my_bet > 0 and my_bet >= max_bet else 0.0

        # Aggression count this street ≈ log2 of how many BB the max bet is
        # at, normalized by /4 so 16BB ≈ 1.0.
        if max_bet > big_blind:
            feats[2] = float(np.clip(
                math.log2(max(max_bet / big_blind, 1.0)) / 4.0, 0.0, 1.0))

        return feats
