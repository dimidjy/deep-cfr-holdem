"""
FastEnhancedPokerEngine — feature extraction on top of the fast pokers backend.

Produces the same feature vector as EnhancedPokerEngine so trained models
are interchangeable between engines.
"""

import math
import numpy as np

from .fast_poker_engine import (
    FastPokerEngine,
    _pkrs_card_rank, _pkrs_card_suit,
    _PlayerStatus,
    STAGE_PREFLOP, STAGE_FLOP, STAGE_TURN, STAGE_RIVER,
)

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


class FastEnhancedPokerEngine(FastPokerEngine):
    """
    Drop-in replacement for EnhancedPokerEngine using the fast pokers backend.
    Produces identical feature vectors for the same game state.
    """

    def __init__(self, num_players: int = 6,
                 add_position_features: bool = True,
                 add_pot_odds_features: bool = True,
                 add_hand_strength_features: bool = True,
                 add_stack_features: bool = True,
                 add_equity_features: bool = True,
                 add_street_features: bool = True,
                 add_game_state_features: bool = True,
                 add_betting_history_features: bool = True,
                 legacy_game_state: bool = False):
        self.add_position_features = add_position_features
        self.add_pot_odds_features = add_pot_odds_features
        self.add_hand_strength_features = add_hand_strength_features
        self.add_stack_features = add_stack_features
        self.add_equity_features = add_equity_features and _TREYS_AVAILABLE
        self.add_street_features = add_street_features
        self.add_game_state_features = add_game_state_features
        self.add_betting_history_features = add_betting_history_features
        self.legacy_game_state = legacy_game_state  # 3 dims instead of 6

        self.num_players = num_players
        self.enhanced_obs_size = self._calculate_enhanced_size()
        super().__init__(num_players)

    def _calculate_enhanced_size(self) -> int:
        size = 0
        if self.add_position_features:
            size += self.num_players + 2
        if self.add_pot_odds_features:
            size += 3
        if self.add_hand_strength_features:
            size += 5
        if self.add_stack_features:
            size += 2
        if self.add_equity_features:
            size += 8
        if self.add_street_features:
            size += 4
        if self.add_game_state_features:
            size += 3 if self.legacy_game_state else 6
        if self.add_betting_history_features:
            size += 5
        return size

    def get_observation_space_size(self) -> int:
        return self.enhanced_obs_size

    def get_observation(self) -> np.ndarray:
        if self._done or not self.current_agent:
            return np.zeros(self.enhanced_obs_size, dtype=np.float32)

        features = []
        if self.add_position_features:
            features.append(self._get_position_features())
        if self.add_pot_odds_features:
            features.append(self._get_pot_odds_features())
        if self.add_hand_strength_features:
            features.append(self._get_hand_strength_features())
        if self.add_stack_features:
            features.append(self._get_stack_features())
        if self.add_equity_features:
            features.append(self._get_equity_features())
        if self.add_street_features:
            features.append(self._get_street_features())
        if self.add_game_state_features:
            features.append(self._get_game_state_features())
        if self.add_betting_history_features:
            features.append(self._get_betting_history_features())

        if features:
            return np.concatenate(features).astype(np.float32)
        return np.zeros(self.enhanced_obs_size, dtype=np.float32)

    # ── Helper: quick access to current state ────────────────────────────────

    def _cur_idx(self):
        return self.agents.index(self.current_agent)

    def _cur_ps(self):
        return self._state.players_state[self._cur_idx()]

    @staticmethod
    def _rank_to_value(rank_str: str) -> int:
        _map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
        }
        return _map.get(rank_str, 0)

    # ── Feature blocks (matching EnhancedPokerEngine exactly) ────────────────

    def _get_position_features(self) -> np.ndarray:
        try:
            agent_idx = self._cur_idx()
            dealer_idx = self._state.button
            rel_pos = (agent_idx - dealer_idx) % self.num_players
            one_hot = np.zeros(self.num_players, dtype=np.float32)
            one_hot[rel_pos] = 1.0
            norm_pos = rel_pos / self.num_players
            is_button = 1.0 if rel_pos == 0 else 0.0
            return np.array([*one_hot, norm_pos, is_button], dtype=np.float32)
        except Exception:
            return np.zeros(self.num_players + 2, dtype=np.float32)

    def _get_pot_odds_features(self) -> np.ndarray:
        try:
            ps = self._cur_ps()
            pot = self._state.pot
            my_chips = ps.stake
            total_chips = self.num_players * 100

            pot_normalized = float(np.clip(pot / max(total_chips, 1), 0.0, 1.0))
            raw_stp = my_chips / max(pot, 1)
            stack_to_pot = float(np.clip(
                math.log1p(raw_stp) / math.log1p(50), 0.0, 1.0))

            max_bet = max(p.bet_chips for p in self._state.players_state)
            # Call amount = difference in current round bets only
            call_amount = max(0, max_bet - ps.bet_chips)
            pot_odds = float(pot / max(pot + call_amount, 1)) if call_amount > 0 else 0.0

            return np.array([pot_normalized, stack_to_pot, pot_odds], dtype=np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)

    def _get_hand_strength_features(self) -> np.ndarray:
        try:
            ps = self._cur_ps()
            cards = ps.hand
            if len(cards) < 2:
                return np.zeros(5, dtype=np.float32)

            r0 = self._rank_to_value(_pkrs_card_rank(cards[0]))
            r1 = self._rank_to_value(_pkrs_card_rank(cards[1]))
            s0 = _pkrs_card_suit(cards[0])
            s1 = _pkrs_card_suit(cards[1])

            is_pair = 1.0 if r0 == r1 else 0.0
            is_suited = 1.0 if s0 == s1 else 0.0
            hi, lo = max(r0, r1), min(r0, r1)
            high_card_norm = (hi - 2) / 12.0
            low_card_norm = (lo - 2) / 12.0
            rank_gap_norm = (hi - lo) / 12.0

            return np.array(
                [is_pair, is_suited, high_card_norm, low_card_norm, rank_gap_norm],
                dtype=np.float32)
        except Exception:
            return np.zeros(5, dtype=np.float32)

    def _get_stack_features(self) -> np.ndarray:
        try:
            ps = self._cur_ps()
            my_stack = ps.stake
            total_chips = sum(
                p.stake + p.pot_chips + p.bet_chips
                for p in self._state.players_state)
            pot = self._state.pot

            m_ratio = min(my_stack / max(pot + 2, 1), 5.0) / 5.0
            relative_stack = my_stack / max(total_chips, 1)

            return np.array([m_ratio, relative_stack], dtype=np.float32)
        except Exception:
            return np.zeros(2, dtype=np.float32)

    def _get_equity_features(self) -> np.ndarray:
        try:
            ps = self._cur_ps()
            hole = ps.hand
            board_pkrs = self._state.public_cards

            if len(hole) < 2:
                return np.zeros(9, dtype=np.float32)

            r1 = self._rank_to_value(_pkrs_card_rank(hole[0]))
            r2 = self._rank_to_value(_pkrs_card_rank(hole[1]))
            hi, lo = max(r1, r2), min(r1, r2)

            # Monte Carlo equity vs random range (eval7)
            def _to_card_str(c):
                return _pkrs_card_rank(c) + _pkrs_card_suit(c).lower()

            if _EVAL7_AVAILABLE:
                try:
                    e7_hole = [_eval7.Card(_to_card_str(c)) for c in hole[:2]]
                    e7_board = [_eval7.Card(_to_card_str(c)) for c in board_pkrs] if board_pkrs else []
                    n_samples = 1000 if not board_pkrs else 300
                    hand_strength = _eval7.py_hand_vs_range_monte_carlo(
                        e7_hole, _EVAL7_RANDOM_RANGE, e7_board, n_samples)
                except Exception:
                    hand_strength = self._approx_preflop_equity(hole)
            else:
                hand_strength = self._approx_preflop_equity(hole)

            if not board_pkrs:
                return np.zeros(9, dtype=np.float32)

            # Postflop: treys for made hand category + texture features
            def _to_treys(c):
                return TreysCard.new(
                    _pkrs_card_rank(c) + _pkrs_card_suit(c).lower())

            treys_hole = [_to_treys(c) for c in hole[:2]]
            treys_board = [_to_treys(c) for c in board_pkrs]

            rank_int = _TREYS_EVALUATOR.evaluate(treys_board, treys_hole)
            rank_class = _TREYS_EVALUATOR.get_rank_class(rank_int)
            made_hand_cat = (9 - rank_class) / 8.0

            # Within-category rank: how good within its hand category?
            lower, upper = _TREYS_RANK_BOUNDS.get(rank_class, (1, 7462))
            if upper > lower:
                hand_rank_within_cat = float((upper - rank_int) / (upper - lower))
            else:
                hand_rank_within_cat = 1.0

            # Top pair
            board_ranks = [self._rank_to_value(_pkrs_card_rank(c))
                           for c in board_pkrs]
            hole_ranks = [r1, r2]
            max_board = max(board_ranks)
            top_pair = 1.0 if max_board in hole_ranks else 0.0

            # Hero flush participation
            board_suit_counts = {}
            for c in board_pkrs:
                s = _pkrs_card_suit(c)
                board_suit_counts[s] = board_suit_counts.get(s, 0) + 1
            dominant_suit = max(board_suit_counts, key=board_suit_counts.get)
            max_board_suited = board_suit_counts[dominant_suit]
            hero_suits = [_pkrs_card_suit(c) for c in hole[:2]]
            if max_board_suited >= 2:
                hero_match = sum(1 for s in hero_suits if s == dominant_suit)
                hero_flush_part = hero_match / 2.0
            else:
                hero_flush_part = 0.0

            # Hero straight participation
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

            # Board pairing level
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
        except Exception:
            return np.zeros(9, dtype=np.float32)

    def _approx_preflop_equity(self, hole) -> float:
        """Fallback preflop equity when eval7 is unavailable."""
        r1 = self._rank_to_value(_pkrs_card_rank(hole[0]))
        r2 = self._rank_to_value(_pkrs_card_rank(hole[1]))
        hi, lo = max(r1, r2), min(r1, r2)
        is_pair = (r1 == r2)
        is_suited = (_pkrs_card_suit(hole[0]) == _pkrs_card_suit(hole[1]))
        if is_pair:
            hs = 0.53 + (hi - 2) / 12.0 * 0.32
        else:
            gap_penalty = max(0, hi - lo - 1) * 0.01
            suited_bonus = 0.03 if is_suited else 0.0
            hs = (0.36 + (hi - 2) / 12.0 * 0.15
                  + (lo - 2) / 12.0 * 0.08
                  + suited_bonus - gap_penalty)
        return float(np.clip(hs, 0.0, 1.0))

    def _get_street_features(self) -> np.ndarray:
        try:
            s = str(self._state.stage)
            if 'Preflop' in s:
                idx = 0
            elif 'Flop' in s:
                idx = 1
            elif 'Turn' in s:
                idx = 2
            elif 'River' in s:
                idx = 3
            else:
                idx = 0
            one_hot = np.zeros(4, dtype=np.float32)
            one_hot[idx] = 1.0
            return one_hot
        except Exception:
            return np.zeros(4, dtype=np.float32)

    def _get_game_state_features(self) -> np.ndarray:
        try:
            ps = self._cur_ps()
            players = self._state.players_state

            alive = sum(1 for p in players if p.active or p.stake == 0)
            players_alive_norm = alive / self.num_players

            pot = self._state.pot
            max_bet = max(p.bet_chips for p in players)
            # Call amount = difference in current round bets only
            call_amount = max(0, max_bet - ps.bet_chips)
            call_amount_norm = min(call_amount / max(pot, 1), 1.0)

            big_blind = 2
            ratio = max(max_bet, big_blind) / big_blind
            aggression_level = float(np.clip(
                math.log(ratio) / math.log(20), 0.0, 1.0))

            # New facing-bet features
            facing_bet = 1.0 if call_amount > 0 else 0.0
            bet_to_pot = float(np.clip(
                call_amount / max(pot - call_amount, 1), 0.0, 2.0)) / 2.0

            # Can we raise? Check if stack covers more than the call
            my_stake = ps.stake
            can_raise = 1.0 if (my_stake > call_amount and
                                self._state.min_bet <= my_stake - call_amount) else 0.0

            if self.legacy_game_state:
                return np.array(
                    [players_alive_norm, call_amount_norm, aggression_level],
                    dtype=np.float32)
            return np.array(
                [players_alive_norm, call_amount_norm, aggression_level,
                 facing_bet, bet_to_pot, can_raise],
                dtype=np.float32)
        except Exception:
            if self.legacy_game_state:
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _get_betting_history_features(self) -> np.ndarray:
        try:
            ps = self._cur_ps()
            agent_idx = self._cur_idx()
            players = self._state.players_state

            big_blind = 2
            max_bet = max(p.bet_chips for p in players)
            pot = max(self._state.pot, 1)

            my_pot_share = float(np.clip(
                (ps.pot_chips + ps.bet_chips) / pot, 0.0, 1.0))

            # Raise count proxy: how many players have put in more than
            # the minimum required bet this round (indicates raise/re-raise)
            big_blind_amount = 2
            raise_count = sum(
                1 for p in players
                if p.bet_chips > big_blind_amount and p.active
            )
            not_raise_fraction = float(np.clip(
                raise_count / max(self.num_players, 1), 0.0, 1.0))

            facing_3bet_plus = 1.0 if max_bet > 6 * big_blind else 0.0
            pot_was_raised = 1.0 if max_bet > big_blind else 0.0

            # Min opponent stack (include all-in players, exclude only folded)
            # Match old engine: p.status != PlayerStatus.FOLDED
            opp_stacks = [
                p.stake for i, p in enumerate(players)
                if i != agent_idx and (p.active or p.stake == 0)
            ]
            min_opp_stack = min(opp_stacks) if opp_stacks else 0
            min_opp_stack_norm = float(np.clip(min_opp_stack / 100.0, 0.0, 1.0))

            return np.array(
                [my_pot_share, not_raise_fraction, facing_3bet_plus,
                 pot_was_raised, min_opp_stack_norm],
                dtype=np.float32)
        except Exception:
            return np.zeros(5, dtype=np.float32)
