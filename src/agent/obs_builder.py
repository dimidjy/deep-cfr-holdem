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


class ObservationBuilder:
    """Build observation vectors from GameState for model inference.

    When pad_to_6p=True, always outputs 41-dim (6-player format) observations,
    mapping positions to their 6p equivalents. This allows using a single 6p model
    for all table sizes (2-6 players).
    """

    def __init__(self, num_players: int = 6, pad_to_6p: bool = False,
                 behavioral_stats: bool = False, equity_vs_odds: bool = False):
        self.num_players = num_players
        self.pad_to_6p = pad_to_6p and num_players < 6
        self.behavioral_stats = behavioral_stats
        self.equity_vs_odds = equity_vs_odds

    def build(self, game_state: GameState,
              opponent_profile: np.ndarray = None) -> Optional[np.ndarray]:
        """Build observation vector from GameState.

        Args:
            game_state: Current game state from parser.
            opponent_profile: Optional 5-dim cross-hand opponent stats
                (VPIP, PFR, AF_norm, Fold2CBet, WTSD). If None, skipped.

        Returns None if hole cards missing.
        """
        players = game_state.active_players
        if not players:
            return None

        my_player = game_state.my_player
        if not my_player or not my_player.hole_cards:
            return None

        # Find indices
        player_idx = 0
        button_idx = 0
        for i, p in enumerate(players):
            if p.seat == game_state.my_seat:
                player_idx = i
            if p.seat == game_state.button_seat:
                button_idx = i

        hole_cards = [str(c) for c in my_player.hole_cards]
        community = [str(c) for c in game_state.community_cards]

        features = []
        features.append(self._position_features(player_idx, button_idx))
        features.append(self._pot_odds_features(game_state, players, player_idx))
        features.append(self._hand_strength_features(hole_cards))
        features.append(self._stack_features(game_state, players, player_idx))
        n_opponents = max(len([p for p in players if p.is_active]) - 1, 1)
        features.append(self._equity_features(hole_cards, community, n_opponents))
        features.append(self._street_features(game_state.street))
        features.append(self._game_state_features(game_state, players, player_idx))
        features.append(self._betting_history_features(game_state, players, player_idx))

        if self.behavioral_stats:
            features.append(self._behavioral_stats_features(
                game_state, players, player_idx))

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

        return obs

    def get_position_label(self, game_state: GameState) -> str:
        players = game_state.active_players
        player_idx = 0
        button_idx = 0
        for i, p in enumerate(players):
            if p.seat == game_state.my_seat:
                player_idx = i
            if p.seat == game_state.button_seat:
                button_idx = i
        rel_pos = (player_idx - button_idx) % self.num_players
        return _POS_LABELS.get(rel_pos, 'MP')

    def _position_features(self, player_idx: int, button_idx: int) -> np.ndarray:
        rel_pos = (player_idx - button_idx) % self.num_players

        if self.pad_to_6p:
            # Map to 6p position and build 8-dim (6+2) position block
            mapped_pos = _POS_MAP_TO_6P[self.num_players].get(rel_pos, 0)
            one_hot = np.zeros(6, dtype=np.float32)
            one_hot[mapped_pos] = 1.0
            norm_pos = mapped_pos / 6.0
            is_button = 1.0 if rel_pos == 0 else 0.0
            return np.array([*one_hot, norm_pos, is_button], dtype=np.float32)
        else:
            one_hot = np.zeros(self.num_players, dtype=np.float32)
            one_hot[rel_pos] = 1.0
            norm_pos = rel_pos / self.num_players
            is_button = 1.0 if rel_pos == 0 else 0.0
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
                         n_opponents: int = 1) -> np.ndarray:
        if len(hole_cards) < 2:
            return np.zeros(9, dtype=np.float32)

        r1 = _rank_val(hole_cards[0][0])
        r2 = _rank_val(hole_cards[1][0])
        hi, lo = max(r1, r2), min(r1, r2)

        if _EVAL7_AVAILABLE:
            try:
                e7_hole = [_eval7.Card(c) for c in hole_cards[:2]]
                e7_board = [_eval7.Card(c) for c in community_cards] if community_cards else []
                if n_opponents <= 1:
                    n_samples = 1000 if not community_cards else 300
                    hand_strength = _eval7.py_hand_vs_range_monte_carlo(
                        e7_hole, _EVAL7_RANDOM_RANGE, e7_board, n_samples)
                else:
                    # Multiway equity: simulate vs N random opponents
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

    def _behavioral_stats_features(self, gs: GameState, players: List[PlayerInfo],
                                   player_idx: int) -> np.ndarray:
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

        # 0: opp_aggression_ratio — opponents who bet/raised (have current bets > BB)
        opp_with_bets = sum(1 for p in active_opps if p.bet_current_round > 1.0 or p.is_all_in)
        opp_actions_est = max(len(active_opps), 1)
        feats[0] = opp_with_bets / opp_actions_est

        # 1: opp_fold_ratio — fraction of opponents who folded this hand
        total_opps_in_hand = len(active_opps) + len(folded_opps)
        feats[1] = len(folded_opps) / max(total_opps_in_hand, 1)

        # 2: avg_bet_sizing — average opp bet / pot (normalized by /3)
        opp_bets = [p.bet_current_round for p in active_opps if p.bet_current_round > 0]
        if opp_bets:
            pot_for_sizing = max(pot - sum(opp_bets), 1.0)
            avg_ratio = sum(b / pot_for_sizing for b in opp_bets) / len(opp_bets)
            feats[2] = min(avg_ratio / 3.0, 1.0)

        # 3: max_bet_sizing — max opp bet / pot (normalized by /3)
        if opp_bets:
            max_bet = max(opp_bets)
            pot_for_sizing = max(pot - max_bet, 1.0)
            feats[3] = min((max_bet / pot_for_sizing) / 3.0, 1.0)

        # 4: num_raises_norm — estimate from max bet level / 6
        max_bet_all = max((p.bet_current_round for p in players), default=0)
        if max_bet_all > 1.0:
            est_raises = math.log2(max(max_bet_all, 1))
            feats[4] = min(est_raises / 6.0, 1.0)

        # 5: street_aggression — are opponents betting on this street?
        if active_opps:
            agg_count = sum(1 for p in active_opps
                           if p.bet_current_round > 0 or p.is_all_in)
            feats[5] = agg_count / len(active_opps)

        # 6: opp_check_ratio — opponents with zero current bet (checked)
        if active_opps:
            check_count = sum(1 for p in active_opps
                             if p.bet_current_round == 0 and not p.is_all_in)
            feats[6] = check_count / len(active_opps)

        # 7: pot_escalation — pot / avg starting stack
        # Approximate starting stack: current stack + pot share (since total_invested may be 0)
        avg_stack = max(sum(p.stack for p in all_players) / max(num_total, 1) + pot / max(num_total, 1), 1.0)
        feats[7] = min(pot / avg_stack, 1.0)

        return feats
