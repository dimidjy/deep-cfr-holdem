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
                 behavioral_stats: bool = False):
        self.num_players = num_players
        self.pad_to_6p = pad_to_6p and num_players < 6
        self.behavioral_stats = behavioral_stats

    def build(self, game_state: GameState) -> Optional[np.ndarray]:
        """Build observation vector from GameState. Returns None if hole cards missing."""
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
        features.append(self._equity_features(hole_cards, community))
        features.append(self._street_features(game_state.street))
        features.append(self._game_state_features(game_state, players, player_idx))
        features.append(self._betting_history_features(game_state, players, player_idx))

        if self.behavioral_stats:
            features.append(self._behavioral_stats_features(
                game_state, players, player_idx))

        return np.concatenate(features).astype(np.float32)

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
                         community_cards: List[str]) -> np.ndarray:
        if len(hole_cards) < 2:
            return np.zeros(8, dtype=np.float32)

        r1 = _rank_val(hole_cards[0][0])
        r2 = _rank_val(hole_cards[1][0])
        hi, lo = max(r1, r2), min(r1, r2)

        if _EVAL7_AVAILABLE:
            try:
                e7_hole = [_eval7.Card(c) for c in hole_cards[:2]]
                e7_board = [_eval7.Card(c) for c in community_cards] if community_cards else []
                n_samples = 1000 if not community_cards else 300
                hand_strength = _eval7.py_hand_vs_range_monte_carlo(
                    e7_hole, _EVAL7_RANDOM_RANGE, e7_board, n_samples)
            except Exception:
                hand_strength = self._approx_preflop_equity(hole_cards)
        else:
            hand_strength = self._approx_preflop_equity(hole_cards)

        if not community_cards:
            return np.array([hand_strength, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        if not _TREYS_AVAILABLE:
            return np.array([hand_strength, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        try:
            treys_hole = [TreysCard.new(c[0] + c[1].lower()) for c in hole_cards[:2]]
            treys_board = [TreysCard.new(c[0] + c[1].lower()) for c in community_cards]
        except Exception:
            return np.array([hand_strength, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        rank_int = _TREYS_EVALUATOR.evaluate(treys_board, treys_hole)
        rank_class = _TREYS_EVALUATOR.get_rank_class(rank_int)
        made_hand_cat = (9 - rank_class) / 8.0

        all_ranks = [_rank_val(c[0]) for c in hole_cards[:2]] + \
                    [_rank_val(c[0]) for c in community_cards]
        all_suits = [c[1] for c in hole_cards[:2]] + \
                    [c[1] for c in community_cards]

        suit_counts = {}
        for s in all_suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        has_flush_draw = 1.0 if any(v == 4 for v in suit_counts.values()) else 0.0

        ranks_all = sorted(set(all_ranks))
        if 14 in ranks_all:
            ranks_all = sorted(set([1] + ranks_all))
        has_straight_draw = 0.0
        for start in range(1, 12):
            if len(set(range(start, start + 5)) & set(ranks_all)) >= 4:
                has_straight_draw = 1.0
                break

        board_ranks = [_rank_val(c[0]) for c in community_cards]
        hole_ranks = [r1, r2]
        max_board = max(board_ranks)
        top_pair = 1.0 if max_board in hole_ranks else 0.0

        board_is_paired = 1.0 if len(board_ranks) != len(set(board_ranks)) else 0.0

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

        return np.array(
            [hand_strength, made_hand_cat, has_flush_draw, has_straight_draw,
             top_pair, board_is_paired, board_flush_texture,
             board_straight_texture],
            dtype=np.float32)

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

        my_pot_share = float(np.clip(player.total_invested / pot, 0.0, 1.0))

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

        num_total = len(gs.players)
        active = [p for p in players if p.is_active]
        opps = [p for i, p in enumerate(players) if i != player_idx]
        active_opps = [p for p in opps if p.is_active]
        folded_opps = [p for p in opps if not p.is_active]

        if not opps:
            return feats

        pot = max(gs.pot_total, 1.0)

        # 0: opp_aggression_ratio — opponents who bet/raised (have current bets > BB)
        opp_with_bets = sum(1 for p in opps if p.bet_current_round > 1.0 or p.is_all_in)
        opp_actions_est = max(len(opps), 1)
        feats[0] = opp_with_bets / opp_actions_est

        # 1: opp_fold_ratio — fraction of opponents who folded
        feats[1] = len(folded_opps) / max(num_total - 1, 1)

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
        avg_stack = max(sum(p.stack + p.total_invested for p in players) / max(num_total, 1), 1.0)
        feats[7] = min(pot / avg_stack, 1.0)

        return feats
