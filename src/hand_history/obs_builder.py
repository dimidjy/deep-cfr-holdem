"""Reconstruct 41-dim observation vectors from parsed hand histories.

Replicates the exact feature computation from FastEnhancedPokerEngine
(src/engine/fast_enhanced_engine.py) using game state reconstructed
from the parsed action sequence.
"""

import math
from collections import defaultdict
from typing import List, Optional

import numpy as np

from .models import ParsedHand, ParsedAction, DecisionPoint, dollars_to_chips
from .action_mapper import map_action, get_legal_actions

try:
    from treys import Card as TreysCard, Evaluator as TreysEvaluator
    _TREYS_EVALUATOR = TreysEvaluator()
    _TREYS_AVAILABLE = True
except ImportError:
    _TREYS_AVAILABLE = False


_RANK_MAP = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
}

_POS_LABELS_6 = {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'UTG', 4: 'MP', 5: 'CO'}


def _rank_val(rank_char: str) -> int:
    return _RANK_MAP.get(rank_char, 0)


class _PlayerState:
    """Mirrors pokers PlayerState / FastPokerEngine._Player."""
    __slots__ = ('name', 'stake', 'pot_chips', 'bet_chips', 'active',
                 'hole_cards', 'seat')

    def __init__(self, name: str, stake: int, seat: int,
                 hole_cards: Optional[List[str]] = None):
        self.name = name
        self.stake = stake          # remaining chips (not yet wagered)
        self.pot_chips = 0          # chips wagered in previous streets
        self.bet_chips = 0          # chips wagered this round
        self.active = True          # False if folded
        self.hole_cards = hole_cards
        self.seat = seat


class ObservationBuilder:
    """Build observation vectors from parsed hand histories.

    Produces the same 41-dim (for 6-player) feature vector as
    FastEnhancedPokerEngine.
    """

    def __init__(self, num_players: int = 6):
        self.num_players = num_players

    def build_decision_points(self, hand: ParsedHand,
                              allow_unknown_cards: bool = False) -> List[DecisionPoint]:
        """Replay a hand and build DecisionPoint at each non-blind decision.

        Args:
            allow_unknown_cards: If True, build observations even for players
                without known hole cards (using neutral card features).
                Useful for supervised learning to include fold examples.
        """
        active_players = hand.active_players
        if len(active_players) != self.num_players:
            return []

        # Build player index by name (seat order among active players)
        name_to_idx = {}
        players: List[_PlayerState] = []
        for i, pp in enumerate(active_players):
            name_to_idx[pp.name] = i
            players.append(_PlayerState(
                name=pp.name,
                stake=pp.stack_chips,
                seat=pp.seat,
                hole_cards=pp.hole_cards
            ))

        # Find button index
        button_idx = self._find_button_idx(active_players, hand.button_seat)

        # Community cards available at each street
        cc = hand.community_cards_at_street
        current_street = 'preflop'
        community_cards: List[str] = []

        decision_points: List[DecisionPoint] = []
        player_action_count = defaultdict(int)  # name -> count of non-blind actions

        for action in hand.actions:
            if action.player_name not in name_to_idx:
                continue

            pidx = name_to_idx[action.player_name]
            player = players[pidx]

            # Handle street transitions
            if action.street != current_street:
                # Move bet_chips -> pot_chips for all players
                for p in players:
                    p.pot_chips += p.bet_chips
                    p.bet_chips = 0
                current_street = action.street
                community_cards = cc.get(current_street, [])

            # Skip blind actions (not decisions)
            if action.is_blind:
                self._apply_blind(player, action)
                continue

            # Skip actions from folded/all-in players
            if not player.active or player.stake == 0:
                continue

            # ── Build observation and record decision point ───────────
            pot = self._compute_pot(players)
            max_bet = max(p.bet_chips for p in players)

            legal = get_legal_actions(
                max_bet=max_bet,
                player_bet_chips=player.bet_chips,
                player_stake=player.stake,
                pot=pot,
            )

            mapped_action = map_action(
                action=action,
                pot=pot,
                max_bet=max_bet,
                player_bet_chips=player.bet_chips,
                player_stake=player.stake,
            )

            obs = self._build_observation(
                players=players,
                player_idx=pidx,
                button_idx=button_idx,
                street=current_street,
                community_cards=community_cards,
                allow_unknown_cards=allow_unknown_cards,
            )

            has_cards = bool(player.hole_cards)
            if obs is not None and (has_cards or allow_unknown_cards):
                rel_pos = (pidx - button_idx) % self.num_players
                pos_label = _POS_LABELS_6.get(rel_pos, 'MP')

                decision_points.append(DecisionPoint(
                    player_name=player.name,
                    observation=obs,
                    legal_actions=legal,
                    actual_action=mapped_action,
                    street=current_street,
                    hole_cards=player.hole_cards,
                    community_cards=list(community_cards),
                    position_label=pos_label,
                    player_idx=pidx,
                    hand_id=hand.hand_id,
                    action_index=player_action_count[player.name],
                ))
                player_action_count[player.name] += 1

            # ── Apply action to state ─────────────────────────────────
            self._apply_action(player, action, max_bet)

        return decision_points

    def _find_button_idx(self, active_players, button_seat: int) -> int:
        """Find button index in the active player list."""
        for i, p in enumerate(active_players):
            if p.seat == button_seat:
                return i
        return 0

    def _compute_pot(self, players: List[_PlayerState]) -> int:
        return sum(p.pot_chips + p.bet_chips for p in players)

    def _apply_blind(self, player: _PlayerState, action: ParsedAction):
        """Apply a blind posting."""
        chips = action.amount_chips
        actual = min(chips, player.stake)
        player.bet_chips += actual
        player.stake -= actual

    def _apply_action(self, player: _PlayerState, action: ParsedAction,
                      max_bet: int):
        """Update player state after an action."""
        if action.action_type == 'fold':
            player.active = False
            return

        if action.action_type == 'check':
            return

        if action.action_type == 'call':
            call_cost = max_bet - player.bet_chips
            actual = min(call_cost, player.stake)
            player.bet_chips += actual
            player.stake -= actual
            if action.is_all_in:
                player.stake = 0
            return

        if action.action_type in ('bet', 'raise'):
            if action.action_type == 'bet':
                # Bet: new_bet_chips = amount
                new_total_bet = action.amount_chips
            else:
                # Raise "to X": new total bet_chips = X in chips
                new_total_bet = action.amount_chips

            cost = new_total_bet - player.bet_chips
            actual = min(cost, player.stake)
            player.bet_chips += actual
            player.stake -= actual
            if action.is_all_in:
                player.stake = 0
            return

    # ── Observation building ──────────────────────────────────────────────

    def _build_observation(self, players: List[_PlayerState],
                           player_idx: int, button_idx: int,
                           street: str,
                           community_cards: List[str],
                           allow_unknown_cards: bool = False) -> Optional[np.ndarray]:
        """Build the 41-dim observation vector.

        Mirrors FastEnhancedPokerEngine.get_observation() exactly.
        If allow_unknown_cards is True and hole cards are missing,
        uses neutral/average values for card-dependent features.
        """
        player = players[player_idx]
        if not player.hole_cards and not allow_unknown_cards:
            return None

        features = []
        features.append(self._position_features(player_idx, button_idx))
        features.append(self._pot_odds_features(players, player_idx))

        hole = player.hole_cards
        if hole:
            features.append(self._hand_strength_features(hole))
            features.append(self._stack_features(players, player_idx))
            features.append(self._equity_features(hole, community_cards))
        else:
            # Neutral card features for unknown hands
            features.append(np.array([0.5, 0.5, 0.5, 0.5, 0.2],
                                     dtype=np.float32))  # avg hand
            features.append(self._stack_features(players, player_idx))
            features.append(np.array([0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     dtype=np.float32))  # avg equity

        features.append(self._street_features(street))
        features.append(self._game_state_features(players, player_idx))
        features.append(self._betting_history_features(players, player_idx))

        return np.concatenate(features).astype(np.float32)

    def _position_features(self, player_idx: int, button_idx: int) -> np.ndarray:
        """8 dims: one-hot(6) + normalized_pos + is_button."""
        rel_pos = (player_idx - button_idx) % self.num_players
        one_hot = np.zeros(self.num_players, dtype=np.float32)
        one_hot[rel_pos] = 1.0
        norm_pos = rel_pos / self.num_players
        is_button = 1.0 if rel_pos == 0 else 0.0
        return np.array([*one_hot, norm_pos, is_button], dtype=np.float32)

    def _pot_odds_features(self, players: List[_PlayerState],
                           player_idx: int) -> np.ndarray:
        """3 dims: pot_normalized, stack_to_pot, pot_odds."""
        player = players[player_idx]
        pot = self._compute_pot(players)
        my_chips = player.stake
        total_chips = self.num_players * 100

        pot_normalized = float(np.clip(pot / max(total_chips, 1), 0.0, 1.0))
        raw_stp = my_chips / max(pot, 1)
        stack_to_pot = float(np.clip(
            math.log1p(raw_stp) / math.log1p(50), 0.0, 1.0))

        max_bet = max(p.bet_chips for p in players)
        my_in_chips = player.pot_chips + player.bet_chips
        call_amount = max(0, max_bet - my_in_chips)
        pot_odds = float(pot / max(pot + call_amount, 1)) if call_amount > 0 else 0.0

        return np.array([pot_normalized, stack_to_pot, pot_odds], dtype=np.float32)

    def _hand_strength_features(self, hole_cards: List[str]) -> np.ndarray:
        """5 dims: is_pair, is_suited, high_card_norm, low_card_norm, rank_gap_norm."""
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

    def _stack_features(self, players: List[_PlayerState],
                        player_idx: int) -> np.ndarray:
        """2 dims: m_ratio, relative_stack."""
        player = players[player_idx]
        my_stack = player.stake
        total_chips = sum(
            p.stake + p.pot_chips + p.bet_chips for p in players)
        pot = self._compute_pot(players)

        m_ratio = min(my_stack / max(pot + 2, 1), 5.0) / 5.0
        relative_stack = my_stack / max(total_chips, 1)

        return np.array([m_ratio, relative_stack], dtype=np.float32)

    def _equity_features(self, hole_cards: List[str],
                         community_cards: List[str]) -> np.ndarray:
        """8 dims: hand_strength + board texture features."""
        if len(hole_cards) < 2:
            return np.zeros(8, dtype=np.float32)

        r1 = _rank_val(hole_cards[0][0])
        r2 = _rank_val(hole_cards[1][0])
        hi, lo = max(r1, r2), min(r1, r2)
        is_pair = (r1 == r2)
        is_suited = (hole_cards[0][1] == hole_cards[1][1])

        if not community_cards:
            # Preflop equity approximation
            if is_pair:
                hs = 0.53 + (hi - 2) / 12.0 * 0.32
            else:
                gap_penalty = max(0, hi - lo - 1) * 0.01
                suited_bonus = 0.03 if is_suited else 0.0
                hs = (0.36
                      + (hi - 2) / 12.0 * 0.15
                      + (lo - 2) / 12.0 * 0.08
                      + suited_bonus - gap_penalty)
            hs = float(np.clip(hs, 0.0, 1.0))
            return np.array([hs, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        if not _TREYS_AVAILABLE:
            return np.zeros(8, dtype=np.float32)

        # Postflop: treys evaluation
        try:
            treys_hole = [TreysCard.new(c[0] + c[1].lower()) for c in hole_cards[:2]]
            treys_board = [TreysCard.new(c[0] + c[1].lower()) for c in community_cards]
        except Exception:
            return np.zeros(8, dtype=np.float32)

        rank_int = _TREYS_EVALUATOR.evaluate(treys_board, treys_hole)
        hand_strength = (7462 - rank_int) / 7462.0

        rank_class = _TREYS_EVALUATOR.get_rank_class(rank_int)
        made_hand_cat = (9 - rank_class) / 8.0

        # Flush/straight draws
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

        # Top pair
        board_ranks = [_rank_val(c[0]) for c in community_cards]
        hole_ranks = [r1, r2]
        max_board = max(board_ranks)
        top_pair = 1.0 if max_board in hole_ranks else 0.0

        # Board texture
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

    def _street_features(self, street: str) -> np.ndarray:
        """4 dims: one-hot [preflop, flop, turn, river]."""
        idx = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}.get(street, 0)
        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[idx] = 1.0
        return one_hot

    def _game_state_features(self, players: List[_PlayerState],
                             player_idx: int) -> np.ndarray:
        """6 dims: players_alive_norm, call_amount_norm, aggression_level,
           facing_bet, bet_to_pot, can_raise."""
        player = players[player_idx]

        # alive = active or all-in (stake==0 but not folded)
        alive = sum(1 for p in players if p.active or (p.stake == 0 and p.pot_chips + p.bet_chips > 0))
        players_alive_norm = alive / self.num_players

        pot = self._compute_pot(players)
        max_bet = max(p.bet_chips for p in players)
        my_in_chips = player.pot_chips + player.bet_chips
        call_amount = max(0, max_bet - my_in_chips)
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
        # min_bet approximation: use big_blind as min raise increment
        min_raise = big_blind
        can_raise = 1.0 if (player.stake > call_amount and
                            min_raise <= player.stake - call_amount) else 0.0

        return np.array(
            [players_alive_norm, call_amount_norm, aggression_level,
             facing_bet, bet_to_pot, can_raise],
            dtype=np.float32)

    def _betting_history_features(self, players: List[_PlayerState],
                                  player_idx: int) -> np.ndarray:
        """5 dims: my_pot_share, not_raise_fraction, facing_3bet_plus,
           pot_was_raised, min_opp_stack_norm."""
        player = players[player_idx]
        big_blind = 2
        max_bet = max(p.bet_chips for p in players)
        pot = max(self._compute_pot(players), 1)

        my_pot_share = float(np.clip(
            (player.pot_chips + player.bet_chips) / pot, 0.0, 1.0))

        # Approximate not_raise_fraction (matches FastEnhancedPokerEngine)
        active_count = max(sum(1 for p in players if p.active), 1)
        not_raise_fraction = float(np.clip(
            (active_count - 1) / max(self.num_players, 1), 0.0, 1.0))

        facing_3bet_plus = 1.0 if max_bet > 6 * big_blind else 0.0
        pot_was_raised = 1.0 if max_bet > big_blind else 0.0

        # Min opponent stack (active or all-in, exclude self)
        opp_stacks = [
            p.stake for i, p in enumerate(players)
            if i != player_idx and (p.active or p.stake == 0)
        ]
        min_opp_stack = min(opp_stacks) if opp_stacks else 0
        min_opp_stack_norm = float(np.clip(min_opp_stack / 100.0, 0.0, 1.0))

        return np.array(
            [my_pot_share, not_raise_fraction, facing_3bet_plus,
             pot_was_raised, min_opp_stack_norm],
            dtype=np.float32)
