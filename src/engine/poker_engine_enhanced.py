import numpy as np
from typing import List
from .poker_engine import PokerEngine

try:
    from treys import Card as TreysCard, Evaluator as TreysEvaluator
    _TREYS_EVALUATOR = TreysEvaluator()
    _TREYS_AVAILABLE = True
except ImportError:
    _TREYS_AVAILABLE = False


class EnhancedPokerEngine(PokerEngine):

    def __init__(self, num_players: int = 2,
                 add_position_features: bool = True,
                 add_pot_odds_features: bool = True,
                 add_hand_strength_features: bool = True,
                 add_stack_features: bool = True,
                 add_equity_features: bool = True,
                 add_street_features: bool = True,
                 add_game_state_features: bool = True,
                 add_betting_history_features: bool = True):
        super().__init__(num_players)
        self.add_position_features = add_position_features
        self.add_pot_odds_features = add_pot_odds_features
        self.add_hand_strength_features = add_hand_strength_features
        self.add_stack_features = add_stack_features
        self.add_equity_features = add_equity_features and _TREYS_AVAILABLE
        self.add_street_features = add_street_features
        self.add_game_state_features = add_game_state_features
        self.add_betting_history_features = add_betting_history_features

        self.enhanced_obs_size = self._calculate_enhanced_size()

    def _calculate_enhanced_size(self) -> int:
        # Raw card encoding dropped — all info captured by features below
        size = 0

        if self.add_position_features:
            size += self.num_players  # position one-hot
            size += 1                  # normalized relative position
            size += 1                  # is_button flag

        if self.add_pot_odds_features:
            size += 1  # pot_normalized
            size += 1  # stack_to_pot
            size += 1  # pot_odds

        if self.add_hand_strength_features:
            # is_pair, is_suited, high_card_norm, low_card_norm, rank_gap_norm
            size += 5

        if self.add_stack_features:
            size += 2  # m_ratio, relative_stack

        if self.add_equity_features:
            size += 8  # hand_strength, made_hand_cat, has_flush_draw, has_straight_draw,
                       # top_pair, board_is_paired, board_flush_texture, board_straight_texture

        if self.add_street_features:
            size += 4  # preflop, flop, turn, river (one-hot)

        if self.add_game_state_features:
            size += 3  # players_alive_norm, call_amount_norm, aggression_level

        if self.add_betting_history_features:
            size += 5  # raise_level, my_pot_share, not_raise_fraction,
                       # facing_3bet_plus, pot_was_raised

        return size

    def get_observation_space_size(self) -> int:
        return self.enhanced_obs_size

    def _get_position_features(self, current_agent: str) -> np.ndarray:
        try:
            game = self._get_game()
            if game:
                agent_idx = self.agents.index(current_agent)
                dealer_idx = game.dealer_id

                # Relative position from button: 0=button, 1=SB, 2=BB, 3=UTG, ...
                # This generalizes across seats — the network learns "button raises"
                # not "player_0 raises", which is what matters for positional play.
                rel_pos = (agent_idx - dealer_idx) % self.num_players
                position_one_hot = np.zeros(self.num_players, dtype=np.float32)
                position_one_hot[rel_pos] = 1.0

                normalized_position = rel_pos / self.num_players
                is_button = 1.0 if rel_pos == 0 else 0.0

                return np.array(
                    [*position_one_hot, normalized_position, is_button],
                    dtype=np.float32
                )
        except Exception:
            pass

        return np.zeros(self.num_players + 2, dtype=np.float32)

    def _get_pot_odds_features(self, current_agent: str) -> np.ndarray:
        try:
            game = self._get_game()
            if game:
                agent_idx = self.agents.index(current_agent)
                player = game.players[agent_idx]

                pot = sum(p.in_chips for p in game.players)
                my_chips = player.remained_chips

                pot_normalized = min(pot / 100.0, 1.0)
                stack_to_pot = min(my_chips / max(pot, 1), 2.0) / 2.0
                pot_odds = pot / max(pot + my_chips, 1) if pot > 0 else 0.0

                return np.array([pot_normalized, stack_to_pot, pot_odds], dtype=np.float32)
        except Exception:
            pass

        return np.zeros(3, dtype=np.float32)

    def _get_hand_strength_features(self, current_agent: str) -> np.ndarray:
        try:
            game = self._get_game()
            if game:
                agent_idx = self.agents.index(current_agent)
                player_cards = game.players[agent_idx].hand

                if len(player_cards) >= 2:
                    ranks = [self._rank_to_value(card.rank) for card in player_cards[:2]]
                    suits = [card.suit for card in player_cards[:2]]

                    is_pair = 1.0 if ranks[0] == ranks[1] else 0.0
                    is_suited = 1.0 if suits[0] == suits[1] else 0.0

                    high_rank = max(ranks)
                    low_rank = min(ranks)
                    # Normalize: rank 2 → 0.0, Ace (14) → 1.0
                    high_card_norm = (high_rank - 2) / 12.0
                    low_card_norm = (low_rank - 2) / 12.0
                    rank_gap_norm = (high_rank - low_rank) / 12.0

                    return np.array(
                        [is_pair, is_suited, high_card_norm, low_card_norm, rank_gap_norm],
                        dtype=np.float32
                    )
        except Exception:
            pass

        return np.zeros(5, dtype=np.float32)

    def _get_stack_features(self, current_agent: str) -> np.ndarray:
        try:
            game = self._get_game()
            if game:
                agent_idx = self.agents.index(current_agent)
                player = game.players[agent_idx]

                my_stack = player.remained_chips
                total_chips = sum(p.remained_chips + p.in_chips for p in game.players)
                pot = sum(p.in_chips for p in game.players)

                # M-ratio: stack relative to pot (normalized, capped at 5x)
                big_blind = 2
                m_ratio = min(my_stack / max(pot + big_blind, 1), 5.0) / 5.0

                # Relative stack: my share of total chips in play
                relative_stack = my_stack / max(total_chips, 1)

                return np.array([m_ratio, relative_stack], dtype=np.float32)
        except Exception:
            pass

        return np.zeros(2, dtype=np.float32)

    @staticmethod
    def _to_treys_card(card) -> int:
        """Convert RLCard card object to treys integer format."""
        return TreysCard.new(card.rank + card.suit.lower())

    def _get_equity_features(self, current_agent: str) -> np.ndarray:
        """
        8 features:
          hand_strength          – [0,1] preflop equity approx OR postflop treys percentile
          made_hand_cat          – [0,1] hand category (0=high-card, 1=str-flush), postflop only
          has_flush_draw         – 1 if player has 4 cards to a flush (postflop)
          has_straight_draw      – 1 if player has 4-card straight draw (postflop)
          top_pair               – 1 if player paired the highest board card (postflop)
          board_is_paired        – 1 if board contains a pair (postflop)
          board_flush_texture    – [0,1] board wetness by suit: 0=rainbow, 1=4-flush on board
          board_straight_texture – [0,1] board connectedness: 0=disconnected, 1=fully connected
        """
        try:
            game = self._get_game()
            if not game:
                return np.zeros(8, dtype=np.float32)

            agent_idx = self.agents.index(current_agent)
            hole = game.players[agent_idx].hand
            board = game.public_cards

            if len(hole) < 2:
                return np.zeros(8, dtype=np.float32)

            r1 = self._rank_to_value(hole[0].rank)
            r2 = self._rank_to_value(hole[1].rank)
            hi, lo = max(r1, r2), min(r1, r2)
            is_pair   = (r1 == r2)
            is_suited = (hole[0].suit == hole[1].suit)

            if not board:
                # ── Preflop: equity approximation (no MC, instant) ──────────
                if is_pair:
                    hand_strength = 0.53 + (hi - 2) / 12.0 * 0.32
                else:
                    gap_penalty  = max(0, hi - lo - 1) * 0.01
                    suited_bonus = 0.03 if is_suited else 0.0
                    hand_strength = (0.36
                                     + (hi - 2) / 12.0 * 0.15
                                     + (lo - 2) / 12.0 * 0.08
                                     + suited_bonus
                                     - gap_penalty)
                hand_strength = float(np.clip(hand_strength, 0.0, 1.0))
                return np.array([hand_strength, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                dtype=np.float32)

            # ── Postflop: treys evaluation ───────────────────────────────────
            treys_hole  = [self._to_treys_card(c) for c in hole[:2]]
            treys_board = [self._to_treys_card(c) for c in board]

            rank_int      = _TREYS_EVALUATOR.evaluate(treys_board, treys_hole)
            hand_strength = (7462 - rank_int) / 7462.0   # 1.0=nuts, 0.0=worst

            rank_class    = _TREYS_EVALUATOR.get_rank_class(rank_int)
            # treys classes: 1=SF 2=Quads 3=FH 4=Flush 5=Straight 6=Trips
            #                7=TwoPair 8=Pair 9=HighCard
            made_hand_cat = (9 - rank_class) / 8.0   # 0=HighCard, 1=StraightFlush

            # ── Flush / straight draws (player) ─────────────────────────────
            all_cards   = list(hole[:2]) + list(board)
            suit_counts = {}
            for c in all_cards:
                suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
            has_flush_draw = 1.0 if any(v == 4 for v in suit_counts.values()) else 0.0

            ranks_all = sorted(set(self._rank_to_value(c.rank) for c in all_cards))
            if 14 in ranks_all:
                ranks_all = sorted(set([1] + ranks_all))
            has_straight_draw = 0.0
            for start in range(1, 12):
                if len(set(range(start, start + 5)) & set(ranks_all)) >= 4:
                    has_straight_draw = 1.0
                    break

            # ── Top pair: player's hole card matches highest board rank ──────
            board_ranks = [self._rank_to_value(c.rank) for c in board]
            hole_ranks  = [r1, r2]
            max_board   = max(board_ranks)
            top_pair    = 1.0 if max_board in hole_ranks else 0.0

            # ── Board is paired: any rank appears 2+ times on board ──────────
            board_is_paired = 1.0 if len(board_ranks) != len(set(board_ranks)) else 0.0

            # ── Board flush texture: how suited the board is ─────────────────
            # 0=rainbow, 0.33=2-flush, 0.67=3-flush, 1.0=4-flush on board
            board_suit_counts = {}
            for c in board:
                board_suit_counts[c.suit] = board_suit_counts.get(c.suit, 0) + 1
            max_board_suit = max(board_suit_counts.values()) if board_suit_counts else 1
            board_flush_texture = float(np.clip((max_board_suit - 1) / 3.0, 0.0, 1.0))

            # ── Board straight texture: how connected the board is ───────────
            # Count consecutive rank pairs among unique board ranks, normalize to [0,1]
            board_ranks_unique = sorted(set(board_ranks))
            if 14 in board_ranks_unique:
                board_ranks_unique = sorted(set([1] + board_ranks_unique))
            consecutive_pairs = sum(
                1 for i in range(len(board_ranks_unique) - 1)
                if board_ranks_unique[i + 1] - board_ranks_unique[i] == 1
            )
            max_possible = max(len(board_ranks_unique) - 1, 1)
            board_straight_texture = float(consecutive_pairs / max_possible)

            return np.array(
                [hand_strength, made_hand_cat, has_flush_draw, has_straight_draw,
                 top_pair, board_is_paired, board_flush_texture, board_straight_texture],
                dtype=np.float32
            )
        except Exception:
            return np.zeros(8, dtype=np.float32)

    def _get_street_features(self) -> np.ndarray:
        """4-dim one-hot: [preflop, flop, turn, river]"""
        try:
            from rlcard.games.nolimitholdem.game import Stage
            game = self._get_game()
            if game:
                stage_map = {
                    Stage.PREFLOP: 0,
                    Stage.FLOP: 1,
                    Stage.TURN: 2,
                    Stage.RIVER: 3,
                }
                idx = stage_map.get(game.stage, 0)
                one_hot = np.zeros(4, dtype=np.float32)
                one_hot[idx] = 1.0
                return one_hot
        except Exception:
            pass
        return np.zeros(4, dtype=np.float32)

    def _get_game_state_features(self, current_agent: str) -> np.ndarray:
        """3 features: players_alive_norm, call_amount_norm, aggression_level"""
        try:
            import math
            from rlcard.games.nolimitholdem.player import PlayerStatus
            game = self._get_game()
            if game:
                agent_idx = self.agents.index(current_agent)
                player = game.players[agent_idx]

                alive = sum(
                    1 for p in game.players
                    if p.status not in (PlayerStatus.FOLDED,)
                )
                players_alive_norm = alive / self.num_players

                pot = sum(p.in_chips for p in game.players)
                max_raised = max(game.round.raised) if game.round.raised else 0
                call_amount = max(0, max_raised - player.in_chips)
                call_amount_norm = min(call_amount / max(pot, 1), 1.0)

                # Log-scale aggression: 0=no raise, ~0.37=single raise(6BB),
                # ~0.73=3-bet(18BB), 1.0=4-bet+(40BB)
                big_blind = game.big_blind if hasattr(game, 'big_blind') else 2
                ratio = max(max_raised, big_blind) / big_blind
                aggression_level = float(np.clip(
                    math.log(ratio) / math.log(20), 0.0, 1.0
                ))

                return np.array(
                    [players_alive_norm, call_amount_norm, aggression_level],
                    dtype=np.float32
                )
        except Exception:
            pass
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def _get_betting_history_features(self, current_agent: str) -> np.ndarray:
        """
        5 features capturing the betting action context:
          raise_level        – log-scale aggression capped at 4-bet (0=limp, 1=4bet+)
          my_pot_share       – fraction of pot I've personally put in [0,1]
          not_raise_fraction – fraction of players who haven't raised this round [0,1]
          facing_3bet_plus   – 1 if max_raised > 6×BB (3-bet or more)
          pot_was_raised     – 1 if any raise occurred this round (max_raised > BB)
        """
        try:
            import math
            game = self._get_game()
            if game:
                agent_idx = self.agents.index(current_agent)
                player = game.players[agent_idx]

                big_blind = game.big_blind if hasattr(game, 'big_blind') else 2
                max_raised = max(game.round.raised) if game.round.raised else 0
                pot = max(sum(p.in_chips for p in game.players), 1)

                ratio = max(max_raised, big_blind) / big_blind
                raise_level = float(np.clip(math.log(ratio) / math.log(40), 0.0, 1.0))

                my_pot_share = float(np.clip(player.in_chips / pot, 0.0, 1.0))

                not_raise_fraction = float(
                    np.clip(game.round.not_raise_num / max(self.num_players, 1), 0.0, 1.0)
                )

                facing_3bet_plus = 1.0 if max_raised > 6 * big_blind else 0.0
                pot_was_raised   = 1.0 if max_raised > big_blind else 0.0

                return np.array(
                    [raise_level, my_pot_share, not_raise_fraction,
                     facing_3bet_plus, pot_was_raised],
                    dtype=np.float32
                )
        except Exception:
            pass
        return np.zeros(5, dtype=np.float32)

    def _rank_to_value(self, rank: str) -> int:
        rank_map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        return rank_map.get(rank, 0)

    def get_observation(self) -> np.ndarray:
        if not self.current_agent:
            # Fallback: return zero vector of the right size
            return np.zeros(self.enhanced_obs_size, dtype=np.float32)

        features = []

        if self.add_position_features:
            features.append(self._get_position_features(self.current_agent))

        if self.add_pot_odds_features:
            features.append(self._get_pot_odds_features(self.current_agent))

        if self.add_hand_strength_features:
            features.append(self._get_hand_strength_features(self.current_agent))

        if self.add_stack_features:
            features.append(self._get_stack_features(self.current_agent))

        if self.add_equity_features:
            features.append(self._get_equity_features(self.current_agent))

        if self.add_street_features:
            features.append(self._get_street_features())

        if self.add_game_state_features:
            features.append(self._get_game_state_features(self.current_agent))

        if self.add_betting_history_features:
            features.append(self._get_betting_history_features(self.current_agent))

        if features:
            return np.concatenate(features).astype(np.float32)

        return np.zeros(self.enhanced_obs_size, dtype=np.float32)
