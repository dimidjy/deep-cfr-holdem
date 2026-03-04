import numpy as np
from typing import List
from .poker_engine import PokerEngine


class EnhancedPokerEngine(PokerEngine):

    def __init__(self, num_players: int = 2,
                 add_position_features: bool = True,
                 add_pot_odds_features: bool = True,
                 add_hand_strength_features: bool = True,
                 add_stack_features: bool = True):
        super().__init__(num_players)
        self.add_position_features = add_position_features
        self.add_pot_odds_features = add_pot_odds_features
        self.add_hand_strength_features = add_hand_strength_features
        self.add_stack_features = add_stack_features

        self.base_obs_size = super().get_observation_space_size()
        self.enhanced_obs_size = self._calculate_enhanced_size()

    def _calculate_enhanced_size(self) -> int:
        size = self.base_obs_size

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

        return size

    def get_observation_space_size(self) -> int:
        return self.enhanced_obs_size

    def _get_position_features(self, current_agent: str) -> np.ndarray:
        try:
            game = self._get_game()
            if game:
                agent_idx = self.agents.index(current_agent)
                dealer_idx = game.dealer_id

                position_one_hot = np.zeros(self.num_players, dtype=np.float32)
                position_one_hot[agent_idx] = 1.0

                relative_position = (agent_idx - dealer_idx) % self.num_players
                normalized_position = relative_position / self.num_players

                is_button = 1.0 if agent_idx == dealer_idx else 0.0

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

    def _rank_to_value(self, rank: str) -> int:
        rank_map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        return rank_map.get(rank, 0)

    def get_observation(self) -> np.ndarray:
        base_obs = super().get_observation()

        if not self.current_agent:
            return base_obs

        extra_features = []

        if self.add_position_features:
            extra_features.append(self._get_position_features(self.current_agent))

        if self.add_pot_odds_features:
            extra_features.append(self._get_pot_odds_features(self.current_agent))

        if self.add_hand_strength_features:
            extra_features.append(self._get_hand_strength_features(self.current_agent))

        if self.add_stack_features:
            extra_features.append(self._get_stack_features(self.current_agent))

        if extra_features:
            return np.concatenate([base_obs] + extra_features).astype(np.float32)

        return base_obs
