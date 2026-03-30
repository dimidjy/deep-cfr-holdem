"""Preflop range override for training.

Loads GTO-style preflop ranges from configs/preflop_ranges.json and provides
a function to determine the correct preflop action given hole cards, position,
and pot size.

Used during CFR training to lock preflop play to known-good ranges,
so the model only learns postflop strategy.
"""

import json
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np


# Position mapping: relative_pos → range key, per table size
# rel_pos: 0=BTN, 1=SB, 2=BB, 3+=EP positions
_POS_TO_RANGE_KEY_BY_PLAYERS = {
    6: {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'LJ', 4: 'HJ', 5: 'CO'},
    5: {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'HJ', 4: 'CO'},
    4: {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'CO'},
    3: {0: 'BTN', 1: 'SB', 2: 'BB'},
    2: {0: 'BTN', 1: 'BB'},
}
# Fallback for legacy code that doesn't pass num_players
_POS_TO_RANGE_KEY = _POS_TO_RANGE_KEY_BY_PLAYERS[6]

# Action constants (matching engine action space)
FOLD = 0
CALL = 1
RAISE_HALF = 2
RAISE_POT = 3
ALL_IN = 4
RAISE_MIN = 5  # min raise — min raise (2BB)

_RANK_ORDER = '23456789TJQKA'


def _cards_to_combo(card1: str, card2: str) -> str:
    """Convert two card strings like 'As', 'Kh' to range notation like 'AKO' or 'AKS'.

    Returns the canonical combo string used in the ranges JSON.
    For pairs: 'AA', 'KK', etc.
    For non-pairs: 'AKS' (suited) or 'AKO' (offsuit), always high card first.
    """
    r1 = card1[0].upper()
    r2 = card2[0].upper()
    s1 = card1[1].lower()
    s2 = card2[1].lower()

    # Sort by rank (high first)
    idx1 = _RANK_ORDER.index(r1)
    idx2 = _RANK_ORDER.index(r2)
    if idx1 < idx2:
        r1, r2 = r2, r1
        s1, s2 = s2, s1

    if r1 == r2:
        return f'{r1}{r2}'  # pair: 'AA', 'KK'
    elif s1 == s2:
        return f'{r1}{r2}S'  # suited: 'AKS'
    else:
        return f'{r1}{r2}O'  # offsuit: 'AKO'


def _is_in_range(combo: str, range_dict: dict) -> bool:
    """Check if a combo is in a range dictionary.

    The range JSON has keys like 'AA', 'AKS', 'AKO' with frequency values.
    Some keys have ':' prefix in values — we ignore the value and just check presence.
    """
    return combo in range_dict


class PreflopRanges:
    """Preflop range chart for N-max poker (2-6 players).

    Determines preflop action (fold/call/raise/all-in) based on:
    - Hole cards
    - Position (relative to button)
    - Pot size in BB (determines scenario: RFI, facing raise, facing 3bet, etc.)
    """

    def __init__(self, ranges_path: Optional[str] = None, num_players: int = 6):
        self.num_players = num_players
        self.pos_map = _POS_TO_RANGE_KEY_BY_PLAYERS.get(
            num_players, _POS_TO_RANGE_KEY_BY_PLAYERS[6])

        if ranges_path is None:
            configs_dir = Path(__file__).parent.parent / 'configs'
            # Try player-specific config first, fall back to generic
            player_specific = configs_dir / f'preflop_ranges_{num_players}p.json'
            if player_specific.exists():
                ranges_path = str(player_specific)
            else:
                ranges_path = str(configs_dir / 'preflop_ranges.json')

        with open(ranges_path) as f:
            self.ranges = json.load(f)

    def get_action(self, hole_cards: List[str], rel_pos: int,
                   pot_bb: float, legal_actions: List[int]) -> Optional[int]:
        """Determine preflop action from range charts.

        Args:
            hole_cards: ['As', 'Kh'] — two card strings
            rel_pos: 0=BTN, 1=SB, 2=BB, 3=UTG, 4=MP, 5=CO
            pot_bb: current pot size in big blinds
            legal_actions: list of legal action indices [0,1,2,3,4]

        Returns:
            Action index (0-4), or None if ranges don't cover this situation.
        """
        if len(hole_cards) < 2:
            return None

        combo = _cards_to_combo(hole_cards[0], hole_cards[1])
        pos_key = self.pos_map.get(rel_pos, 'BTN')

        # Determine scenario based on pot size (in BB)
        # pot_bb < 4: no raise yet → check RFI range
        # pot_bb < 18: facing single raise (may include callers) → 3bet / call
        # pot_bb < 50: facing 3bet → check 4bet / call range
        # pot_bb >= 50: facing 4bet+ → check all-in range

        if pot_bb < 4:
            return self._rfi_decision(combo, pos_key, legal_actions)
        elif pot_bb < 18:
            return self._facing_raise_decision(combo, pos_key, legal_actions)
        elif pot_bb < 50:
            return self._facing_3bet_decision(combo, pos_key, legal_actions)
        else:
            return self._facing_4bet_decision(combo, pos_key, legal_actions)

    def get_strategy(self, hole_cards: List[str], rel_pos: int,
                     pot_bb: float, legal_actions: List[int],
                     action_size: int = 6) -> np.ndarray:
        """Get a strategy vector (probability distribution over actions).

        Returns a near-deterministic strategy based on the range chart.
        Uses 95% on the chosen action, 5% spread across other legal actions
        to maintain some exploration.
        """
        strategy = np.zeros(action_size, dtype=np.float32)
        action = self.get_action(hole_cards, rel_pos, pot_bb, legal_actions)

        if action is None or action not in legal_actions:
            # Fallback: uniform over legal actions
            for a in legal_actions:
                strategy[a] = 1.0 / len(legal_actions)
            return strategy

        # Deterministic: 100% on chosen action
        strategy[action] = 1.0

        # Normalize
        s = strategy.sum()
        if s > 0:
            strategy /= s
        return strategy

    def _best_raise(self, legal_actions: List[int]) -> int:
        """Pick best available raise action (never all-in unless in allin range)."""
        for a in [RAISE_POT, RAISE_HALF]:
            if a in legal_actions:
                return a
        if CALL in legal_actions:
            return CALL
        return FOLD

    def _rfi_decision(self, combo: str, pos_key: str,
                      legal_actions: List[int]) -> int:
        """RFI (raise first in) — no raise before us.

        Sizing logic:
        - 3bet-range hands (premium): 80% pot raise, 20% min raise (2BB)
        - Other RFI hands: 80% min raise (2BB), 20% pot raise
        Min raise uses RAISE_MIN (action 5) = min raise.
        """
        rfi_range = self.ranges.get('rfi', {}).get(pos_key, {})
        if _is_in_range(combo, rfi_range):
            # Check if hand is also in 3bet range (premium)
            key_3b = f'{pos_key}_3b'
            range_3b = self.ranges.get('3b', {}).get(key_3b, {})
            is_premium = _is_in_range(combo, range_3b)

            rng = np.random.random()
            if is_premium:
                # 80% pot, 20% min raise
                action = RAISE_POT if rng < 0.8 else RAISE_MIN
            else:
                # 80% min raise, 20% pot
                action = RAISE_MIN if rng < 0.8 else RAISE_POT

            # Fall back if chosen raise not in legal actions
            if action not in legal_actions:
                for fallback in [RAISE_MIN, RAISE_HALF, RAISE_POT, CALL]:
                    if fallback != action and fallback in legal_actions:
                        return fallback
            return action

        # SB defense: complete instead of fold
        if pos_key == 'SB':
            sb_def = self.ranges.get('SB_DEF', {}).get('SB_DEF_call', {})
            if _is_in_range(combo, sb_def):
                if CALL in legal_actions:
                    return CALL

        return FOLD if FOLD in legal_actions else CALL

    def _facing_raise_decision(self, combo: str, pos_key: str,
                               legal_actions: List[int]) -> int:
        """Facing a single raise — check 3bet or call."""
        # 3-bet range
        key_3b = f'{pos_key}_3b'
        range_3b = self.ranges.get('3b', {}).get(key_3b, {})
        if _is_in_range(combo, range_3b):
            return self._best_raise(legal_actions)

        # Call range
        call_range = self.ranges.get('rfi_call', {}).get(pos_key, {})
        if _is_in_range(combo, call_range):
            if CALL in legal_actions:
                return CALL

        return FOLD if FOLD in legal_actions else CALL

    def _facing_3bet_decision(self, combo: str, pos_key: str,
                              legal_actions: List[int]) -> int:
        """Facing a 3-bet — check 4bet or call."""
        # 4-bet range
        key_4b = f'{pos_key}_4b'
        range_4b = self.ranges.get('4b', {}).get(key_4b, {})
        if _is_in_range(combo, range_4b):
            return self._best_raise(legal_actions)

        # Call range
        key_call = f'{pos_key}_call'
        call_range = self.ranges.get('3b_call', {}).get(key_call, {})
        if _is_in_range(combo, call_range):
            if CALL in legal_actions:
                return CALL

        return FOLD if FOLD in legal_actions else CALL

    def _facing_4bet_decision(self, combo: str, pos_key: str,
                              legal_actions: List[int]) -> int:
        """Facing 4bet+ — all-in or fold."""
        allin_range = self.ranges.get('Allin', {}).get('allin_base', {})
        if _is_in_range(combo, allin_range):
            if ALL_IN in legal_actions:
                return ALL_IN
            return self._best_raise(legal_actions)

        return FOLD if FOLD in legal_actions else CALL
