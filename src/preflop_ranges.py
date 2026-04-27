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
#
# Short-handed adjustment: in 4-max and 5-max, the "first to act" position is
# structurally UTG-equivalent (more players behind than 6-max CO). Map those
# positions to the TIGHTEST 6-max EP range (LJ) so we don't over-open. Same
# logic for 5-max: first-to-act → LJ (tight), second-to-act → HJ (medium).
_POS_TO_RANGE_KEY_BY_PLAYERS = {
    6: {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'LJ', 4: 'HJ', 5: 'CO'},
    5: {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'LJ', 4: 'HJ'},
    4: {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'LJ'},
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
                   pot_bb: float, legal_actions: List[int],
                   num_limpers: int = 0, effective_stack: float = 100.0,
                   num_callers: int = 0) -> Optional[int]:
        """Determine preflop action from range charts.

        Args:
            hole_cards: ['As', 'Kh'] — two card strings
            rel_pos: 0=BTN, 1=SB, 2=BB, 3=UTG, 4=MP, 5=CO
            pot_bb: current pot size in big blinds
            legal_actions: list of legal action indices [0,1,2,3,4]
            num_limpers: number of players who limped before us
            effective_stack: (unused) kept for API compatibility; preflop
                logic now relies entirely on range charts regardless of
                stack depth
            num_callers: number of cold-callers of the open (excludes opener)

        Returns:
            Action index (0-4), or None if ranges don't cover this situation.
        """
        if len(hole_cards) < 2:
            return None

        combo = _cards_to_combo(hole_cards[0], hole_cards[1])
        pos_key = self.pos_map.get(rel_pos, 'BTN')

        # Preflop decisions are fully driven by range charts (hand + position +
        # scenario). Stack depth is intentionally ignored — the ranges are the
        # single source of truth. Previously a "short-stack override" (<25BB →
        # push/fold) fired here, but it caused edge-case bugs when an opponent
        # was already all-in (effective_stack collapsed to that opponent's
        # committed amount, triggering shove-mode for non-shove spots).

        # Determine scenario based on pot size (in BB)
        if pot_bb < 4:
            # Check for limpers: if players limped, adjust strategy
            if num_limpers > 0:
                return self._facing_limpers_decision(combo, pos_key,
                                                      legal_actions, num_limpers)
            return self._rfi_decision(combo, pos_key, legal_actions)
        elif pot_bb < 18:
            # Squeeze spot: open + 1+ cold-callers → premium-only 3-bet
            if num_callers > 0:
                return self._facing_squeeze_decision(combo, pos_key,
                                                      legal_actions, num_callers)
            return self._facing_raise_decision(combo, pos_key, legal_actions)
        elif pot_bb < 50:
            return self._facing_3bet_decision(combo, pos_key, legal_actions)
        else:
            return self._facing_4bet_decision(combo, pos_key, legal_actions)

    def get_strategy(self, hole_cards: List[str], rel_pos: int,
                     pot_bb: float, legal_actions: List[int],
                     action_size: int = 6,
                     num_limpers: int = 0,
                     effective_stack: float = 100.0,
                     num_callers: int = 0) -> np.ndarray:
        """Get a strategy vector (probability distribution over actions)."""
        strategy = np.zeros(action_size, dtype=np.float32)
        action = self.get_action(hole_cards, rel_pos, pot_bb, legal_actions,
                                 num_limpers=num_limpers,
                                 effective_stack=effective_stack,
                                 num_callers=num_callers)

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
        """Pick best available raise action (biggest sizing)."""
        for a in [RAISE_POT, RAISE_HALF, RAISE_MIN, ALL_IN]:
            if a in legal_actions:
                return a
        if CALL in legal_actions:
            return CALL
        return FOLD

    def _best_3bet_raise(self, legal_actions: List[int]) -> int:
        """Pick sizing for a 3-bet vs a single raiser.

        Proper 3-bet is ~3–3.5x the open, which maps closer to RAISE_HALF
        than RAISE_POT at typical stack depths. Using pot-raise here bloats
        the pot and over-commits on non-premium hands.
        """
        for a in [RAISE_HALF, RAISE_POT, RAISE_MIN, ALL_IN]:
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

            # Previously: SB non-premium had 40% complete (limp) logic here.
            # Removed — at NL5 the exploit value of open-raising as SB vs
            # passive BB outweighs the OOP pot-control benefit of completing.
            # Net effect: SB opens its full RFI range → PFR up, VPIP gap down.

            rng = np.random.random()
            if is_premium:
                # 80% pot, 20% min raise
                action = RAISE_POT if rng < 0.8 else RAISE_MIN
            else:
                # 80% min raise, 20% pot
                action = RAISE_MIN if rng < 0.8 else RAISE_POT

            # Fall back if chosen raise not in legal actions
            if action not in legal_actions:
                # Premium hands can go all-in if no other raise available
                fallbacks = [RAISE_MIN, RAISE_HALF, RAISE_POT, ALL_IN, CALL] if is_premium \
                       else [RAISE_MIN, RAISE_HALF, RAISE_POT, CALL]
                for fallback in fallbacks:
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
        """Facing a single raise, heads-up to the opener — 3bet, call, or fold."""
        # 3-bet range
        key_3b = f'{pos_key}_3b'
        range_3b = self.ranges.get('3b', {}).get(key_3b, {})
        if _is_in_range(combo, range_3b):
            return self._best_3bet_raise(legal_actions)

        # Call range
        call_range = self.ranges.get('rfi_call', {}).get(pos_key, {})
        if _is_in_range(combo, call_range):
            if CALL in legal_actions:
                return CALL

        return FOLD if FOLD in legal_actions else CALL

    def _facing_squeeze_decision(self, combo: str, pos_key: str,
                                  legal_actions: List[int],
                                  num_callers: int) -> int:
        """Facing open + at least one cold-caller — multiway squeeze spot.

        Multiway facing two+ ranges, so the 3-bet range is much tighter
        than HU (uses 4-bet range = QQ+/AKs/AKo for most positions).
        Non-premium hands that would normally 3-bet heads-up should call
        (if they flat well multiway) or fold.

        Squeeze sizing is pot-raise to charge the cold-caller's equity.
        """
        # Premium squeeze range = position's 4-bet range (QQ+, AKs, AKo)
        key_4b = f'{pos_key}_4b'
        range_squeeze = self.ranges.get('4b', {}).get(key_4b, {})
        if _is_in_range(combo, range_squeeze):
            # Squeeze big — pot raise (or best available)
            return self._best_raise(legal_actions)

        # Multiway call range: hands that realize equity well multiway
        # (pairs, suited broadways, suited connectors). Use rfi_call as
        # the flat range — it's already a "good to call a raise" range.
        call_range = self.ranges.get('rfi_call', {}).get(pos_key, {})
        if _is_in_range(combo, call_range):
            # Tighten further with more callers: 2+ callers = only pairs
            # and suited broadways realize equity well enough.
            if num_callers >= 2:
                is_pair = len(combo) == 2
                is_suited = combo.endswith('S')
                if not (is_pair or is_suited):
                    return FOLD if FOLD in legal_actions else CALL
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

    def _facing_limpers_decision(self, combo: str, pos_key: str,
                                 legal_actions: List[int],
                                 num_limpers: int) -> int:
        """Facing limpers (callers before us, no raise).

        Strategy vs limpers:
        - Premium hands: raise bigger (pot-size to punish limpers)
        - Good hands still in RFI range: raise (iso-raise)
        - Speculative hands (small pairs, suited connectors): limp behind
        - Trash: fold
        """
        rfi_range = self.ranges.get('rfi', {}).get(pos_key, {})
        key_3b = f'{pos_key}_3b'
        range_3b = self.ranges.get('3b', {}).get(key_3b, {})
        is_premium = _is_in_range(combo, range_3b)

        if is_premium:
            # Premium vs limpers: always pot-raise (punish the limpers)
            if RAISE_POT in legal_actions:
                return RAISE_POT
            return self._best_raise(legal_actions)

        # Speculative hands: limp behind instead of raising (better multiway)
        r1 = _RANK_ORDER.index(combo[0]) if combo[0] in _RANK_ORDER else 0
        r2_char = combo[1] if len(combo) >= 2 and combo[1] in _RANK_ORDER else combo[0]
        r2 = _RANK_ORDER.index(r2_char) if r2_char in _RANK_ORDER else 0
        is_pair = len(combo) == 2 and combo[0] == combo[1]
        is_suited = combo.endswith('S')
        gap = abs(r1 - r2)

        # Limp with: small/medium pairs, suited connectors/gappers
        if is_pair and r1 <= _RANK_ORDER.index('7'):
            if CALL in legal_actions:
                return CALL
        if is_suited and gap <= 2 and min(r1, r2) >= _RANK_ORDER.index('3'):
            if CALL in legal_actions:
                return CALL

        # SB OOP vs limpers: limp-behind any non-broadway non-premium RFI hand.
        # Iso-raising marginal hands (T7s, K5s, Q8s) OOP bloats the pot and
        # forces tough decisions postflop. Iso only with broadway combos
        # (both cards ≥T) and medium+ pairs (88+).
        if pos_key == 'SB' and num_limpers >= 1:
            T_idx = _RANK_ORDER.index('T')
            iso_ok = (
                (is_pair and r1 >= _RANK_ORDER.index('8'))
                or (not is_pair and r1 >= T_idx and r2 >= T_idx)
            )
            if not iso_ok:
                if _is_in_range(combo, rfi_range) and CALL in legal_actions:
                    return CALL
                return FOLD if FOLD in legal_actions else CALL

        if _is_in_range(combo, rfi_range):
            # Standard RFI hand (not speculative): iso-raise
            rng = np.random.random()
            if num_limpers >= 2:
                action = RAISE_POT if rng < 0.7 else RAISE_HALF
            else:
                action = RAISE_POT if rng < 0.4 else RAISE_MIN
            if action not in legal_actions:
                for fb in [RAISE_POT, RAISE_HALF, RAISE_MIN, CALL]:
                    if fb in legal_actions:
                        return fb
            return action

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
