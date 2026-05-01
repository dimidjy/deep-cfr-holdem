"""Shared spot-key derivation for preflop lookup table.

Single source of truth used by BOTH the offline build script
(`scripts/build_preflop_lookup.py`) and the runtime lookup
(`src/preflop_lookup.py`). Build/runtime divergence on key derivation
is the #1 production bug source for systems like this — keep ALL key
construction here.

Spot key fields:
    hole_class       169-class string ('AA', 'AKS', 'AKO', ...)
    position         seat label ('BTN', 'SB', 'BB', 'LJ', 'HJ', 'CO')
    action_history   summary of preflop actions before us:
                       'open'         — no voluntary actions yet
                       'limped_pot'   — 1+ limps, 0 raises
                       'vs_raise'     — exactly 1 raise
                       'vs_3bet'      — exactly 2 raises
                       'vs_4bet+'     — 3+ raises
    stack_bucket     '<40' / '40-70' / '70-150' / '150+' BB
    num_to_act       players still to act behind hero, clamped to {0..4}
"""
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from src.preflop_ranges import _cards_to_combo, _POS_TO_RANGE_KEY_BY_PLAYERS


# Cohort and lookup constants
BB_USD = 0.05
MIN_HANDS = 20_000
MIN_BB100 = 10.0
MIN_SAMPLES_PER_SPOT = 5

# Action-history enum values
HIST_OPEN = 'open'
HIST_LIMPED_POT = 'limped_pot'
HIST_VS_RAISE = 'vs_raise'
HIST_VS_3BET = 'vs_3bet'
HIST_VS_4BET = 'vs_4bet+'
ACTION_HISTORY_VALUES = (
    HIST_OPEN, HIST_LIMPED_POT, HIST_VS_RAISE, HIST_VS_3BET, HIST_VS_4BET,
)

# Stack bucket boundaries (in BB)
STACK_BUCKETS = (
    (40.0, '<40'),
    (70.0, '40-70'),
    (150.0, '70-150'),
    (float('inf'), '150+'),
)


@dataclass(frozen=True)
class SpotKey:
    hole_class: str
    position: str
    action_history: str
    stack_bucket: str
    num_to_act: int

    def to_tuple(self) -> Tuple[str, str, str, str, int]:
        return (self.hole_class, self.position, self.action_history,
                self.stack_bucket, self.num_to_act)


def cards_to_class(card1: str, card2: str) -> str:
    """169-class hole-card label. Reuses preflop_ranges._cards_to_combo
    so the lookup keys exactly match the existing range JSON keys."""
    return _cards_to_combo(card1, card2)


def position_label(rel_pos: int, num_players: int) -> str:
    """Map (rel_pos, num_players) → position string. Mirrors the existing
    range JSON convention (BTN/SB/BB/LJ/HJ/CO for 6-max).

    rel_pos: 0=BTN, 1=SB, 2=BB, 3+=earlier positions counted backward
             from button.
    """
    pos_map = _POS_TO_RANGE_KEY_BY_PLAYERS.get(
        num_players, _POS_TO_RANGE_KEY_BY_PLAYERS[6])
    return pos_map.get(rel_pos, f'P{rel_pos}')


def stack_bucket(effective_bb: float) -> str:
    for upper, label in STACK_BUCKETS:
        if effective_bb < upper:
            return label
    return STACK_BUCKETS[-1][1]


def num_to_act_bucket(n: int) -> int:
    """Clamp to {0, 1, 2, 3, 4} so very-full tables collapse together."""
    if n < 0:
        return 0
    return min(n, 4)


def derive_action_history(prior_actions: Iterable[Tuple[str, float]]) -> str:
    """Classify the spot from the actions taken BEFORE hero this preflop street.

    Args:
        prior_actions: ordered iterable of (action_type, size_bb) tuples for
            voluntary preflop actions only — DO NOT include posted blinds.
            action_type ∈ {'fold', 'call', 'raise'}. size_bb is the total
            BB amount of the action (call amount or raise-to amount); used
            to disambiguate limp vs cold-call.

    Returns one of HIST_* constants.
    """
    raises = 0
    limps = 0
    for action_type, size_bb in prior_actions:
        if action_type == 'raise':
            raises += 1
        elif action_type == 'call' and raises == 0:
            # Any call before a raise is a limp (including SB completing
            # for 0.5 BB). Calls after a raise are cold-calls and don't
            # count toward limped_pot — they leave us in vs_raise.
            limps += 1
        # folds don't move the spot category

    if raises == 0:
        return HIST_OPEN if limps == 0 else HIST_LIMPED_POT
    if raises == 1:
        return HIST_VS_RAISE
    if raises == 2:
        return HIST_VS_3BET
    return HIST_VS_4BET


def build_spot_key(hole_cards: List[str], rel_pos: int, num_players: int,
                   prior_actions: Iterable[Tuple[str, float]],
                   effective_bb: float, num_to_act: int) -> SpotKey:
    """One-stop key constructor for both build-time and runtime.

    hole_cards: ['As', 'Kh']
    rel_pos: 0=BTN, 1=SB, 2=BB, 3+ earlier positions
    num_players: active player count at the table (2..6)
    prior_actions: see derive_action_history()
    effective_bb: smaller of (hero stack, deepest opponent stack still in)
    num_to_act: players left to act after hero (will be clamped)
    """
    return SpotKey(
        hole_class=cards_to_class(hole_cards[0], hole_cards[1]),
        position=position_label(rel_pos, num_players),
        action_history=derive_action_history(prior_actions),
        stack_bucket=stack_bucket(effective_bb),
        num_to_act=num_to_act_bucket(num_to_act),
    )
