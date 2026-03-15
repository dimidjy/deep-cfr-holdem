"""Data models for parsed PokerStars hand histories."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# $0.05 (1 BB) = 2 chips in the engine
DOLLARS_TO_CHIPS = 2.0 / 0.05  # = 40.0


def dollars_to_chips(amount: float) -> int:
    return int(round(amount * DOLLARS_TO_CHIPS))


@dataclass
class ParsedPlayer:
    name: str
    seat: int
    stack_dollars: float
    stack_chips: int = 0
    hole_cards: Optional[List[str]] = None
    is_sitting_out: bool = False

    def __post_init__(self):
        if self.stack_chips == 0:
            self.stack_chips = dollars_to_chips(self.stack_dollars)


@dataclass
class ParsedAction:
    player_name: str
    action_type: str      # fold/check/call/bet/raise/all_in/blind_sb/blind_bb
    amount_dollars: float = 0.0
    amount_chips: int = 0
    street: str = 'preflop'
    is_blind: bool = False
    is_all_in: bool = False

    def __post_init__(self):
        if self.amount_chips == 0 and self.amount_dollars > 0:
            self.amount_chips = dollars_to_chips(self.amount_dollars)


@dataclass
class HandResult:
    total_pot_dollars: float = 0.0
    rake_dollars: float = 0.0
    winners: List[Dict] = field(default_factory=list)  # [{name, amount_dollars}]


@dataclass
class ParsedHand:
    hand_id: str = ''
    table_name: str = ''
    timestamp: str = ''
    max_seats: int = 6
    button_seat: int = 1
    small_blind_dollars: float = 0.02
    big_blind_dollars: float = 0.05
    players: List[ParsedPlayer] = field(default_factory=list)
    actions: List[ParsedAction] = field(default_factory=list)
    flop: Optional[List[str]] = None    # e.g. ['9h', '9d', 'Qs']
    turn: Optional[str] = None          # e.g. 'Kc'
    river: Optional[str] = None         # e.g. '3d'
    shown_hands: Dict[str, List[str]] = field(default_factory=dict)
    result: HandResult = field(default_factory=HandResult)

    @property
    def community_cards_at_street(self) -> Dict[str, List[str]]:
        """Community cards available at each street."""
        cards = {'preflop': []}
        if self.flop:
            cards['flop'] = list(self.flop)
            cards['turn'] = list(self.flop)
            if self.turn:
                cards['turn'].append(self.turn)
                cards['river'] = cards['turn'][:]
                if self.river:
                    cards['river'].append(self.river)
        return cards

    @property
    def active_players(self) -> List[ParsedPlayer]:
        return [p for p in self.players if not p.is_sitting_out]

    @property
    def num_active_players(self) -> int:
        return len(self.active_players)

    @property
    def has_showdown(self) -> bool:
        return len(self.shown_hands) > 0


@dataclass
class DecisionPoint:
    player_name: str
    observation: np.ndarray          # 38-dim feature vector
    legal_actions: List[int]         # subset of [0,1,2,3,4]
    actual_action: int               # mapped to 0-4
    street: str                      # preflop/flop/turn/river
    hole_cards: Optional[List[str]]  # if known from showdown
    community_cards: List[str]       # board cards at this point
    position_label: str = ''         # BTN/SB/BB/UTG/MP/CO
    player_idx: int = 0              # index in active player list
    hand_id: str = ''                # hand ID for grouping
    action_index: int = 0            # index of this action for the player in the hand
    pot_dollars: float = 0.0         # pot size in dollars at decision time
    source_file: str = ''            # source HH file path (for file-based splits)
    player_net_dollars: float = 0.0  # player's net profit/loss for this hand
    is_last_action: bool = False     # True if this is the player's final action in the hand
