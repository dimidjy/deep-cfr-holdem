from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_RAISE_HALF = 2
ACTION_RAISE_POT = 3
ACTION_ALL_IN = 4
ACTION_RAISE_MIN = 5
# 6 reserved for fast_fold (agent-only, not a model output)
ACTION_RAISE_THIRD = 7  # 1/3-pot bet

ACTION_NAMES = {
    ACTION_FOLD: 'Fold',
    ACTION_CALL: 'Check/Call',
    ACTION_RAISE_HALF: 'Raise Half Pot',
    ACTION_RAISE_POT: 'Raise Pot',
    ACTION_ALL_IN: 'All-In',
    ACTION_RAISE_MIN: 'Raise Min',
    ACTION_RAISE_THIRD: 'Raise 1/3 Pot',
}

NUM_ACTIONS = 6   # model output width — new actions >=6 are agent-only overrides


@dataclass
class CardInfo:
    rank: str   # '2','3',...,'9','T','J','Q','K','A'
    suit: str   # 'h','d','c','s'

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    @classmethod
    def from_str(cls, s: str) -> 'CardInfo':
        return cls(rank=s[0].upper(), suit=s[1].lower())

    def to_dict(self) -> dict:
        return {'rank': self.rank, 'suit': self.suit}

    @classmethod
    def from_dict(cls, d: dict) -> 'CardInfo':
        return cls(rank=d['rank'], suit=d['suit'])


@dataclass
class PlayerInfo:
    seat: int
    name: str = ''
    stack: float = 0.0             # in big blinds
    stack_dollars: float = 0.0
    is_active: bool = True         # still in the hand
    is_all_in: bool = False
    bet_current_round: float = 0.0 # current street bet in BBs
    total_invested: float = 0.0    # total chips in pot in BBs
    hole_cards: Optional[List[CardInfo]] = None
    has_button: bool = False
    is_sitting_out: bool = False

    def to_dict(self) -> dict:
        d = {
            'seat': self.seat,
            'name': self.name,
            'stack': self.stack,
            'stack_dollars': self.stack_dollars,
            'is_active': self.is_active,
            'is_all_in': self.is_all_in,
            'bet_current_round': self.bet_current_round,
            'total_invested': self.total_invested,
            'has_button': self.has_button,
            'is_sitting_out': self.is_sitting_out,
        }
        if self.hole_cards:
            d['hole_cards'] = [c.to_dict() for c in self.hole_cards]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'PlayerInfo':
        cards = None
        if 'hole_cards' in d and d['hole_cards']:
            cards = [CardInfo.from_dict(c) for c in d['hole_cards']]
        return cls(
            seat=d['seat'], name=d.get('name', ''),
            stack=d.get('stack', 0.0), stack_dollars=d.get('stack_dollars', 0.0),
            is_active=d.get('is_active', True), is_all_in=d.get('is_all_in', False),
            bet_current_round=d.get('bet_current_round', 0.0),
            total_invested=d.get('total_invested', 0.0),
            hole_cards=cards, has_button=d.get('has_button', False),
            is_sitting_out=d.get('is_sitting_out', False),
        )


@dataclass
class GameState:
    players: List[PlayerInfo] = field(default_factory=list)
    my_seat: int = 0
    community_cards: List[CardInfo] = field(default_factory=list)
    pot_total: float = 0.0         # in BBs
    street: str = 'preflop'        # 'preflop', 'flop', 'turn', 'river'
    button_seat: int = 0
    small_blind: float = 0.5       # in BBs (typically 0.5)
    big_blind: float = 1.0         # in BBs (typically 1.0)
    small_blind_dollars: float = 0.02
    big_blind_dollars: float = 0.05
    legal_actions: List[int] = field(default_factory=lambda: [0, 1])
    num_active_players: int = 2
    call_amount: float = 0.0       # in BBs
    min_raise: float = 0.0         # in BBs
    max_raise: float = 0.0         # in BBs
    timestamp: str = ''
    raw_source_path: str = ''

    @property
    def my_player(self) -> Optional[PlayerInfo]:
        for p in self.players:
            if p.seat == self.my_seat:
                return p
        return None

    @property
    def my_hole_cards(self) -> List[CardInfo]:
        p = self.my_player
        if p and p.hole_cards:
            return p.hole_cards
        return []

    @property
    def active_players(self) -> List[PlayerInfo]:
        return [p for p in self.players if p.is_active and not p.is_sitting_out]

    def to_dict(self) -> dict:
        return {
            'players': [p.to_dict() for p in self.players],
            'my_seat': self.my_seat,
            'community_cards': [c.to_dict() for c in self.community_cards],
            'pot_total': self.pot_total,
            'street': self.street,
            'button_seat': self.button_seat,
            'small_blind': self.small_blind,
            'big_blind': self.big_blind,
            'small_blind_dollars': self.small_blind_dollars,
            'big_blind_dollars': self.big_blind_dollars,
            'legal_actions': self.legal_actions,
            'num_active_players': self.num_active_players,
            'call_amount': self.call_amount,
            'min_raise': self.min_raise,
            'max_raise': self.max_raise,
            'timestamp': self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'GameState':
        players = [PlayerInfo.from_dict(p) for p in d.get('players', [])]
        cards = [CardInfo.from_dict(c) for c in d.get('community_cards', [])]
        return cls(
            players=players, my_seat=d.get('my_seat', 0),
            community_cards=cards, pot_total=d.get('pot_total', 0.0),
            street=d.get('street', 'preflop'), button_seat=d.get('button_seat', 0),
            small_blind=d.get('small_blind', 0.5), big_blind=d.get('big_blind', 1.0),
            small_blind_dollars=d.get('small_blind_dollars', 0.02),
            big_blind_dollars=d.get('big_blind_dollars', 0.05),
            legal_actions=d.get('legal_actions', [0, 1]),
            num_active_players=d.get('num_active_players', 2),
            call_amount=d.get('call_amount', 0.0),
            min_raise=d.get('min_raise', 0.0), max_raise=d.get('max_raise', 0.0),
            timestamp=d.get('timestamp', ''),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> 'GameState':
        return cls.from_dict(json.loads(s))
