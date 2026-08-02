"""A sorted, parametric catalogue of postflop spots for inspecting a model.

`scripts/inspect_model_spots.py` hand-writes ~25 one-off spots. That is fine
for a smoke test but useless for answering "how does this model treat middle
pair OOP against a pot bet, 3-handed?" — you cannot vary one dimension at a
time.

Here a spot is the product of independent axes:

    hand category  x  street  x  position  x  players  x  what we face

so any slice can be held fixed while another varies. Boards are fixed per
(category, street) and chosen so the category is unambiguous — the turn and
river cards are deliberate blanks that do not complete a flush or straight,
otherwise "middle pair on the turn" would quietly become something else.

Ordering is stable and by hand strength (monster first, air last) so a list
of results reads top-down from strongest to weakest.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Hand categories, strongest first. `rank` drives the default sort order.
# ─────────────────────────────────────────────────────────────────────────────
CATEGORIES: List[Dict] = [
    {'key': 'quads',       'label': 'Quads',              'rank': 0,  'group': 'monster'},
    {'key': 'full_house',  'label': 'Full house',         'rank': 1,  'group': 'monster'},
    {'key': 'flush',       'label': 'Made flush',         'rank': 2,  'group': 'monster'},
    {'key': 'straight',    'label': 'Made straight',      'rank': 3,  'group': 'strong'},
    {'key': 'set',         'label': 'Set',                'rank': 4,  'group': 'strong'},
    {'key': 'trips',       'label': 'Trips',              'rank': 5,  'group': 'strong'},
    {'key': 'two_pair',    'label': 'Two pair',           'rank': 6,  'group': 'strong'},
    {'key': 'overpair',    'label': 'Overpair',           'rank': 7,  'group': 'made'},
    {'key': 'top_pair',    'label': 'Top pair',           'rank': 8,  'group': 'made'},
    {'key': 'mid_pair',    'label': 'Middle pair',        'rank': 9,  'group': 'marginal'},
    {'key': 'low_pair',    'label': 'Bottom pair',        'rank': 10, 'group': 'marginal'},
    {'key': 'flush_draw',  'label': 'Flush draw',         'rank': 11, 'group': 'draw'},
    {'key': 'oesd',        'label': 'Open-ended draw',    'rank': 12, 'group': 'draw'},
    {'key': 'gutshot',     'label': 'Gutshot',            'rank': 13, 'group': 'draw'},
    {'key': 'overcards',   'label': 'Two overcards',      'rank': 14, 'group': 'air'},
    {'key': 'air',         'label': 'Complete air',       'rank': 15, 'group': 'air'},
]
CATEGORY_BY_KEY = {c['key']: c for c in CATEGORIES}

# (hole, flop) per category. Turn/river cards are appended below.
_BASE: Dict[str, Tuple[List[str], List[str]]] = {
    'quads':      (['7h', '7c'], ['7s', '7d', 'Kh']),
    'full_house': (['7h', '7c'], ['7s', 'Kd', 'Kc']),
    'flush':      (['Ah', '5h'], ['Kh', '8h', '3h']),
    'straight':   (['9h', '8c'], ['7s', '6d', '5c']),
    'set':        (['7h', '7c'], ['7s', 'Kd', '3c']),
    'trips':      (['Ah', '7d'], ['7s', '7c', 'Kd']),
    'two_pair':   (['Ah', 'Kd'], ['As', 'Kc', '3d']),
    'overpair':   (['Ah', 'Ad'], ['Ks', '7d', '2c']),
    'top_pair':   (['Ah', 'Kd'], ['As', '7d', '2c']),
    'mid_pair':   (['Th', '9c'], ['Ks', 'Td', '5c']),
    'low_pair':   (['5h', '4c'], ['Ks', 'Td', '5c']),
    'flush_draw': (['Ah', '5h'], ['Kh', '8h', '3c']),
    'oesd':       (['9h', '8c'], ['7s', '6d', 'Kc']),
    'gutshot':    (['9h', '8c'], ['7s', '5d', 'Kc']),
    'overcards':  (['Ah', 'Kd'], ['9s', '7d', '2c']),
    'air':        (['7h', '2c'], ['As', 'Kd', '9s']),
}

# Blanks that keep each category intact on later streets: no third suit card
# for the flush-draw spots, no straight completion for the draw spots.
_RUNOUT: Dict[str, Tuple[str, str]] = {
    'quads':      ('2c', '3d'),
    'full_house': ('2c', '3d'),
    'flush':      ('2c', '3d'),
    'straight':   ('2c', '3d'),
    'set':        ('2h', '4d'),
    'trips':      ('2h', '4d'),
    'two_pair':   ('2h', '4d'),
    'overpair':   ('3h', '4d'),
    'top_pair':   ('3h', '4d'),
    'mid_pair':   ('2h', '3d'),
    'low_pair':   ('2h', '3d'),
    'flush_draw': ('2c', '3d'),   # never a heart — the draw must stay a draw
    'oesd':       ('2c', '3d'),   # never 5 or T — would complete the straight
    'gutshot':    ('2c', '3h'),   # never a 6
    'overcards':  ('3h', '4d'),
    'air':        ('4h', '3d'),
}

STREETS = ['flop', 'turn', 'river']
POSITIONS = [
    {'key': 'IP', 'label': 'In position (BTN)'},
    {'key': 'OOP', 'label': 'Out of position (BB)'},
]
#: What hero is facing. `bet_frac` is a fraction of the pot; 0 = checked to.
FACING = [
    {'key': 'checked_to', 'label': 'Checked to',      'bet_frac': 0.0},
    {'key': 'bet_third',  'label': 'vs 1/3 pot bet',  'bet_frac': 0.33},
    {'key': 'bet_half',   'label': 'vs 1/2 pot bet',  'bet_frac': 0.50},
    {'key': 'bet_pot',    'label': 'vs pot bet',      'bet_frac': 1.00},
]
PLAYER_COUNTS = [2, 3, 4, 5, 6]
DEFAULT_POT_BB = 7.0
DEFAULT_STACK_BB = 100.0


@dataclass
class Spot:
    id: str
    category: str
    category_label: str
    group: str
    rank: int
    street: str
    position: str
    num_players: int
    facing: str
    facing_label: str
    hole: List[str]
    board: List[str]
    pot_bb: float
    call_bb: float
    stack_bb: float

    @property
    def spr(self) -> float:
        return self.stack_bb / self.pot_bb if self.pot_bb else 0.0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['spr'] = round(self.spr, 1)
        return d


def board_for(category: str, street: str) -> List[str]:
    _, flop = _BASE[category]
    board = list(flop)
    turn, river = _RUNOUT[category]
    if street in ('turn', 'river'):
        board.append(turn)
    if street == 'river':
        board.append(river)
    return board


def build_spot(*, category: str, street: str, position: str,
               num_players: int, facing: str,
               pot_bb: float = DEFAULT_POT_BB,
               stack_bb: float = DEFAULT_STACK_BB) -> Spot:
    cat = CATEGORY_BY_KEY[category]
    face = next(f for f in FACING if f['key'] == facing)
    hole, _ = _BASE[category]
    # `pot_bb` is the pot BEFORE villain bets. GameState.pot_total must
    # include that bet — the convention scripts/inspect_model_spots.py uses
    # (pot=14, call=7 for a pot-sized bet into 7). Setting pot_total to the
    # pre-bet pot instead would make every bet read one size larger.
    call_bb = round(pot_bb * face['bet_frac'], 2)
    pot_total = round(pot_bb + call_bb, 2)
    return Spot(
        id=f'{category}|{street}|{position}|{num_players}p|{facing}',
        category=category, category_label=cat['label'], group=cat['group'],
        rank=cat['rank'], street=street, position=position,
        num_players=num_players, facing=facing, facing_label=face['label'],
        hole=list(hole), board=board_for(category, street),
        pot_bb=pot_total, call_bb=call_bb, stack_bb=stack_bb)


def generate(*, categories: Optional[List[str]] = None,
             streets: Optional[List[str]] = None,
             positions: Optional[List[str]] = None,
             player_counts: Optional[List[int]] = None,
             facings: Optional[List[str]] = None,
             pot_bb: float = DEFAULT_POT_BB,
             stack_bb: float = DEFAULT_STACK_BB) -> List[Spot]:
    """Cartesian product of the requested axes, sorted strongest-hand first."""
    categories = categories or [c['key'] for c in CATEGORIES]
    streets = streets or ['flop']
    positions = positions or ['IP', 'OOP']
    player_counts = player_counts or [6]
    facings = facings or [f['key'] for f in FACING]

    out: List[Spot] = []
    for cat in categories:
        if cat not in CATEGORY_BY_KEY:
            continue
        for street in streets:
            for pos in positions:
                for n in player_counts:
                    for fac in facings:
                        out.append(build_spot(
                            category=cat, street=street, position=pos,
                            num_players=n, facing=fac,
                            pot_bb=pot_bb, stack_bb=stack_bb))
    out.sort(key=lambda s: (s.rank, STREETS.index(s.street), s.position,
                            s.num_players,
                            [f['key'] for f in FACING].index(s.facing)))
    return out


def catalog() -> Dict:
    """Everything the UI needs to render its filter controls."""
    return {
        'categories': CATEGORIES,
        'streets': STREETS,
        'positions': POSITIONS,
        'facing': FACING,
        'player_counts': PLAYER_COUNTS,
        'defaults': {'pot_bb': DEFAULT_POT_BB, 'stack_bb': DEFAULT_STACK_BB},
    }
