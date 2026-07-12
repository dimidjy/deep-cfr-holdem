"""Map between TexasSolver action strings and the project's action indices.

Project action indices (src.agent.game_state):
    0 = Fold, 1 = Call/Check, 2 = Raise½ (small bet), 3 = RaisePot (big bet),
    4 = AllIn.  (5 = RaiseMin and 7 = RaiseThird are agent-only; GTO never
    produces them, so the GTO strategy vector uses indices 0-4 inside a
    length-6 array to line up with the rest of the codebase.)

TexasSolver actions at a node look like: 'FOLD', 'CHECK', 'CALL',
'BET 75.000000', 'RAISE 200.000000', 'ALLIN'. The numeric value is in the
solver's chip units.

Bet sizes are CHOSEN by us (see `bet_size_lines`) so the solver only ever
offers ½-pot, pot, and all-in — which snap cleanly to indices 2/3/4. The
mapping is done at the NODE level (`map_node_actions`) because classifying a
bet as ½ vs pot vs all-in needs the whole action set (rank order) and the
effective stack (all-in detection).
"""
from typing import Dict, List, Optional, Tuple

# Project action indices (kept local so this module has no heavy imports).
FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN = 0, 1, 2, 3, 4
PROJECT_VECTOR_LEN = 6  # indices 0..5 (idx 5/7 unused by GTO)

# Pot-fraction bet sizes we instruct the solver to use, per player/street.
# ½-pot -> RaiseHalf(2), pot -> RaisePot(3); all-in added via threshold -> (4).
_BET_PCTS = (50, 100)      # for BET (first-in)
_RAISE_PCTS = (100,)       # for RAISE (facing a bet) — single "pot raise"
ALLIN_THRESHOLD = 0.67     # solver collapses bets >= this fraction of stack to all-in


def bet_size_lines(streets: Tuple[str, ...] = ('flop', 'turn', 'river'),
                   extra: Optional[Dict[Tuple[str, str, str], List[int]]] = None
                   ) -> List[str]:
    """`set_bet_sizes` command lines for both players across the given streets,
    plus the all-in threshold. Sizes are pot percentages.

    `extra` injects additional sizes for specific cells, keyed by
    `(side, street, kind)` → `[pcts]` (side ∈ oop/ip, kind ∈ bet/raise). Used
    to add the EXACT pot-fraction a villain actually bet in a logged spot, so
    the solved tree contains a node matching it and the facing-bet navigation
    is exact rather than snapped to the nearest preset size.
    """
    extra = extra or {}
    lines: List[str] = []
    for who in ('oop', 'ip'):
        for st in streets:
            bets = sorted(set(_BET_PCTS) | set(extra.get((who, st, 'bet'), ())))
            raises = sorted(set(_RAISE_PCTS) | set(extra.get((who, st, 'raise'), ())))
            lines.append(f'set_bet_sizes {who},{st},bet,'
                         + ','.join(str(p) for p in bets))
            lines.append(f'set_bet_sizes {who},{st},raise,'
                         + ','.join(str(p) for p in raises))
            lines.append(f'set_bet_sizes {who},{st},allin')
    lines.append(f'set_allin_threshold {ALLIN_THRESHOLD}')
    return lines


def parse_solver_action(action_str: str) -> Tuple[str, Optional[float]]:
    """'BET 75.000000' -> ('BET', 75.0); 'CHECK' -> ('CHECK', None)."""
    parts = action_str.strip().split()
    verb = parts[0].upper()
    size = None
    if len(parts) > 1:
        try:
            size = float(parts[1])
        except ValueError:
            size = None
    return verb, size


def map_node_actions(actions: List[str], eff_stack_chips: float
                     ) -> Dict[str, int]:
    """Map every action string at a node to a project action index 0-4.

    Rules:
      FOLD -> 0 ; CHECK/CALL -> 1 ; ALLIN -> 4.
      BET/RAISE with size >= 0.9*eff_stack -> 4 (all-in by size).
      Remaining BET/RAISE: on a BETTING node (CHECK available / verb BET)
        the smallest is Raise½ (2) and the rest are RaisePot (3); on a
        RAISING node (facing a bet) all map to RaisePot (3) since we only
        configure a single pot-sized raise.
    """
    allin_size_thr = 0.9 * eff_stack_chips if eff_stack_chips > 0 else float('inf')
    verbs = [parse_solver_action(a)[0] for a in actions]
    is_bet_node = 'CHECK' in verbs or any(v == 'BET' for v in verbs)

    out: Dict[str, int] = {}
    non_allin: List[Tuple[str, float]] = []   # (action_str, size)
    for a in actions:
        verb, size = parse_solver_action(a)
        if verb == 'FOLD':
            out[a] = FOLD
        elif verb in ('CHECK', 'CALL'):
            out[a] = CALL
        elif verb == 'ALLIN':
            out[a] = ALL_IN
        elif verb in ('BET', 'RAISE'):
            if size is not None and size >= allin_size_thr:
                out[a] = ALL_IN
            else:
                non_allin.append((a, size if size is not None else 0.0))
        else:
            out[a] = CALL  # unknown verb — treat as passive (safe default)

    for i, (a, _sz) in enumerate(sorted(non_allin, key=lambda x: x[1])):
        out[a] = RAISE_HALF if (is_bet_node and i == 0) else RAISE_POT
    return out


def fold_strategy_to_vector(action_freqs: Dict[str, float],
                            eff_stack_chips: float) -> List[float]:
    """Collapse a {solver_action_str: freq} dict for one hand into a length-6
    project action vector (indices summed where actions share a bucket)."""
    vec = [0.0] * PROJECT_VECTOR_LEN
    idx_map = map_node_actions(list(action_freqs.keys()), eff_stack_chips)
    for a, freq in action_freqs.items():
        idx = idx_map.get(a, CALL)
        vec[idx] += float(freq)
    return vec
