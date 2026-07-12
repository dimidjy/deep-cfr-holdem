"""Navigate a solved TexasSolver tree to hero's decision node and read it.

The solved tree's root is the OOP player's first decision on the solved
street (OOP acts first postflop). `childrens` is keyed by action strings
('CHECK', 'BET 100.000000', 'RAISE 200.000000', 'CALL', 'FOLD'). We descend
along the line that reproduces the logged spot — determined by `nav_type` on
the `GTOSpotInputs` — until we reach hero's node, then collapse hero's hand's
action frequencies into the project's length-6 action vector.

Bet-size matching: the solver offers the pot-fraction sizes we configured
(½-pot, pot, all-in). The observed villain bet (as a fraction of the
street-start pot) is matched to the nearest offered size by tracking the pot
down the line.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.gto.action_mapping import (
    parse_solver_action, fold_strategy_to_vector, map_node_actions,
    PROJECT_VECTOR_LEN, FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN,
)
from src.gto.spot_inputs import GTOSpotInputs

# Project action indices that aren't model outputs but appear in logs.
_PLAYED_IDX_REMAP = {5: 2, 7: 2}   # RaiseMin / RaiseThird → RaiseHalf bucket

# made_hand_cat below this = no showdown value (high card / busted draw). A
# bet/raise with such a hand is a pure bluff — what --bluff-factor trims. Set
# just under HAND_ONE_PAIR (0.125) so even weak/board pairs (thin value vs a
# calling-station pool) are left intact.
_BLUFF_MADE_HAND_MAX = 0.0625
_AGGRESSIVE_IDX = (RAISE_HALF, RAISE_POT, ALL_IN)
_FACING_BET_NAVS = ('ip_face_bet', 'oop_check_ip_bet', 'oop_bet_ip_raise')


def _apply_bluff_reduction(vec, inputs, bluff_factor):
    """Trim hero's bluff frequency for a no-showdown-value hand.

    Exploit tilt for a calling-station pool: pure GTO bluffs at the
    indifference frequency, which is -EV against players who don't fold.
    When hero's hand is high-card (no showdown value) and bluff_factor < 1,
    scale the bet/raise mass by bluff_factor and move the removed mass to the
    realistic give-up: CHECK if hero is betting/checked-to, FOLD if hero faces
    a bet. Value hands (any pair+) are never touched.
    """
    if bluff_factor >= 1.0:
        return vec
    if inputs.hero_made_hand_cat >= _BLUFF_MADE_HAND_MAX:
        return vec
    removed = sum(vec[i] for i in _AGGRESSIVE_IDX) * (1.0 - bluff_factor)
    if removed <= 0.0:
        return vec
    for i in _AGGRESSIVE_IDX:
        vec[i] *= bluff_factor
    if inputs.nav_type in _FACING_BET_NAVS:
        vec[FOLD] += removed
    else:
        vec[CALL] += removed          # CALL idx doubles as CHECK when not facing a bet
    return vec


def _nav_steps(inputs: GTOSpotInputs) -> List[Tuple[str, Optional[float]]]:
    """Action-edge steps from root to hero's node for each nav_type."""
    f = inputs.facing_bet_frac_start_pot
    return {
        'root_hero': [],
        'ip_after_check': [('CHECK', None)],
        'ip_face_bet': [('BET', f)],
        'oop_check_ip_bet': [('CHECK', None), ('BET', f)],
        'oop_bet_ip_raise': [('BET', None), ('RAISE', f)],
    }.get(inputs.nav_type, [])


def _follow(children: Dict[str, dict], verb: str,
            target_frac: Optional[float], pot: float
            ) -> Tuple[Optional[str], Optional[dict], float]:
    """Pick the child edge for `verb`, returning (action_str, child, new_pot).

    For CHECK/CALL/FOLD the match is exact. For BET/RAISE (and the ALLIN that
    a big bet collapses to), candidates are ranked by chip size; with a
    `target_frac` we pick the size nearest `target_frac * pot`, else the
    smallest. `new_pot` adds the chosen bet so deeper edges match correctly.
    """
    cands = []   # (action_str, child, size)
    for a, child in children.items():
        v, sz = parse_solver_action(a)
        if v == verb or (verb in ('BET', 'RAISE') and v == 'ALLIN'):
            cands.append((a, child, sz if sz is not None else float('inf')))
    if not cands:
        return None, None, pot
    if verb in ('CHECK', 'CALL', 'FOLD'):
        a, child, _ = cands[0]
        return a, child, pot
    cands.sort(key=lambda x: x[2])
    if target_frac is None:
        a, child, sz = cands[0]
    else:
        desired = target_frac * pot
        a, child, sz = min(cands, key=lambda x: abs(x[2] - desired))
    new_pot = pot + (sz if sz != float('inf') else 0.0)
    return a, child, new_pot


def select_hero_node(result: dict, inputs: GTOSpotInputs
                     ) -> Tuple[Optional[dict], str, Optional[str]]:
    """Descend to hero's decision node. Returns (node, path_str, error)."""
    node = result
    pot = float(inputs.pot_chips)
    path: List[str] = []
    for verb, frac in _nav_steps(inputs):
        children = node.get('childrens') or {}
        if not children:
            return None, _path_str(path), f'no children before {verb}'
        a, child, pot = _follow(children, verb, frac, pot)
        if child is None:
            return (None, _path_str(path),
                    f'no {verb} child (have {sorted(children)})')
        path.append(a)
        node = child
    return node, _path_str(path), None


def _path_str(path: List[str]) -> str:
    return 'root' if not path else 'root>' + '>'.join(path)


def _combo_candidates(combo: str) -> List[str]:
    """Hero combo string in both card orders (solver canonicalizes one)."""
    if len(combo) >= 4:
        c1, c2 = combo[:2], combo[2:4]
        return [combo, c2 + c1]
    return [combo]


def _played_idx(action_idx: int) -> Optional[int]:
    if action_idx is None or action_idx < 0:
        return None
    idx = _PLAYED_IDX_REMAP.get(action_idx, action_idx)
    return idx if 0 <= idx < PROJECT_VECTOR_LEN else None


def extract_gto_strategy(node: dict, inputs: GTOSpotInputs,
                         bluff_factor: float = 1.0) -> dict:
    """Read hero's hand's frequencies at `node` into project action terms.

    `bluff_factor` < 1.0 trims hero's bluff frequency on no-showdown-value
    hands (see `_apply_bluff_reduction`) before computing the modal action.
    """
    s = node.get('strategy') or {}
    actions = s.get('actions')
    strat = s.get('strategy')
    if not actions or not isinstance(strat, dict):
        return {'gto_error': 'node has no strategy'}
    freqs = None
    for cand in _combo_candidates(inputs.hero_hand_combo):
        if cand in strat:
            freqs = strat[cand]
            break
    if freqs is None:
        return {'gto_error':
                f'hero combo {inputs.hero_hand_combo} not in solved range'}
    action_freqs = {a: float(f) for a, f in zip(actions, freqs)}
    vec = fold_strategy_to_vector(action_freqs, inputs.eff_stack_chips)
    vec = _apply_bluff_reduction(vec, inputs, bluff_factor)
    gto_action = int(np.argmax(vec)) if any(vec) else -1
    pidx = _played_idx(inputs.action_idx)
    gto_freq_played = float(vec[pidx]) if pidx is not None else None
    return {
        'gto_strategy': [round(float(x), 5) for x in vec],
        'gto_action': gto_action,
        'gto_freq_played': gto_freq_played,
        'gto_node_actions': list(actions),
        'gto_action_map': {a: i for a, i in
                           map_node_actions(actions, inputs.eff_stack_chips).items()},
    }


def solve_result_to_gto_fields(result: dict, inputs: GTOSpotInputs,
                               bluff_factor: float = 1.0) -> dict:
    """End-to-end: navigate + extract → the gto_* field dict for a spot."""
    base = {
        'gto_board': list(inputs.board),
        'gto_pot_chips': inputs.pot_chips,
        'gto_eff_stack_chips': inputs.eff_stack_chips,
        'gto_spot_key': list(inputs.hero_spot_key.as_tuple()),
        'gto_nav_type': inputs.nav_type,
        'gto_bluff_factor': bluff_factor,
    }
    node, path, err = select_hero_node(result, inputs)
    base['gto_node'] = path
    if err is not None:
        base['gto_error'] = err
        return base
    strat = extract_gto_strategy(node, inputs, bluff_factor=bluff_factor)
    base.update(strat)
    return base
