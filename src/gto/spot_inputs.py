"""Turn a logged action_debug spot JSON into the inputs a GTO solve needs.

`build_gto_spot_inputs(spot_json)` recomputes everything TexasSolver needs to
solve hero's decision: the board, the pot/effective-stack at the START of the
current street (so a faced bet stays a node in the tree, not baked into the
pot), hero's IP/OOP seat, hero's exact hand, and the OppRangeSpotKey for both
hero and villain (villain = hero's key with IP/OOP flipped — see the GTO plan).

Scope v1: heads-up (`num_active==2`), streets in `scope_streets`. Returns None
(with a reason on the object via `.skip_reason` when called with collect) for
preflop / non-HU / missing-cards / missing-obs spots.

All chip amounts are `round(bb * 100)` ints — only ratios matter to the solver.
The spot-key SPR/action-context use the decision pipeline's conventions
(pot_total, hero stack + call) so the lookup hits the same cells the agent would.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.agent.game_state import GameState
from src.opp_range_spot_key import (
    OppRangeSpotKey, build_opp_range_spot_key, hand_class_from_hole,
)
from src.postflop_spot_key import classify_position, IDX_MADE_HAND

_SCOPE_DEFAULT = ('river', 'turn')
_STREET_BOARD_LEN = {'flop': 3, 'turn': 4, 'river': 5}
_LATER_STREETS = {'flop': ('flop', 'turn', 'river'),
                  'turn': ('turn', 'river'),
                  'river': ('river',)}


@dataclass
class GTOSpotInputs:
    street: str
    board: List[str]                  # current community cards ['8c','Qd',...]
    solve_streets: Tuple[str, ...]    # streets to configure bet sizes for
    pot_chips: int                    # pot at START of street (x100 BB)
    eff_stack_chips: int              # effective stack at street start (x100 BB)
    hero_is_oop: bool                 # hero acts first postflop?
    hero_hand_combo: str              # 'Jh5h'
    hero_hand_class: str              # 'J5s'
    hero_spot_key: OppRangeSpotKey
    villain_spot_key: OppRangeSpotKey
    nav_type: str                     # how to reach hero's node (see extract.py)
    facing_bet_size_pct: Optional[float]       # call/pot_total (action_context)
    facing_bet_frac_start_pot: Optional[float] # villain_bet/pot_at_start (nav)
    action_idx: int                   # hero's logged action (leak metric)
    aggression_tier: int              # tier index for range fallback
    villain_bet_side: Optional[str]   # 'oop'/'ip' side that made the faced bet
    villain_bet_pct: Optional[int]    # faced bet as % of pot — injected into the
                                      # solver so the tree has an exact-match node
    hero_made_hand_cat: float         # obs made_hand_cat (0=high card .. 0.875=quads);
                                      # used by the --bluff-factor reduction


def _tier_from_context(action_context: str) -> int:
    """Pick a fallback aggression tier from the action context."""
    if action_context == 'face_check_raise':
        return 3
    if action_context == 'face_bet_large':
        return 2
    return 1


def build_gto_spot_inputs(spot_json: dict,
                          scope_streets: Tuple[str, ...] = _SCOPE_DEFAULT
                          ) -> Optional[GTOSpotInputs]:
    """Build solve inputs from a spot JSON, or None if out of scope."""
    gs_dict = spot_json.get('game_state')
    if not gs_dict:
        return None
    try:
        gs = GameState.from_dict(gs_dict)
    except Exception:
        return None

    street = gs.street
    if street not in scope_streets or street not in _STREET_BOARD_LEN:
        return None
    if int(gs.num_active_players) != 2:
        return None

    active = gs.active_players
    if len(active) != 2:
        return None
    hero = gs.my_player
    if hero is None or not hero.hole_cards or len(hero.hole_cards) < 2:
        return None
    villain = next((p for p in active if p.seat != hero.seat), None)
    if villain is None:
        return None

    board = [str(c) for c in gs.community_cards]
    if len(board) != _STREET_BOARD_LEN[street]:
        return None

    hole = [str(c) for c in hero.hole_cards[:2]]
    hero_hand_combo = ''.join(hole)
    hero_hand_class = hand_class_from_hole(hole)
    if hero_hand_class is None:
        return None

    obs = spot_json.get('obs_vector')
    if not obs or len(obs) < 27:
        return None
    obs = [float(x) for x in obs]

    rel_pos = gs.compute_rel_pos()
    hero_position = classify_position(rel_pos, gs.num_active_players)
    hero_is_oop = (hero_position == 'OOP')

    pot_total = float(gs.pot_total)
    call_amount = float(gs.call_amount)
    hero_bet = float(hero.bet_current_round)
    villain_bet = float(villain.bet_current_round)
    facing_bet = call_amount > 0.01

    # Pot/eff stack at the START of this street (before this street's bets) so
    # a faced bet remains a node in the solved tree.
    pot_at_start = max(pot_total - hero_bet - villain_bet, 0.01)
    hero_start_stack = hero.stack + hero_bet
    villain_start_stack = villain.stack + villain_bet
    eff_bb = min(hero_start_stack, villain_start_stack)
    pot_chips = int(round(pot_at_start * 100))
    eff_stack_chips = int(round(max(eff_bb, 0.01) * 100))

    facing_bet_size_pct = None
    if facing_bet and pot_total > 0.01:
        facing_bet_size_pct = call_amount / pot_total
    facing_bet_frac_start_pot = None
    if villain_bet > 0.01 and pot_at_start > 0.01:
        facing_bet_frac_start_pot = villain_bet / pot_at_start

    # Navigation type: how to descend the solved tree to hero's decision node.
    if hero_is_oop:
        if not facing_bet:
            nav_type = 'root_hero'              # hero OOP first to act = root
        elif hero_bet <= 0.01:
            nav_type = 'oop_check_ip_bet'       # hero checked, villain bet
        else:
            nav_type = 'oop_bet_ip_raise'       # hero bet, villain raised
    else:
        if not facing_bet:
            nav_type = 'ip_after_check'         # villain checked to hero
        else:
            nav_type = 'ip_face_bet'            # villain bet into hero

    # Inject the villain's actual bet size into the solver so the tree has an
    # exact-match node (else an 81%-pot bet snaps to the nearest preset, 50%
    # or all-in). Only for the clean single-bet navs; the bettor's side is
    # OOP when hero is IP facing a bet, IP when hero checked and villain bet.
    villain_bet_side = None
    villain_bet_pct = None
    if facing_bet_frac_start_pot:
        if nav_type == 'ip_face_bet':
            villain_bet_side = 'oop'
        elif nav_type == 'oop_check_ip_bet':
            villain_bet_side = 'ip'
        if villain_bet_side is not None:
            villain_bet_pct = max(1, int(round(facing_bet_frac_start_pot * 100)))

    # Spot keys (default_core conventions for runtime consistency).
    was_preflop_raiser = (float(hero.total_invested) - hero_bet) >= 2.5
    hero_spot_key = build_opp_range_spot_key(
        street=street, obs=obs,
        rel_pos=rel_pos, num_active=gs.num_active_players,
        effective_stack_bb=hero.stack + call_amount,
        pot_bb=pot_total if pot_total > 0 else 0.01,
        was_preflop_raiser=was_preflop_raiser,
        facing_bet_size_pct=facing_bet_size_pct,
    )
    # Villain range: same spot, position flipped (v1 approximation — action
    # context is not mirrored; documented in the plan's risks).
    villain_position = 'IP' if hero_position == 'OOP' else 'OOP'
    villain_spot_key = OppRangeSpotKey(
        street=hero_spot_key.street,
        board_texture=hero_spot_key.board_texture,
        position=villain_position,
        spr_bucket=hero_spot_key.spr_bucket,
        action_context=hero_spot_key.action_context,
    )

    return GTOSpotInputs(
        street=street,
        board=board,
        solve_streets=_LATER_STREETS[street],
        pot_chips=pot_chips,
        eff_stack_chips=eff_stack_chips,
        hero_is_oop=hero_is_oop,
        hero_hand_combo=hero_hand_combo,
        hero_hand_class=hero_hand_class,
        hero_spot_key=hero_spot_key,
        villain_spot_key=villain_spot_key,
        nav_type=nav_type,
        facing_bet_size_pct=facing_bet_size_pct,
        facing_bet_frac_start_pot=facing_bet_frac_start_pot,
        action_idx=int(spot_json.get('action_idx', -1)),
        aggression_tier=_tier_from_context(hero_spot_key.action_context),
        villain_bet_side=villain_bet_side,
        villain_bet_pct=villain_bet_pct,
        hero_made_hand_cat=float(obs[IDX_MADE_HAND]) if len(obs) > IDX_MADE_HAND else 0.0,
    )
