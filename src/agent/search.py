"""
Real-time subtree search for postflop decisions.

Simulates each action to terminal using:
  - Hero: tries each legal action, then follows model strategy on later streets
  - Opponent: uses the Deep CFR model to predict actions
  - Board: deals random turn/river cards
  - Showdown: eval7 hand evaluation

For each of hero's actions, runs N simulations to terminal and averages
the rewards. Overrides the model when it picks a clearly -EV action.

Runs in ~0.5-2 seconds per decision (configurable via n_sims).
"""

import numpy as np
import random
from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    import eval7
    _EVAL7 = True
except ImportError:
    _EVAL7 = False

FOLD = 0
CALL = 1
RAISE_HALF = 2
RAISE_POT = 3
ALL_IN = 4

STREETS = ['preflop', 'flop', 'turn', 'river']


def _make_deck_minus(exclude: List[str]) -> list:
    """Build eval7 deck excluding known cards."""
    exclude_set = set(exclude)
    return [eval7.Card(r + s)
            for r in '23456789TJQKA' for s in 'shdc'
            if (r + s) not in exclude_set]


def _evaluate_showdown(hole_hero: list, hole_opp: list, board: list) -> float:
    """Evaluate showdown. Returns +1 win, 0 tie, -1 loss for hero."""
    hero_score = eval7.evaluate(hole_hero + board)
    opp_score = eval7.evaluate(hole_opp + board)
    if hero_score > opp_score:
        return 1.0
    elif hero_score < opp_score:
        return -1.0
    return 0.0


def _get_model_action(model, obs: np.ndarray, legal: List[int]) -> int:
    """Get action from the Deep CFR model for opponent simulation."""
    try:
        strategy = model.get_strategy(obs, 0, legal)
        # Sample from strategy
        probs = np.array([strategy[a] if a < len(strategy) else 0.0 for a in legal])
        probs = np.maximum(probs, 0.0)
        s = probs.sum()
        if s > 0:
            probs /= s
        else:
            probs = np.ones(len(legal)) / len(legal)
        return int(np.random.choice(legal, p=probs))
    except Exception:
        return int(np.random.choice(legal))


def _build_full_obs(hole_cards: List[str], board: List[str],
                    pot_bb: float, stack_bb: float,
                    rel_pos: int, n_active: int,
                    facing_bet: bool, call_bb: float,
                    button_seat: int = 0,
                    my_seat: int = 0) -> np.ndarray:
    """Build a full 50-dim observation using the agent ObservationBuilder.

    Uses equity model (cached) for proper hand strength features.
    """
    try:
        from src.agent.obs_builder import ObservationBuilder
        from src.agent.game_state import GameState, PlayerInfo, CardInfo

        cards = [CardInfo(c[0], c[1]) for c in hole_cards]
        community = [CardInfo(c[0], c[1]) for c in board]

        # Build minimal player list
        players = []
        for i in range(max(n_active, 2)):
            p = PlayerInfo(
                seat=i,
                name=f'P{i}',
                stack=stack_bb if i == my_seat else 100.0,
                is_active=True,
                has_button=(i == button_seat),
                hole_cards=cards if i == my_seat else None,
            )
            players.append(p)

        gs = GameState(
            players=players,
            my_seat=my_seat,
            community_cards=community,
            pot_total=pot_bb,
            street='flop' if len(board) <= 3 else ('turn' if len(board) == 4 else 'river'),
            button_seat=button_seat,
            small_blind=0.5,
            big_blind=1.0,
            legal_actions=[0, 1, 2, 3],
            num_active_players=n_active,
            call_amount=call_bb if facing_bet else 0.0,
        )

        builder = ObservationBuilder(num_players=6)
        obs = builder.build(gs)
        if obs is not None:
            # Pad to 50 if needed (model expects behavioral stats dims)
            if len(obs) < 50:
                padded = np.zeros(50, dtype=np.float32)
                padded[:len(obs)] = obs
                return padded
            return obs
    except Exception:
        pass

    # Fallback: zeros (model will make a default decision)
    return np.zeros(50, dtype=np.float32)


def simulate_to_terminal(
    hero_action: int,
    hole_hero: List[str],
    hole_opp: List[str],
    board_so_far: List[str],
    pot_bb: float,
    hero_stack: float,
    opp_stack: float,
    street: str,
    hero_invested: float,
    model=None,
    hero_pos: int = 0,
    opp_pos: int = 2,
) -> float:
    """Simulate one hand from hero's action to terminal, return hero's profit.

    Uses the Deep CFR model for BOTH players' decisions on future streets:
      - Opponent: model sees opponent's actual cards → realistic fold/call/raise
      - Hero: model sees hero's cards → consistent with how hero would play

    This means an opponent with 88 on 478 will NOT fold to a bet (model knows
    they have top pair), while an opponent with 32o WILL fold.
    """
    half_pot = pot_bb * 0.5
    pot_size = pot_bb

    def _model_decision(cards, board, pot, stack, pos, facing, call_bb, legal):
        """Get model's action for a player using full obs with equity."""
        if model is None:
            r = random.random()
            if facing:
                return FOLD if r < 0.35 else (CALL if r < 0.95 else RAISE_POT)
            else:
                return CALL if r < 0.65 else (RAISE_HALF if r < 0.85 else RAISE_POT)
        obs = _build_full_obs(cards, board, pot, stack, pos, 2, facing, call_bb)
        return _get_model_action(model, obs, legal)

    # Handle hero's current action
    if hero_action == FOLD:
        return -hero_invested

    elif hero_action == CALL:
        pass  # check/call, continue

    elif hero_action in (RAISE_HALF, RAISE_POT, ALL_IN):
        if hero_action == RAISE_HALF:
            raise_size = min(half_pot, hero_stack)
        elif hero_action == RAISE_POT:
            raise_size = min(pot_size, hero_stack)
        else:
            raise_size = hero_stack

        pot_bb += raise_size
        hero_invested += raise_size
        hero_stack -= raise_size

        # Opponent responds — MODEL sees opponent's actual hand
        opp_legal = [FOLD, CALL]
        if opp_stack > raise_size * 2:
            opp_legal.append(RAISE_POT)
        opp_action = _model_decision(
            hole_opp, board_so_far, pot_bb, opp_stack,
            opp_pos, True, raise_size, opp_legal)

        if opp_action == FOLD:
            return pot_bb - hero_invested

        elif opp_action == CALL:
            call_amt = min(raise_size, opp_stack)
            pot_bb += call_amt
            opp_stack -= call_amt

        elif opp_action in (RAISE_POT, RAISE_HALF, ALL_IN):
            reraise = min(pot_bb, opp_stack)
            pot_bb += reraise
            opp_stack -= reraise
            call_amt = min(reraise, hero_stack)
            pot_bb += call_amt
            hero_invested += call_amt
            hero_stack -= call_amt

    # Deal remaining board cards
    all_known = hole_hero + hole_opp + board_so_far
    deck = _make_deck_minus([c if isinstance(c, str) else str(c) for c in all_known])
    random.shuffle(deck)

    full_board = list(board_so_far)
    didx = 0

    # Play remaining streets with model decisions
    street_order = ['flop', 'turn', 'river']
    current_idx = street_order.index(street) if street in street_order else 0

    for st_idx in range(current_idx + 1, 3):
        # Deal next card
        if len(full_board) < (st_idx + 3):  # flop=3, turn=4, river=5
            target = st_idx + 3
            while len(full_board) < target and didx < len(deck):
                full_board.append(str(deck[didx]))
                didx += 1

        st_name = street_order[st_idx]

        if hero_stack <= 0 or opp_stack <= 0:
            break

        # Hero acts first (simplified — ignores position)
        hero_legal = [CALL]  # check
        if hero_stack > 0:
            hero_legal.extend([RAISE_HALF, RAISE_POT])
        hero_act = _model_decision(
            hole_hero, full_board, pot_bb, hero_stack,
            hero_pos, False, 0, hero_legal)

        if hero_act in (RAISE_HALF, RAISE_POT):
            bet = min(pot_bb * (0.5 if hero_act == RAISE_HALF else 1.0), hero_stack)
            pot_bb += bet
            hero_invested += bet
            hero_stack -= bet

            # Opponent responds
            opp_legal = [FOLD, CALL]
            opp_act = _model_decision(
                hole_opp, full_board, pot_bb, opp_stack,
                opp_pos, True, bet, opp_legal)

            if opp_act == FOLD:
                return pot_bb - hero_invested
            else:
                call_amt = min(bet, opp_stack)
                pot_bb += call_amt
                opp_stack -= call_amt
        else:
            # Hero checked — opponent may bet
            opp_legal = [CALL, RAISE_HALF, RAISE_POT]
            opp_act = _model_decision(
                hole_opp, full_board, pot_bb, opp_stack,
                opp_pos, False, 0, opp_legal)

            if opp_act in (RAISE_HALF, RAISE_POT):
                bet = min(pot_bb * (0.5 if opp_act == RAISE_HALF else 1.0), opp_stack)
                pot_bb += bet
                opp_stack -= bet

                # Hero responds
                hero_resp_legal = [FOLD, CALL]
                hero_resp = _model_decision(
                    hole_hero, full_board, pot_bb, hero_stack,
                    hero_pos, True, bet, hero_resp_legal)

                if hero_resp == FOLD:
                    return -hero_invested
                else:
                    call_amt = min(bet, hero_stack)
                    pot_bb += call_amt
                    hero_invested += call_amt
                    hero_stack -= call_amt

    # Fill remaining board for showdown
    while len(full_board) < 5 and didx < len(deck):
        full_board.append(str(deck[didx]))
        didx += 1

    # Showdown
    e7_hero = [eval7.Card(c) if isinstance(c, str) else c for c in hole_hero]
    e7_opp = [eval7.Card(c) if isinstance(c, str) else c for c in hole_opp]
    e7_board = [eval7.Card(c) if isinstance(c, str) else c for c in full_board[:5]]

    result = _evaluate_showdown(e7_hero, e7_opp, e7_board)

    if result > 0:
        return pot_bb - hero_invested
    elif result < 0:
        return -hero_invested
    else:
        return pot_bb * 0.5 - hero_invested


def search_ev(
    hole_cards: List[str],
    board: List[str],
    pot_bb: float,
    call_amount_bb: float,
    stack_bb: float,
    street: str,
    legal_actions: List[int],
    n_active: int = 2,
    model=None,
    n_sims: int = 200,
) -> dict:
    """Run Monte Carlo simulations for each action, return EV dict.

    For each action:
      - Deal random opponent hand
      - Simulate to terminal (opponent uses model or NL5 heuristic)
      - Record hero's profit
      - Average over n_sims
    """
    if not _EVAL7 or not board:
        return {}

    deck = _make_deck_minus(hole_cards + board)

    # In raised pots, sample opponent hands from a realistic calling range
    # instead of random. NL5 callers have ~30% of hands: pairs, broadways,
    # suited connectors, Ax. This dramatically changes equity for weak hands.
    opp_hands_pool = None
    if _EVAL7 and pot_bb > 4.0:
        try:
            excluded = set(hole_cards + board)
            all_combos = []
            for i, c1 in enumerate(deck):
                for c2 in deck[i+1:]:
                    s1, s2 = str(c1), str(c2)
                    if s1 not in excluded and s2 not in excluded:
                        # Check if hand is in NL5 calling range
                        r1, r2 = s1[0], s2[0]
                        suit1, suit2 = s1[1], s2[1]
                        suited = suit1 == suit2
                        ranks = '23456789TJQKA'
                        ri1, ri2 = ranks.index(r1), ranks.index(r2)
                        hi, lo = max(ri1, ri2), min(ri1, ri2)
                        # NL5 calling range: pairs, A7o+, A2s+, K9o+, K8s+, etc.
                        in_range = False
                        if ri1 == ri2:  # pair
                            in_range = True
                        elif suited:
                            if hi >= 12:  # Axs
                                in_range = True
                            elif hi >= 11 and lo >= 6:  # K8s+
                                in_range = True
                            elif hi >= 10 and lo >= 7:  # Q9s+
                                in_range = True
                            elif hi >= 9 and lo >= 7:  # J9s+
                                in_range = True
                            elif hi >= 8 and lo >= 6:  # T8s+
                                in_range = True
                            elif hi - lo <= 1 and lo >= 3:  # suited connectors 54s+
                                in_range = True
                        else:
                            if hi >= 12 and lo >= 5:  # A7o+
                                in_range = True
                            elif hi >= 11 and lo >= 7:  # K9o+
                                in_range = True
                            elif hi >= 10 and lo >= 8:  # QTo+
                                in_range = True
                            elif hi >= 9 and lo >= 8:  # JTo
                                in_range = True
                        if in_range:
                            all_combos.append((s1, s2))
            if all_combos:
                opp_hands_pool = all_combos
        except Exception:
            pass

    evs = {a: [] for a in legal_actions}

    for _ in range(n_sims):
        # Deal opponent hand from range or random
        if opp_hands_pool:
            opp_hand = list(opp_hands_pool[random.randint(0, len(opp_hands_pool) - 1)])
        else:
            random.shuffle(deck)
            opp_hand = [str(deck[0]), str(deck[1])]

        for action in legal_actions:
            profit = simulate_to_terminal(
                hero_action=action,
                hole_hero=hole_cards,
                hole_opp=opp_hand,
                board_so_far=board,
                pot_bb=pot_bb,
                hero_stack=stack_bb,
                opp_stack=100.0,  # assume 100BB opponent
                street=street,
                hero_invested=0.0,
                model=model,
            )
            evs[action].append(profit)

    return {a: np.mean(v) if v else 0.0 for a, v in evs.items()}


def search_override(
    strategy: np.ndarray,
    hole_cards: List[str],
    board: List[str],
    pot_bb: float,
    call_amount_bb: float,
    stack_bb: float,
    street: str,
    legal_actions: List[int],
    n_active: int = 2,
    min_ev_diff: float = 2.0,
    model=None,
    n_sims: int = 200,
) -> Tuple[np.ndarray, Optional[dict]]:
    """Override model strategy when search finds a clearly better action.

    Runs full Monte Carlo simulation to terminal for each action.
    Overrides only when EV difference exceeds min_ev_diff BB.
    """
    if street == 'preflop' or not board:
        return strategy, None

    if not _EVAL7:
        return strategy, None

    # Adjust sims for multiway (more opponents = more variance = need more sims)
    sims = n_sims
    if n_active >= 3:
        sims = max(sims, 300)

    evs = search_ev(
        hole_cards, board, pot_bb, call_amount_bb, stack_bb,
        street, legal_actions, n_active, model=model, n_sims=sims)

    if not evs:
        return strategy, None

    best_ev_action = max(evs, key=evs.get)
    model_action = max(legal_actions, key=lambda a: strategy[a] if a < len(strategy) else 0)

    best_ev = evs[best_ev_action]
    model_ev = evs.get(model_action, 0)

    # Override if model's action is clearly worse
    if model_ev < best_ev - min_ev_diff:
        new_strategy = strategy.copy()
        # Blend: 70% EV-optimal, 30% model
        for a in legal_actions:
            if a < len(new_strategy):
                new_strategy[a] *= 0.3
        if best_ev_action < len(new_strategy):
            new_strategy[best_ev_action] += 0.7
        # Renormalize
        total = sum(new_strategy[a] for a in legal_actions if a < len(new_strategy))
        if total > 0:
            for a in range(len(new_strategy)):
                new_strategy[a] /= total
        return new_strategy, evs

    return strategy, evs
