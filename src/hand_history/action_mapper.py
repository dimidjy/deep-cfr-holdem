"""Map PokerStars hand history actions to the engine's 5-action discrete space.

Action space (matching FastPokerEngine / PettingZoo texas_holdem_no_limit_v6):
    0 = Fold
    1 = Check / Call
    2 = Raise half pot
    3 = Raise pot
    4 = All-in
"""

from .models import ParsedAction, dollars_to_chips

FOLD = 0
CALL = 1       # also covers check
RAISE_HALF = 2
RAISE_POT = 3
ALL_IN = 4


def map_action(action: ParsedAction,
               pot: int,
               max_bet: int,
               player_bet_chips: int,
               player_stake: int) -> int:
    """Map a parsed HH action to the 5-action space.

    Args:
        action: The parsed action from hand history.
        pot: Current total pot in chips (sum of all pot_chips + bet_chips).
        max_bet: Current highest bet_chips among all players this round.
        player_bet_chips: This player's current bet_chips this round.
        player_stake: This player's remaining stack (chips not yet wagered).
    """
    if action.action_type == 'fold':
        return FOLD

    if action.action_type in ('check', 'call'):
        return CALL

    if action.is_all_in:
        return ALL_IN

    if action.action_type in ('bet', 'raise'):
        # For raises: amount_dollars is the "to" amount (total bet this round)
        # For bets: amount_dollars is the bet size
        action_total_chips = action.amount_chips

        if action.action_type == 'bet':
            # A bet means no prior bets this round, so raise_increment = bet amount
            raise_increment = action_total_chips
        else:
            # Raise "to X" means new total bet is X chips
            raise_increment = action_total_chips - max_bet

        # Compute what each discrete action would produce
        # (mirrors FastPokerEngine._map_action logic)
        half_pot_raise = max(int(pot / 2), 1)
        pot_raise = max(int(pot), 1)
        all_in_raise = player_stake  # entire remaining stack

        # Check if this IS an all-in (player bet everything)
        if action_total_chips >= player_stake + player_bet_chips:
            return ALL_IN

        # Find closest discrete action
        candidates = [
            (RAISE_HALF, half_pot_raise),
            (RAISE_POT, pot_raise),
            (ALL_IN, all_in_raise),
        ]

        best_action = RAISE_HALF
        best_dist = float('inf')
        for act_id, act_amount in candidates:
            dist = abs(raise_increment - act_amount)
            if dist < best_dist:
                best_dist = dist
                best_action = act_id

        return best_action

    # Blind actions shouldn't reach here, but just in case
    return CALL


def get_legal_actions(max_bet: int,
                      player_bet_chips: int,
                      player_stake: int,
                      pot: int,
                      min_raise: int = 2) -> list:
    """Determine legal actions given current game state.

    Mirrors FastPokerEngine._compute_legal_actions() logic.

    Args:
        max_bet: Highest bet_chips among all players this round.
        player_bet_chips: This player's current bet_chips this round.
        player_stake: This player's remaining stack.
        pot: Total pot.
        min_raise: Minimum raise increment (default BB=2).
    """
    call_cost = max_bet - player_bet_chips
    legal = []

    # Fold: legal if facing a bet (call_cost > 0)
    if call_cost > 0:
        legal.append(FOLD)

    # Check/Call: always legal
    legal.append(CALL)

    # Raise actions: need enough chips
    max_raise = player_stake - call_cost  # max affordable raise increment
    if max_raise < min_raise:
        return legal

    # Raise half pot
    half_pot = int(pot / 2)
    half_pot_clamped = max(half_pot, min_raise)
    if half_pot_clamped <= max_raise:
        legal.append(RAISE_HALF)

    # Raise pot
    pot_raise = int(pot)
    pot_raise_clamped = max(pot_raise, min_raise)
    if pot_raise_clamped <= max_raise:
        legal.append(RAISE_POT)

    # All-in: always legal if can raise at all
    legal.append(ALL_IN)

    return legal
