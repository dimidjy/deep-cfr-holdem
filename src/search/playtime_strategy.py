"""
Play-time strategy refinement with GTO base + exploitative adjustments.

Three layers applied on top of the raw strategy network output:

1. **Temperature sharpening** — raises strategy probabilities to a power
   (1/T) then re-normalizes. T < 1 sharpens the distribution, suppressing
   low-probability tails (like spurious all-in on trash hands).
   Mathematically equivalent to temperature scaling on logits.

2. **All-in penalty** — multiplies the all-in probability by a penalty
   factor based on hand strength and stack-to-pot ratio.
   - Preflop: penalizes weak hands that shouldn't be shoving.
   - Postflop: penalizes shoving when hand strength is low AND the bet
     is a massive overbet relative to the pot (risking 100 to win 5).

3. **Exploit layer** — adjusts bluff/aggression frequencies against weak
   opponent pools (calling stations). Controlled by exploit_factor:
     0.0 = pure GTO (no adjustment)
     1.0 = full exploit (max bluff reduction against stations)
   The layer dampens raises with weak hands and boosts value bets with
   strong hands, since fish overcall and under-fold.

Usage:
    # Pure GTO
    refiner = PlaytimeStrategy(deep_cfr, num_players=6, temperature=0.5)

    # Exploitative vs fish
    refiner = PlaytimeStrategy(deep_cfr, num_players=6, temperature=0.5,
                               exploit_factor=0.7)
"""

import numpy as np
from typing import List

FOLD = 0
CALL = 1
RAISE_HALF = 2
RAISE_FULL = 3
ALL_IN = 4


class PlaytimeStrategy:

    def __init__(self, deep_cfr, num_players: int = 6,
                 temperature: float = 0.5,
                 allin_penalty: bool = True,
                 exploit_factor: float = 0.0):
        """
        Args:
            deep_cfr:       trained DeepCFR instance
            num_players:    number of players (for observation layout)
            temperature:    < 1.0 sharpens, 1.0 = no change, > 1.0 softens
            allin_penalty:  if True, apply hand-strength-based all-in penalty
            exploit_factor: 0.0 = pure GTO, 1.0 = max exploit vs calling stations
        """
        self.deep_cfr = deep_cfr
        self.num_players = num_players
        self.temperature = temperature
        self.allin_penalty = allin_penalty
        self.exploit_factor = max(0.0, min(1.0, exploit_factor))

        # Observation layout indices (depend on num_players N):
        #   Position:       N+2 dims  [0 .. N+1]
        #   PotOdds:        3 dims    [N+2 .. N+4]
        #     → pot_normalized, stack_to_pot, pot_odds
        #   HandStrength:   5 dims    [N+5 .. N+9]
        #     → is_pair, is_suited, high_card_norm, low_card_norm, rank_gap_norm
        #   Stack:          2 dims    [N+10 .. N+11]
        #     → m_ratio, relative_stack
        #   Equity:         8 dims    [N+12 .. N+19]
        #     → hand_strength, made_hand_cat, has_flush_draw, has_straight_draw,
        #       top_pair, board_is_paired, board_flush_texture, board_straight_texture
        #   Street:         4 dims    [N+20 .. N+23]
        #     → preflop, flop, turn, river one-hot
        #   GameState:      3 dims    [N+24 .. N+26]
        #     → players_alive_norm, call_amount_norm, aggression_level
        #   BettingHistory: 5 dims    [N+27 .. N+31]
        #     → my_pot_share, not_raise_fraction, facing_3bet_plus,
        #       pot_was_raised, min_opp_stack_norm
        N = num_players
        self._idx_pot_normalized = N + 2
        self._idx_stack_to_pot   = N + 3
        self._idx_is_pair        = N + 5
        self._idx_high_card      = N + 7
        self._idx_low_card       = N + 8
        self._idx_equity         = N + 12   # hand_strength / treys percentile
        self._idx_made_hand_cat  = N + 13   # 0=high card .. 1=straight flush
        self._idx_has_flush_draw = N + 14   # now hero_flush_participation (0/0.5/1.0)
        self._idx_has_str_draw   = N + 15   # now hero_straight_participation (0/0.5/1.0)
        self._idx_top_pair       = N + 16
        self._idx_street_preflop = N + 21   # shifted +1 due to new hand_rank_within_cat dim
        self._idx_street_flop    = N + 22
        self._idx_street_turn    = N + 23
        self._idx_street_river   = N + 24
        self._idx_aggression     = N + 27  # aggression_level from GameState

    def get_strategy(self, observation: np.ndarray, player_idx: int,
                     legal_actions: List[int]) -> np.ndarray:
        """Get refined strategy with temperature and all-in penalty applied."""
        strategy = self.deep_cfr.get_strategy(observation, player_idx, legal_actions)

        # --- Temperature sharpening ---
        if self.temperature != 1.0 and self.temperature > 0:
            strategy = self._apply_temperature(strategy, legal_actions)

        # --- All-in penalty ---
        if self.allin_penalty and ALL_IN in legal_actions:
            strategy = self._apply_allin_penalty(strategy, observation, legal_actions)

        # --- Exploit layer (bluff dampening vs calling stations) ---
        if self.exploit_factor > 0:
            strategy = self._apply_exploit(strategy, observation, legal_actions)

        return strategy

    def _apply_temperature(self, strategy: np.ndarray,
                           legal_actions: List[int]) -> np.ndarray:
        """
        Temperature sharpening: strategy^(1/T) then re-normalize.
        Equivalent to softmax(logits/T) on the original logits.
        """
        exponent = 1.0 / self.temperature
        sharpened = np.power(np.maximum(strategy, 1e-12), exponent)

        # Zero out illegal actions
        mask = np.zeros_like(sharpened)
        mask[legal_actions] = 1.0
        sharpened *= mask

        total = sharpened.sum()
        if total > 0:
            return sharpened / total
        return mask / mask.sum()

    def _apply_allin_penalty(self, strategy: np.ndarray,
                             observation: np.ndarray,
                             legal_actions: List[int]) -> np.ndarray:
        """
        Reduce all-in probability based on hand strength and context.

        Preflop: penalty based on hand quality (pairs, equity).
        Postflop: penalty based on made-hand strength AND stack-to-pot ratio.
          - Weak hand + large overbet = heavy penalty
          - Strong made hand = no penalty
          - Draws get moderate leniency (semi-bluff)
        """
        if len(observation) <= self._idx_street_river:
            return strategy

        is_preflop = observation[self._idx_street_preflop] > 0.5

        if is_preflop:
            penalty = self._preflop_penalty(observation)
        else:
            penalty = self._postflop_penalty(observation)

        if penalty >= 1.0:
            return strategy

        strategy = strategy.copy()
        strategy[ALL_IN] *= penalty

        # Also penalize raise_full when penalty is strong (< 0.3)
        # to prevent "raise pot with air" when all-in is suppressed
        if penalty < 0.3 and RAISE_FULL in legal_actions:
            raise_full_penalty = penalty + 0.3  # softer: 0.3 → 0.6, 0.05 → 0.35
            strategy[RAISE_FULL] *= min(raise_full_penalty, 1.0)

        # Re-normalize
        legal_mask = np.zeros_like(strategy)
        legal_mask[legal_actions] = 1.0
        strategy *= legal_mask
        total = strategy.sum()
        if total > 0:
            strategy /= total
        else:
            strategy = legal_mask / legal_mask.sum()

        return strategy

    def _preflop_penalty(self, observation: np.ndarray) -> float:
        """Penalty for preflop all-in based on hand quality."""
        is_pair   = observation[self._idx_is_pair]
        high_card = observation[self._idx_high_card]
        equity    = observation[self._idx_equity]

        if is_pair > 0.5:
            hand_quality = 0.4 + 0.6 * high_card  # 0.4 (22) to 1.0 (AA)
        else:
            hand_quality = equity

        if hand_quality >= 0.75:
            return 1.0
        elif hand_quality >= 0.50:
            return 0.2 + (hand_quality - 0.50) / 0.25 * 0.8
        else:
            return max(0.05, 0.05 + (hand_quality - 0.35) / 0.15 * 0.15)

    def _postflop_penalty(self, observation: np.ndarray) -> float:
        """
        Penalty for postflop all-in based on hand strength and bet sizing.

        Core idea: all-in is only justified when:
          A) You have a very strong made hand (set+, two pair+, strong top pair), OR
          B) You have a strong draw (flush draw / OESD) as a semi-bluff, OR
          C) The pot is already large relative to your stack (committed)

        Shoving 100 chips into a 5-chip pot with ace high = heavy penalty.
        """
        hand_strength  = observation[self._idx_equity]       # 0=worst, 1=nuts
        made_hand_cat  = observation[self._idx_made_hand_cat] # 0=high card, 1=SF
        has_flush_draw = observation[self._idx_has_flush_draw]
        has_str_draw   = observation[self._idx_has_str_draw]
        top_pair       = observation[self._idx_top_pair]
        stack_to_pot   = observation[self._idx_stack_to_pot]  # log-scaled, 0=no stack, 1=huge

        # stack_to_pot is log1p(raw_stp)/log1p(50), so:
        #   0.0 = all-in / no stack left
        #   ~0.18 = stack equals pot (SPR=1)
        #   ~0.35 = stack = 3x pot
        #   ~0.55 = stack = 10x pot
        #   ~0.73 = stack = 20x pot
        #   1.0 = stack = 50x pot
        # When SPR is high, all-in is a massive overbet → needs strong hand.

        # --- Strong made hands: no penalty ---
        # made_hand_cat: 0=high card, 0.125=pair, 0.25=two pair, 0.375=trips,
        #                0.5=straight, 0.625=flush, 0.75=FH, 0.875=quads, 1.0=SF
        if made_hand_cat >= 0.25:   # two pair or better
            return 1.0

        # Strong top pair with good kicker + small SPR
        if top_pair > 0.5 and hand_strength >= 0.55 and stack_to_pot < 0.4:
            return 0.8

        # --- Draws: moderate leniency for semi-bluffs ---
        has_draw = has_flush_draw > 0.0 or has_str_draw > 0.0
        if has_draw:
            # Allow semi-bluff all-in when not a huge overbet
            if stack_to_pot < 0.35:   # SPR < ~3
                return 0.7
            elif stack_to_pot < 0.55:  # SPR < ~10
                return 0.3
            else:
                return 0.1  # massive overbet with just a draw

        # --- Weak hands (pair or less, no draw) ---
        # The weaker the hand AND the bigger the overbet, the harsher the penalty.
        # hand_strength < 0.5 is roughly bottom half of hands.
        if stack_to_pot < 0.18:
            # Pot-committed (SPR < 1): mild penalty even with weak hands
            penalty = max(0.3, hand_strength)
        elif stack_to_pot < 0.35:
            # SPR 1-3: moderate penalty
            penalty = max(0.1, hand_strength * 0.5)
        elif stack_to_pot < 0.55:
            # SPR 3-10: strong penalty for weak hands
            penalty = max(0.05, hand_strength * 0.2)
        else:
            # SPR 10+: massive overbet — only nuts should shove
            penalty = max(0.02, hand_strength * 0.1)

        return min(penalty, 1.0)

    # ------------------------------------------------------------------
    # Exploit layer — adjust strategy against weak/stationy opponents
    # ------------------------------------------------------------------

    def _apply_exploit(self, strategy: np.ndarray,
                       observation: np.ndarray,
                       legal_actions: List[int]) -> np.ndarray:
        """
        Exploitative adjustments against calling stations (fish).

        Uses three levers per hand tier:
        - aggr_mult: scale raise/all-in (>1 = bet more, <1 = bet less)
        - fold_mult: scale fold (<1 = fold less, >1 = fold more)
        - call_mult: scale call/check (>1 = call more) — critical for
          medium hands where we want to call down, not raise or fold

        Core principles vs fish:
        1. They overcall → value bet big hands, don't bluff
        2. They're passive → when they bet they have it, but they also
           overvalue weak hands, so call down with medium+ strength
        3. Draws have implied odds (fish pay off) → call, don't semi-bluff
        4. Busted draws / air on river → give up completely

        The exploit_factor (0-1) controls deviation strength from GTO.
        """
        if len(observation) <= self._idx_aggression:
            return strategy

        hand_strength = observation[self._idx_equity]
        made_hand_cat = observation[self._idx_made_hand_cat]
        has_flush_draw = observation[self._idx_has_flush_draw] > 0.0
        has_str_draw = observation[self._idx_has_str_draw] > 0.0
        top_pair = observation[self._idx_top_pair] > 0.5
        is_preflop = observation[self._idx_street_preflop] > 0.5
        is_river = observation[self._idx_street_river] > 0.5

        if is_preflop:
            return strategy

        ef = self.exploit_factor
        has_draw = has_flush_draw or has_str_draw
        call_mult = 1.0  # default: no call adjustment

        if made_hand_cat >= 0.375:
            # Trips+ — max value, never fold. Fish pay off.
            aggr_mult = 1.0 + 0.4 * ef
            fold_mult = max(0.0, 1.0 - 0.95 * ef)

        elif made_hand_cat >= 0.25:
            # Two pair — strong value, almost never fold
            aggr_mult = 1.0 + 0.3 * ef
            fold_mult = max(0.05, 1.0 - 0.85 * ef)

        elif top_pair and hand_strength >= 0.5:
            # Good top pair — the key battleground vs fish
            # They bet with worse (weak pairs, draws), so CALL more
            # Don't raise as much (they fold worse, call better)
            aggr_mult = 1.0 + 0.1 * ef
            fold_mult = max(0.1, 1.0 - 0.6 * ef)
            call_mult = 1.0 + 0.4 * ef  # big call boost

        elif made_hand_cat >= 0.125:
            # Weak/middle pair — call down more vs fish
            # They overvalue their hands, our pair often holds up
            aggr_mult = 1.0
            fold_mult = max(0.35, 1.0 - 0.3 * ef)
            call_mult = 1.0 + 0.3 * ef  # call boost

        elif has_draw and not is_river:
            # Live draws — great implied odds vs fish who pay off
            # Don't semi-bluff (they don't fold), just call cheaply
            aggr_mult = max(0.35, 1.0 - 0.5 * ef)
            fold_mult = max(0.35, 1.0 - 0.4 * ef)
            call_mult = 1.0 + 0.35 * ef  # prefer calling to raising

        elif has_draw and is_river:
            # Missed draw on river = busted, zero equity
            aggr_mult = max(0.05, 1.0 - 0.9 * ef)
            fold_mult = 1.0 + 0.5 * ef

        else:
            # Air — no equity, no draw
            if is_river:
                # River air vs stations = pure punt, never bluff
                aggr_mult = max(0.05, 1.0 - 0.9 * ef)
                fold_mult = 1.0 + 0.5 * ef
            else:
                # Flop/turn air — mostly give up, small chance to improve
                aggr_mult = max(0.15, 1.0 - 0.7 * ef)
                fold_mult = 1.0 + 0.25 * ef

        strategy = strategy.copy()

        # Apply multipliers
        for a in [RAISE_HALF, RAISE_FULL, ALL_IN]:
            if a in legal_actions:
                strategy[a] *= aggr_mult
        if FOLD in legal_actions:
            strategy[FOLD] *= fold_mult
        if CALL in legal_actions:
            strategy[CALL] *= call_mult

        # Re-normalize over legal actions
        legal_mask = np.zeros_like(strategy)
        legal_mask[legal_actions] = 1.0
        strategy *= legal_mask
        total = strategy.sum()
        if total > 0:
            strategy /= total
        else:
            strategy = legal_mask / legal_mask.sum()

        return strategy
