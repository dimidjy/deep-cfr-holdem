"""Reconstruct 41-dim observation vectors from parsed hand histories.

Replicates the exact feature computation from FastEnhancedPokerEngine
(src/engine/fast_enhanced_engine.py) using game state reconstructed
from the parsed action sequence.
"""

import math
from collections import defaultdict
from typing import List, Optional

import numpy as np

from .models import ParsedHand, ParsedAction, DecisionPoint, dollars_to_chips
from .action_mapper import map_action, get_legal_actions

try:
    from treys import Card as TreysCard, Evaluator as TreysEvaluator
    _TREYS_EVALUATOR = TreysEvaluator()
    _TREYS_AVAILABLE = True
except ImportError:
    _TREYS_AVAILABLE = False

try:
    import eval7 as _eval7
    # Build full random range once at import time
    _RANKS = '23456789TJQKA'
    _all_combos = []
    for _i, _r1 in enumerate(_RANKS):
        for _j, _r2 in enumerate(_RANKS):
            if _i < _j:
                _all_combos.append(f'{_r2}{_r1}s')
                _all_combos.append(f'{_r2}{_r1}o')
            elif _i == _j:
                _all_combos.append(f'{_r1}{_r2}')
    _EVAL7_RANDOM_RANGE = _eval7.HandRange(','.join(_all_combos))
    _EVAL7_AVAILABLE = True
except ImportError:
    _EVAL7_AVAILABLE = False


_RANK_MAP = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
}

# Treys rank_class → (best_rank_int, worst_rank_int) for within-category ranking
_TREYS_RANK_BOUNDS = {
    1: (1, 10),       # Straight Flush
    2: (11, 166),     # Four of a Kind
    3: (167, 322),    # Full House
    4: (323, 1599),   # Flush
    5: (1600, 1609),  # Straight
    6: (1610, 2467),  # Three of a Kind
    7: (2468, 3325),  # Two Pair
    8: (3326, 6185),  # One Pair
    9: (6186, 7462),  # High Card
}

_POS_LABELS_6 = {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'UTG', 4: 'MP', 5: 'CO'}


def _rank_val(rank_char: str) -> int:
    return _RANK_MAP.get(rank_char, 0)


class _PlayerState:
    """Mirrors pokers PlayerState / FastPokerEngine._Player."""
    __slots__ = ('name', 'stake', 'pot_chips', 'bet_chips', 'active',
                 'hole_cards', 'seat')

    def __init__(self, name: str, stake: int, seat: int,
                 hole_cards: Optional[List[str]] = None):
        self.name = name
        self.stake = stake          # remaining chips (not yet wagered)
        self.pot_chips = 0          # chips wagered in previous streets
        self.bet_chips = 0          # chips wagered this round
        self.active = True          # False if folded
        self.hole_cards = hole_cards
        self.seat = seat


# ── Behavioral stats helpers ──────────────────────────────────────────────

# Number of behavioral features appended to observation
BEHAVIORAL_STATS_DIM = 8


def _bh_track_action(action, bh):
    """Update behavioral tracking state after a non-blind action.

    Tracks ALL players' actions. Hero-specific filtering happens at compute time.
    """
    atype = action.action_type
    is_agg = atype in ('bet', 'raise') or action.is_all_in
    name = action.player_name

    # Track current street actions (all players)
    bh['street_actions'].append((name, atype, action.is_all_in))

    # Per-player counters
    bh['player_total'][name] += 1
    if is_agg:
        bh['player_agg'][name] += 1
        # Betting-line: most recent aggressor across hand + per-street count
        bh['last_aggressor'] = name
        bh['street_agg_count'] = bh.get('street_agg_count', 0) + 1
    if atype == 'fold':
        bh['player_fold'][name] += 1

    # Sizing ratio for bets/raises (per-player)
    if atype in ('bet', 'raise') and action.amount_chips > 0:
        current_pot = bh['pot_committed'] + sum(bh['street_bets'].values())
        if current_pot > 0:
            ratio = action.amount_chips / current_pot
            bh['sizing_ratios'].append((name, ratio))

    # All-player raise count
    if atype == 'raise':
        bh['raise_count'] += 1

    # Update street bet tracking (for pot computation)
    if atype in ('bet', 'call'):
        bh['street_bets'][name] += action.amount_chips
    elif atype == 'raise':
        already = bh['street_bets'].get(name, 0)
        delta = action.amount_chips - already
        if delta > 0:
            bh['street_bets'][name] = action.amount_chips


def _bh_compute(bh, hero_name):
    """Compute 8-dim behavioral feature vector, excluding hero's own actions.

    Features:
      0. opp_aggression_ratio  — fraction of opp actions that are aggressive
      1. opp_fold_ratio        — fraction of opp actions that are folds
      2. avg_bet_sizing        — mean opp bet/raise size / pot (normalized)
      3. max_bet_sizing        — largest opp bet/raise size / pot (normalized)
      4. num_raises_norm       — total raise count / 6
      5. street_aggression     — fraction of current-street actions that are aggressive
      6. opp_check_ratio       — fraction of opp current-street actions that are checks
      7. pot_escalation        — total pot / avg starting stack
    """
    feats = np.zeros(BEHAVIORAL_STATS_DIM, dtype=np.float32)

    # Opponent totals (exclude hero)
    opp_total = sum(c for n, c in bh['player_total'].items() if n != hero_name)
    opp_agg = sum(c for n, c in bh['player_agg'].items() if n != hero_name)
    opp_fold = sum(c for n, c in bh['player_fold'].items() if n != hero_name)
    opp_sizing = [r for n, r in bh['sizing_ratios'] if n != hero_name]

    opp_denom = max(opp_total, 1)

    # 0: opp aggression ratio
    feats[0] = opp_agg / opp_denom

    # 1: opp fold ratio
    feats[1] = opp_fold / opp_denom

    # 2: avg bet sizing (normalized: /3 so 3x pot → 1.0)
    if opp_sizing:
        feats[2] = min(sum(opp_sizing) / len(opp_sizing) / 3.0, 1.0)

    # 3: max bet sizing
    if opp_sizing:
        feats[3] = min(max(opp_sizing) / 3.0, 1.0)

    # 4: num raises normalized (/6)
    feats[4] = min(bh['raise_count'] / 6.0, 1.0)

    # 5: current street aggression (all players)
    sa = bh['street_actions']
    if sa:
        agg_count = sum(1 for _, a, ai in sa if a in ('bet', 'raise') or ai)
        feats[5] = agg_count / len(sa)

    # 6: opp check ratio on current street
    opp_street = [(n, a) for n, a, _ in sa if n != hero_name]
    if opp_street:
        check_count = sum(1 for _, a in opp_street if a == 'check')
        feats[6] = check_count / len(opp_street)

    # 7: pot escalation (total pot / avg starting stack)
    total_pot = bh['pot_committed'] + sum(bh['street_bets'].values())
    feats[7] = min(total_pot / bh['avg_stack'], 1.0)

    return feats


# Number of betting-line features appended to observation
BETTING_LINE_DIM = 3


def _bl_compute(bh, hero_name):
    """Compute 3-dim betting-line features.

    0. pot_at_street_start_norm — pot at start of current street / avg_stack
    1. hero_has_initiative — 1.0 if hero made the most recent aggressive
       action across the hand
    2. agg_count_this_street_norm — aggressive actions this street / 4
    """
    feats = np.zeros(BETTING_LINE_DIM, dtype=np.float32)
    feats[0] = float(np.clip(
        bh.get('street_start_pot', 0) / max(bh.get('avg_stack', 1), 1),
        0.0, 1.0))
    feats[1] = 1.0 if bh.get('last_aggressor') == hero_name else 0.0
    feats[2] = float(np.clip(
        bh.get('street_agg_count', 0) / 4.0, 0.0, 1.0))
    return feats


class ObservationBuilder:
    """Build observation vectors from parsed hand histories.

    Produces the same 41-dim (for 6-player) feature vector as
    FastEnhancedPokerEngine.
    """

    def __init__(self, num_players: int = 6, legacy_game_state: bool = False):
        self.num_players = num_players
        self.legacy_game_state = legacy_game_state

    def build_decision_points(self, hand: ParsedHand,
                              allow_unknown_cards: bool = False,
                              source_file: str = '',
                              track_action_history: bool = False,
                              behavioral_stats: bool = False,
                              threat_range_equity: bool = False) -> List[DecisionPoint]:
        """Replay a hand and build DecisionPoint at each non-blind decision.

        Args:
            allow_unknown_cards: If True, build observations even for players
                without known hole cards (using neutral card features).
                Useful for supervised learning to include fold examples.
        """
        active_players = hand.active_players
        if len(active_players) != self.num_players:
            return []

        # Build player index by name (seat order among active players)
        name_to_idx = {}
        players: List[_PlayerState] = []
        for i, pp in enumerate(active_players):
            name_to_idx[pp.name] = i
            players.append(_PlayerState(
                name=pp.name,
                stake=pp.stack_chips,
                seat=pp.seat,
                hole_cards=pp.hole_cards
            ))

        # Find button index
        button_idx = self._find_button_idx(active_players, hand.button_seat)

        # Community cards available at each street
        cc = hand.community_cards_at_street
        current_street = 'preflop'
        community_cards: List[str] = []

        decision_points: List[DecisionPoint] = []
        player_action_count = defaultdict(int)  # name -> count of non-blind actions

        # Behavioral stats tracking (replaces range estimator)
        bh = None
        if behavioral_stats:
            avg_stack = max(1, sum(p.stake + p.bet_chips for p in players) / len(players))
            bh = {
                'pot_committed': 0,
                'street_bets': defaultdict(int),
                'player_total': defaultdict(int),
                'player_agg': defaultdict(int),
                'player_fold': defaultdict(int),
                'sizing_ratios': [],     # (player_name, ratio) tuples
                'raise_count': 0,
                'street_actions': [],    # (name, action_type, is_all_in)
                'avg_stack': avg_stack,
                # Betting-line tracking (3 features)
                'last_aggressor': None,    # name of most recent aggressor across hand
                'street_agg_count': 0,     # aggressive actions on current street
                'street_start_pot': 0,     # pot at start of current street
            }

        # Legacy action history tracking for range estimator
        encoded_actions = [] if track_action_history else None
        ah_pot_chips = 0
        ah_player_bets = defaultdict(int)
        ah_player_spent = defaultdict(int)
        ah_start_stacks = {}
        if track_action_history:
            for p in players:
                ah_start_stacks[p.name] = p.stake

        for action in hand.actions:
            if action.player_name not in name_to_idx:
                continue

            pidx = name_to_idx[action.player_name]
            player = players[pidx]

            # Handle street transitions
            if action.street != current_street:
                # Move bet_chips -> pot_chips for all players
                for p in players:
                    p.pot_chips += p.bet_chips
                    p.bet_chips = 0
                current_street = action.street
                community_cards = cc.get(current_street, [])

                if bh is not None:
                    bh['pot_committed'] += sum(bh['street_bets'].values())
                    bh['street_bets'].clear()
                    bh['street_actions'] = []
                    # Betting-line: snapshot pot at street start, reset per-street counter
                    bh['street_start_pot'] = sum(p.pot_chips for p in players)
                    bh['street_agg_count'] = 0

                if track_action_history:
                    ah_pot_chips += sum(ah_player_bets.values())
                    ah_player_bets.clear()

            # Skip blind actions (not decisions)
            if action.is_blind:
                self._apply_blind(player, action)
                if bh is not None:
                    bh['street_bets'][action.player_name] += action.amount_chips
                if track_action_history:
                    ah_player_bets[action.player_name] += action.amount_chips
                    ah_player_spent[action.player_name] += action.amount_chips
                continue

            # Skip actions from folded/all-in players (no DP created)
            # But still track for behavioral stats and action history
            if not player.active or player.stake == 0:
                if bh is not None:
                    _bh_track_action(action, bh)
                if track_action_history:
                    self._encode_ah_action(
                        action, pidx, button_idx, current_street,
                        ah_pot_chips, ah_player_bets, ah_player_spent,
                        ah_start_stacks, avg_stack, encoded_actions,
                        community_cards)
                continue

            # ── Build observation and record decision point ───────────
            pot = self._compute_pot(players)
            max_bet = max(p.bet_chips for p in players)

            legal = get_legal_actions(
                max_bet=max_bet,
                player_bet_chips=player.bet_chips,
                player_stake=player.stake,
                pot=pot,
            )

            mapped_action = map_action(
                action=action,
                pot=pot,
                max_bet=max_bet,
                player_bet_chips=player.bet_chips,
                player_stake=player.stake,
            )

            obs = self._build_observation(
                players=players,
                player_idx=pidx,
                button_idx=button_idx,
                street=current_street,
                community_cards=community_cards,
                allow_unknown_cards=allow_unknown_cards,
            )

            has_cards = bool(player.hole_cards)
            if obs is not None and (has_cards or allow_unknown_cards):
                rel_pos = (pidx - button_idx) % self.num_players
                pos_label = _POS_LABELS_6.get(rel_pos, 'MP')

                # Pot in dollars (chips / 40.0 for NL5)
                pot_dollars = pot / 40.0

                # Append behavioral stats + betting-line features to observation
                if bh is not None:
                    bh_features = _bh_compute(bh, player.name)
                    bl_features = _bl_compute(bh, player.name)
                    obs = np.concatenate([obs, bh_features, bl_features])

                # threat_range_equity now merged into obs[18] (no separate dim)

                dp_kwargs = dict(
                    player_name=player.name,
                    observation=obs,
                    legal_actions=legal,
                    actual_action=mapped_action,
                    street=current_street,
                    hole_cards=player.hole_cards,
                    community_cards=list(community_cards),
                    position_label=pos_label,
                    player_idx=pidx,
                    hand_id=hand.hand_id,
                    action_index=player_action_count[player.name],
                    pot_dollars=pot_dollars,
                    source_file=source_file,
                )
                if track_action_history:
                    dp_kwargs['action_history'] = list(encoded_actions)

                decision_points.append(DecisionPoint(**dp_kwargs))
                player_action_count[player.name] += 1

            # ── Track behavioral stats (after DP, before state update) ──
            if bh is not None:
                _bh_track_action(action, bh)

            # ── Encode action for history (after creating DP, before applying) ──
            if track_action_history:
                self._encode_ah_action(
                    action, pidx, button_idx, current_street,
                    ah_pot_chips, ah_player_bets, ah_player_spent,
                    ah_start_stacks, avg_stack, encoded_actions)

            # ── Apply action to state ─────────────────────────────────
            self._apply_action(player, action, max_bet)

        return decision_points

    def _encode_ah_action(self, action, pidx, button_idx, current_street,
                          ah_pot_chips, ah_player_bets, ah_player_spent,
                          ah_start_stacks, avg_stack, encoded_actions,
                          community_cards=None):
        """Encode one action for the action history and update bet tracking."""
        from src.training.range_estimator import encode_action as _re_encode

        rel_pos = (pidx - button_idx) % self.num_players
        total_pot = ah_pot_chips + sum(ah_player_bets.values())

        if action.is_all_in:
            atype = 'all_in'
        elif action.action_type in ('bet', 'raise', 'call', 'check', 'fold'):
            atype = action.action_type
        else:
            atype = 'check'

        if action.action_type in ('bet', 'raise') and total_pot > 0:
            sizing = action.amount_chips / max(total_pot, 1)
        else:
            sizing = 0.0

        pot_ratio = total_pot / avg_stack

        actor_start = ah_start_stacks.get(action.player_name, avg_stack)
        actor_spent = ah_player_spent.get(action.player_name, 0)
        actor_remaining = max(0, actor_start - actor_spent)
        stack_ratio = actor_remaining / max(1, actor_start)

        encoded_actions.append(_re_encode(
            action_type=atype,
            rel_position=rel_pos,
            sizing=sizing,
            street=current_street,
            pot_ratio=pot_ratio,
            stack_ratio=stack_ratio,
            num_players=self.num_players,
            community_cards=community_cards,
        ))

        # Update bet tracking
        if action.action_type in ('bet', 'call'):
            ah_player_bets[action.player_name] += action.amount_chips
            ah_player_spent[action.player_name] += action.amount_chips
        elif action.action_type == 'raise':
            already = ah_player_bets.get(action.player_name, 0)
            delta = action.amount_chips - already
            if delta > 0:
                ah_player_spent[action.player_name] += delta
                ah_player_bets[action.player_name] = action.amount_chips

    def _find_button_idx(self, active_players, button_seat: int) -> int:
        """Find button index in the active player list."""
        for i, p in enumerate(active_players):
            if p.seat == button_seat:
                return i
        return 0

    def _compute_pot(self, players: List[_PlayerState]) -> int:
        return sum(p.pot_chips + p.bet_chips for p in players)

    def _apply_blind(self, player: _PlayerState, action: ParsedAction):
        """Apply a blind posting."""
        chips = action.amount_chips
        actual = min(chips, player.stake)
        player.bet_chips += actual
        player.stake -= actual

    def _apply_action(self, player: _PlayerState, action: ParsedAction,
                      max_bet: int):
        """Update player state after an action."""
        if action.action_type == 'fold':
            player.active = False
            return

        if action.action_type == 'check':
            return

        if action.action_type == 'call':
            call_cost = max_bet - player.bet_chips
            actual = min(call_cost, player.stake)
            player.bet_chips += actual
            player.stake -= actual
            if action.is_all_in:
                player.stake = 0
            return

        if action.action_type in ('bet', 'raise'):
            if action.action_type == 'bet':
                # Bet: new_bet_chips = amount
                new_total_bet = action.amount_chips
            else:
                # Raise "to X": new total bet_chips = X in chips
                new_total_bet = action.amount_chips

            cost = new_total_bet - player.bet_chips
            actual = min(cost, player.stake)
            player.bet_chips += actual
            player.stake -= actual
            if action.is_all_in:
                player.stake = 0
            return

    # ── Observation building ──────────────────────────────────────────────

    def _build_observation(self, players: List[_PlayerState],
                           player_idx: int, button_idx: int,
                           street: str,
                           community_cards: List[str],
                           allow_unknown_cards: bool = False) -> Optional[np.ndarray]:
        """Build the 41-dim observation vector.

        Mirrors FastEnhancedPokerEngine.get_observation() exactly.
        If allow_unknown_cards is True and hole cards are missing,
        uses neutral/average values for card-dependent features.
        """
        player = players[player_idx]
        if not player.hole_cards and not allow_unknown_cards:
            return None

        features = []
        features.append(self._position_features(player_idx, button_idx))
        features.append(self._pot_odds_features(players, player_idx))

        hole = player.hole_cards
        if hole:
            features.append(self._hand_strength_features(hole))
            features.append(self._stack_features(players, player_idx))
            n_opponents = max(sum(1 for p in players if p.active) - 1, 1)
            features.append(self._equity_features(hole, community_cards, n_opponents,
                                                    players=players))
        else:
            # Neutral card features for unknown hands
            features.append(np.array([0.5, 0.5, 0.5, 0.5, 0.2],
                                     dtype=np.float32))  # avg hand
            features.append(self._stack_features(players, player_idx))
            features.append(np.array([0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     dtype=np.float32))  # avg equity

        features.append(self._street_features(street))
        features.append(self._game_state_features(players, player_idx))
        features.append(self._betting_history_features(players, player_idx))

        return np.concatenate(features).astype(np.float32)

    def _position_features(self, player_idx: int, button_idx: int) -> np.ndarray:
        """8 dims: one-hot(6) + normalized_pos + is_button."""
        rel_pos = (player_idx - button_idx) % self.num_players
        one_hot = np.zeros(self.num_players, dtype=np.float32)
        one_hot[rel_pos] = 1.0
        norm_pos = rel_pos / self.num_players
        is_button = 1.0 if rel_pos == 0 else 0.0
        return np.array([*one_hot, norm_pos, is_button], dtype=np.float32)

    def _pot_odds_features(self, players: List[_PlayerState],
                           player_idx: int) -> np.ndarray:
        """3 dims: pot_normalized, stack_to_pot, pot_odds."""
        player = players[player_idx]
        pot = self._compute_pot(players)
        my_chips = player.stake
        total_chips = self.num_players * 100

        pot_normalized = float(np.clip(pot / max(total_chips, 1), 0.0, 1.0))
        raw_stp = my_chips / max(pot, 1)
        stack_to_pot = float(np.clip(
            math.log1p(raw_stp) / math.log1p(50), 0.0, 1.0))

        max_bet = max(p.bet_chips for p in players)
        my_in_chips = player.pot_chips + player.bet_chips
        call_amount = max(0, max_bet - my_in_chips)
        pot_odds = float(pot / max(pot + call_amount, 1)) if call_amount > 0 else 0.0

        return np.array([pot_normalized, stack_to_pot, pot_odds], dtype=np.float32)

    def _hand_strength_features(self, hole_cards: List[str]) -> np.ndarray:
        """5 dims: is_pair, is_suited, high_card_norm, low_card_norm, rank_gap_norm."""
        if len(hole_cards) < 2:
            return np.zeros(5, dtype=np.float32)

        r0 = _rank_val(hole_cards[0][0])
        r1 = _rank_val(hole_cards[1][0])
        s0 = hole_cards[0][1]
        s1 = hole_cards[1][1]

        is_pair = 1.0 if r0 == r1 else 0.0
        is_suited = 1.0 if s0 == s1 else 0.0
        hi, lo = max(r0, r1), min(r0, r1)
        high_card_norm = (hi - 2) / 12.0
        low_card_norm = (lo - 2) / 12.0
        rank_gap_norm = (hi - lo) / 12.0

        return np.array(
            [is_pair, is_suited, high_card_norm, low_card_norm, rank_gap_norm],
            dtype=np.float32)

    def _stack_features(self, players: List[_PlayerState],
                        player_idx: int) -> np.ndarray:
        """2 dims: m_ratio, relative_stack."""
        player = players[player_idx]
        my_stack = player.stake
        total_chips = sum(
            p.stake + p.pot_chips + p.bet_chips for p in players)
        pot = self._compute_pot(players)

        m_ratio = min(my_stack / max(pot + 2, 1), 5.0) / 5.0
        relative_stack = my_stack / max(total_chips, 1)

        return np.array([m_ratio, relative_stack], dtype=np.float32)

    def _equity_features(self, hole_cards: List[str],
                         community_cards: List[str],
                         n_opponents: int = 1,
                         players=None) -> np.ndarray:
        """9 dims: equity + board texture features.

        obs[0]=hand_strength is now tier-aware (was: equity vs uniform random).
        Requires `players` for per-street aggression signals; without it,
        falls back to equity-vs-random.
        """
        if len(hole_cards) < 2:
            return np.zeros(9, dtype=np.float32)

        r1 = _rank_val(hole_cards[0][0])
        r2 = _rank_val(hole_cards[1][0])
        hi, lo = max(r1, r2), min(r1, r2)

        # Tier-aware equity (replaces equity-vs-random). Postflop only;
        # preflop result is discarded by the early return below.
        if community_cards and _EVAL7_AVAILABLE and players is not None:
            hand_strength = self._compute_obs18_threat_equity(
                hole_cards, community_cards, players)
        elif _EVAL7_AVAILABLE:
            try:
                e7_hole = [_eval7.Card(c) for c in hole_cards[:2]]
                e7_board = [_eval7.Card(c) for c in community_cards] if community_cards else []
                if n_opponents <= 1:
                    n_samples = 1000 if not community_cards else 300
                    hand_strength = _eval7.py_hand_vs_range_monte_carlo(
                        e7_hole, _EVAL7_RANDOM_RANGE, e7_board, n_samples)
                else:
                    hand_strength = self._multiway_equity(
                        e7_hole, e7_board, n_opponents, 200)
            except Exception:
                hand_strength = self._approx_preflop_equity(hole_cards)
        else:
            hand_strength = self._approx_preflop_equity(hole_cards)

        if not community_cards:
            return np.zeros(9, dtype=np.float32)

        if not _TREYS_AVAILABLE:
            return np.array([hand_strength, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        # Postflop: treys for made hand category + texture features
        try:
            treys_hole = [TreysCard.new(c[0] + c[1].lower()) for c in hole_cards[:2]]
            treys_board = [TreysCard.new(c[0] + c[1].lower()) for c in community_cards]
        except Exception:
            return np.array([hand_strength, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        rank_int = _TREYS_EVALUATOR.evaluate(treys_board, treys_hole)
        rank_class = _TREYS_EVALUATOR.get_rank_class(rank_int)
        made_hand_cat = (9 - rank_class) / 8.0

        # Within-category rank: how good is this hand within its category?
        # 1.0 = best in category (e.g. nut flush), 0.0 = worst
        lower, upper = _TREYS_RANK_BOUNDS.get(rank_class, (1, 7462))
        if upper > lower:
            hand_rank_within_cat = float((upper - rank_int) / (upper - lower))
        else:
            hand_rank_within_cat = 1.0

        board_ranks = [_rank_val(c[0]) for c in community_cards]
        hole_ranks = [r1, r2]
        max_board = max(board_ranks)
        top_pair = 1.0 if max_board in hole_ranks else 0.0

        # Hero flush participation: how many hero cards match the board's
        # dominant suit? Replaces has_flush_draw which falsely triggered on
        # 4-flush boards where hero had no matching cards.
        board_suit_counts = {}
        for c in community_cards:
            s = c[1]
            board_suit_counts[s] = board_suit_counts.get(s, 0) + 1
        dominant_suit = max(board_suit_counts, key=board_suit_counts.get)
        max_board_suited = board_suit_counts[dominant_suit]
        if max_board_suited >= 2:
            hero_match = sum(1 for c in hole_cards[:2] if c[1] == dominant_suit)
            hero_flush_part = hero_match / 2.0
        else:
            hero_flush_part = 0.0

        # Hero straight participation: does hero's card contribute to a
        # straight? Replaces has_straight_draw which triggered for ALL hands
        # on 4-straight boards.
        board_ranks_set = set(board_ranks)
        if 14 in board_ranks_set:
            board_ranks_set.add(1)
        hero_ranks_set = set(hole_ranks)
        if 14 in hero_ranks_set:
            hero_ranks_set.add(1)
        hero_straight_part = 0.0
        for start in range(1, 11):
            window = set(range(start, start + 5))
            combined_in = len(window & (board_ranks_set | hero_ranks_set))
            hero_adds = combined_in - len(window & board_ranks_set)
            if combined_in >= 4 and hero_adds > 0:
                if combined_in == 5:
                    hero_straight_part = 1.0
                    break
                else:
                    hero_straight_part = max(hero_straight_part, 0.5)

        # Board pairing level: 0=unpaired, 0.33=single pair, 0.67=double
        # paired, 1.0=trips on board.
        from collections import Counter
        board_rank_counts = Counter(board_ranks)
        max_rank_count = max(board_rank_counts.values())
        num_pairs = sum(1 for c in board_rank_counts.values() if c >= 2)
        if max_rank_count >= 3:
            board_pairing_level = 1.0
        elif num_pairs >= 2:
            board_pairing_level = 0.67
        elif num_pairs == 1:
            board_pairing_level = 0.33
        else:
            board_pairing_level = 0.0

        max_bs = max(board_suit_counts.values()) if board_suit_counts else 1
        board_flush_texture = float(np.clip((max_bs - 1) / 3.0, 0.0, 1.0))

        board_ranks_unique = sorted(set(board_ranks))
        if 14 in board_ranks_unique:
            board_ranks_unique = sorted(set([1] + board_ranks_unique))
        consecutive = sum(
            1 for i in range(len(board_ranks_unique) - 1)
            if board_ranks_unique[i + 1] - board_ranks_unique[i] == 1)
        max_possible = max(len(board_ranks_unique) - 1, 1)
        board_straight_texture = float(consecutive / max_possible)

        return np.array(
            [hand_strength, made_hand_cat, hero_flush_part, hero_straight_part,
             top_pair, board_pairing_level, board_flush_texture,
             board_straight_texture, hand_rank_within_cat],
            dtype=np.float32)

    def _compute_obs18_threat_equity(self, hole_cards: List[str],
                                       community_cards: List[str],
                                       players) -> float:
        """Tier-aware equity for obs[18] (replaces equity vs random).

        Reads per-street aggression from `players[i].bet_chips` and board
        texture from `community_cards`. Returns 0.0 on any error.
        """
        try:
            if len(hole_cards) < 2 or not community_cards:
                return 0.0
            from src.range_equity import (
                select_threat_tier, get_threat_range_equity)

            big_blind = 2  # HH builder uses raw chip values; SB=1, BB=2
            max_opp_bet = max((getattr(p, 'bet_chips', 0) for p in players),
                                default=0)
            pot_was_raised = 1.0 if max_opp_bet > big_blind else 0.0
            facing_3bet_plus = 1.0 if max_opp_bet > 6 * big_blind else 0.0

            board_ranks = [_rank_val(c[0]) for c in community_cards]
            board_suit_counts = {}
            for c in community_cards:
                s = c[1]
                board_suit_counts[s] = board_suit_counts.get(s, 0) + 1
            max_bs = max(board_suit_counts.values()) if board_suit_counts else 1
            board_flush_texture = float(np.clip((max_bs - 1) / 3.0, 0.0, 1.0))

            board_ranks_unique = sorted(set(board_ranks))
            if 14 in board_ranks_unique:
                board_ranks_unique = sorted(set([1] + board_ranks_unique))
            consecutive = sum(
                1 for i in range(len(board_ranks_unique) - 1)
                if board_ranks_unique[i + 1] - board_ranks_unique[i] == 1)
            max_possible = max(len(board_ranks_unique) - 1, 1)
            board_straight_texture = float(consecutive / max_possible)

            tier = select_threat_tier(
                num_raises_norm=0.0, max_bet_sizing=0.0,
                facing_3bet_plus=facing_3bet_plus,
                pot_was_raised=pot_was_raised,
                board_flush_texture=board_flush_texture,
                board_straight_texture=board_straight_texture)

            return get_threat_range_equity(
                hole_cards, community_cards, tier,
                spot_key=None, db_lookup=None, n_samples=300)
        except Exception:
            return 0.0

    @staticmethod
    def _multiway_equity(hole: list, board: list, n_opponents: int,
                         n_samples: int = 500) -> float:
        """Monte Carlo multiway equity: win rate vs N random opponents."""
        import random as _rng
        deck = [_eval7.Card(r + s) for r in '23456789TJQKA' for s in 'shdc']
        used = set(str(c) for c in hole + board)
        available = [c for c in deck if str(c) not in used]
        cards_needed = n_opponents * 2 + (5 - len(board))
        if len(available) < cards_needed:
            return 0.5
        wins = 0
        for _ in range(n_samples):
            _rng.shuffle(available)
            idx = 0
            full_board = list(board) + available[idx:idx + (5 - len(board))]
            idx += 5 - len(board)
            my_score = _eval7.evaluate(list(hole) + full_board)
            beat_all = True
            for _ in range(n_opponents):
                opp = [available[idx], available[idx + 1]]
                idx += 2
                if _eval7.evaluate(opp + full_board) > my_score:  # higher = better in eval7
                    beat_all = False
                    break
            if beat_all:
                wins += 1
        return wins / max(n_samples, 1)

    @staticmethod
    def _approx_preflop_equity(hole_cards: List[str]) -> float:
        """Fallback preflop equity approximation when eval7 is unavailable."""
        r1 = _rank_val(hole_cards[0][0])
        r2 = _rank_val(hole_cards[1][0])
        hi, lo = max(r1, r2), min(r1, r2)
        is_pair = (r1 == r2)
        is_suited = (hole_cards[0][1] == hole_cards[1][1])
        if is_pair:
            hs = 0.53 + (hi - 2) / 12.0 * 0.32
        else:
            gap_penalty = max(0, hi - lo - 1) * 0.01
            suited_bonus = 0.03 if is_suited else 0.0
            hs = (0.36 + (hi - 2) / 12.0 * 0.15
                  + (lo - 2) / 12.0 * 0.08
                  + suited_bonus - gap_penalty)
        return float(np.clip(hs, 0.0, 1.0))

    def _street_features(self, street: str) -> np.ndarray:
        """4 dims: one-hot [preflop, flop, turn, river]."""
        idx = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}.get(street, 0)
        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[idx] = 1.0
        return one_hot

    def _game_state_features(self, players: List[_PlayerState],
                             player_idx: int) -> np.ndarray:
        """6 dims: players_alive_norm, call_amount_norm, aggression_level,
           facing_bet, bet_to_pot, can_raise."""
        player = players[player_idx]

        # alive = active or all-in (stake==0 but not folded)
        alive = sum(1 for p in players if p.active or (p.stake == 0 and p.pot_chips + p.bet_chips > 0))
        players_alive_norm = alive / self.num_players

        pot = self._compute_pot(players)
        max_bet = max(p.bet_chips for p in players)
        my_in_chips = player.pot_chips + player.bet_chips
        call_amount = max(0, max_bet - my_in_chips)
        call_amount_norm = min(call_amount / max(pot, 1), 1.0)

        big_blind = 2
        ratio = max(max_bet, big_blind) / big_blind
        aggression_level = float(np.clip(
            math.log(ratio) / math.log(20), 0.0, 1.0))

        # New facing-bet features
        facing_bet = 1.0 if call_amount > 0 else 0.0
        bet_to_pot = float(np.clip(
            call_amount / max(pot - call_amount, 1), 0.0, 2.0)) / 2.0

        # Can we raise? Check if stack covers more than the call
        # min_bet approximation: use big_blind as min raise increment
        min_raise = big_blind
        can_raise = 1.0 if (player.stake > call_amount and
                            min_raise <= player.stake - call_amount) else 0.0

        if self.legacy_game_state:
            return np.array(
                [players_alive_norm, call_amount_norm, aggression_level],
                dtype=np.float32)
        return np.array(
            [players_alive_norm, call_amount_norm, aggression_level,
             facing_bet, bet_to_pot, can_raise],
            dtype=np.float32)

    def _betting_history_features(self, players: List[_PlayerState],
                                  player_idx: int) -> np.ndarray:
        """5 dims: my_pot_share, not_raise_fraction, facing_3bet_plus,
           pot_was_raised, min_opp_stack_norm."""
        player = players[player_idx]
        big_blind = 2
        max_bet = max(p.bet_chips for p in players)
        pot = max(self._compute_pot(players), 1)

        my_pot_share = float(np.clip(
            (player.pot_chips + player.bet_chips) / pot, 0.0, 1.0))

        # Approximate not_raise_fraction (matches FastEnhancedPokerEngine)
        active_count = max(sum(1 for p in players if p.active), 1)
        not_raise_fraction = float(np.clip(
            (active_count - 1) / max(self.num_players, 1), 0.0, 1.0))

        facing_3bet_plus = 1.0 if max_bet > 6 * big_blind else 0.0
        pot_was_raised = 1.0 if max_bet > big_blind else 0.0

        # Min opponent stack (active or all-in, exclude self)
        opp_stacks = [
            p.stake for i, p in enumerate(players)
            if i != player_idx and (p.active or p.stake == 0)
        ]
        min_opp_stack = min(opp_stacks) if opp_stacks else 0
        min_opp_stack_norm = float(np.clip(min_opp_stack / 100.0, 0.0, 1.0))

        return np.array(
            [my_pot_share, not_raise_fraction, facing_3bet_plus,
             pot_was_raised, min_opp_stack_norm],
            dtype=np.float32)
