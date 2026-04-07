"""
FastEnhancedPokerEngine — feature extraction on top of the fast pokers backend.

Produces the same feature vector as EnhancedPokerEngine so trained models
are interchangeable between engines.
"""

import math
import sys
from pathlib import Path

import numpy as np

from .fast_poker_engine import (
    FastPokerEngine,
    _pkrs_card_rank, _pkrs_card_suit,
    _PlayerStatus,
    STAGE_PREFLOP, STAGE_FLOP, STAGE_TURN, STAGE_RIVER,
)

try:
    from treys import Card as TreysCard, Evaluator as TreysEvaluator
    _TREYS_EVALUATOR = TreysEvaluator()
    _TREYS_AVAILABLE = True
except ImportError:
    _TREYS_AVAILABLE = False

try:
    import eval7 as _eval7
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


# ── GPU Equity Model (optional, replaces Monte Carlo) ─────────────────────────
_EQUITY_MODEL = None
_EQUITY_DEVICE = None

def load_equity_model(path: str, device: str = 'cpu'):
    """Load the trained equity prediction network for fast inference.
    Auto-detects v1 (simple) vs v2 (residual) architecture from checkpoint."""
    global _EQUITY_MODEL, _EQUITY_DEVICE
    import torch

    ckpt = torch.load(path, map_location=device)
    input_size = ckpt.get('input_size', 24)
    output_size = ckpt.get('output_size', 9)
    hidden = ckpt.get('hidden', [128, 128, 64])

    # Try loading via current EquityNetwork class first, then fallback
    import torch.nn as tnn
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
        from train_equity_model import EquityNetwork
        model = EquityNetwork(input_size=input_size, output_size=output_size, hidden=hidden)
        model.load_state_dict(ckpt['model_state_dict'])
    except (RuntimeError, Exception):
        # Fallback: build simple Sequential wrapped in 'net' attribute (v1 compat)
        class _SimpleEquity(tnn.Module):
            def __init__(self, in_sz, out_sz, hid):
                super().__init__()
                layers = []
                prev = in_sz
                for h in hid:
                    layers.extend([tnn.Linear(prev, h), tnn.ReLU(), tnn.Dropout(0.05)])
                    prev = h
                layers.append(tnn.Linear(prev, out_sz))
                self.net = tnn.Sequential(*layers)
            def forward(self, x):
                return self.net(x)
        model = _SimpleEquity(input_size, output_size, hidden)
        model.load_state_dict(ckpt['model_state_dict'])

    model.eval()
    model.to(device)
    _EQUITY_MODEL = model
    _EQUITY_DEVICE = device

_RANK_MAP_EQ = {r: i for i, r in enumerate('23456789TJQKA')}
_SUIT_MAP_EQ = {s: i for i, s in enumerate('shdc')}


def _encode_card_for_equity(rank_char, suit_char):
    """Encode one card as 6 features for the equity model."""
    r = _RANK_MAP_EQ.get(rank_char.upper(), 0)
    s = _SUIT_MAP_EQ.get(suit_char.lower(), 0)
    feat = np.zeros(6, dtype=np.float32)
    feat[0] = r / 12.0
    feat[1 + s] = 1.0
    feat[5] = 1.0
    return feat


def _equity_model_predict(hole_ranks, hole_suits, board_ranks, board_suits,
                           hole_cards_raw, board_cards_raw):
    """Use GPU equity model to predict 9-dim equity features.

    Returns numpy array of shape (9,) or None if model not loaded.
    """
    if _EQUITY_MODEL is None:
        return None

    import torch
    from collections import Counter

    # Detect input size from model's first layer
    first_param = next(_EQUITY_MODEL.parameters())
    input_size = first_param.shape[1] if first_param.dim() == 2 else 82

    features = np.zeros(input_size, dtype=np.float32)

    # Convert from _rank_to_value (2-14) to training scale (0-12)
    def _to_train_rank_v1(r):
        return r - 2

    if input_size <= 24:
        # v1 simple encoding: rank_norm + suit_norm per card + basic derived
        hr_t = [_to_train_rank_v1(r) for r in hole_ranks[:2]]
        br_t = [_to_train_rank_v1(r) for r in board_ranks[:5]]
        for i, (r, s) in enumerate(zip(hr_t, hole_suits[:2])):
            features[i*2] = r / 12.0
            features[i*2 + 1] = s / 3.0
        for i, (r, s) in enumerate(zip(br_t, board_suits[:5])):
            features[4 + i*2] = r / 12.0
            features[4 + i*2 + 1] = s / 3.0
        d = 14
        h_ranks = list(hr_t)
        h_suits = list(hole_suits[:2])
        features[d+0] = 1.0 if h_ranks[0] == h_ranks[1] else 0.0
        features[d+1] = 1.0 if h_suits[0] == h_suits[1] else 0.0
        features[d+2] = max(h_ranks) / 12.0
        features[d+3] = min(h_ranks) / 12.0
        features[d+4] = len(br_t) / 5.0
        if br_t:
            features[d+5] = max(br_t) / 12.0
            from collections import Counter as C2
            brc2 = C2(br_t)
            features[d+6] = 1.0 if max(brc2.values()) >= 2 else 0.0
            bsc2 = C2(board_suits)
            features[d+7] = max(bsc2.values()) / 5.0
            unique2 = sorted(set(br_t))
            mc2 = 1; c2 = 1
            for ii in range(1, len(unique2)):
                if unique2[ii] - unique2[ii-1] == 1: c2 += 1; mc2 = max(mc2, c2)
                else: c2 = 1
            features[d+8] = mc2 / 4.0
            features[d+9] = (max(br_t) - min(br_t)) / 12.0

        with torch.no_grad():
            inp = torch.from_numpy(features).unsqueeze(0)
            if _EQUITY_DEVICE and _EQUITY_DEVICE != 'cpu':
                inp = inp.to(_EQUITY_DEVICE)
            return _EQUITY_MODEL(inp).cpu().numpy()[0]

    # v2 rich encoding (82 dims)
    # Convert from _rank_to_value (2-14) to training scale (0-12)
    # Training used RANK_MAP: 2→0, 3→1, ..., A→12
    # _rank_to_value returns:  2→2, 3→3, ..., A→14
    def _to_train_rank(r):
        return r - 2  # 2→0, 3→1, ..., A→12

    hole_ranks_t = [_to_train_rank(r) for r in hole_ranks[:2]]
    board_ranks_t = [_to_train_rank(r) for r in board_ranks[:5]]

    for i, (r, s) in enumerate(zip(hole_ranks_t, hole_suits[:2])):
        features[i*6] = r / 12.0
        features[i*6 + 1 + s] = 1.0
        features[i*6 + 5] = 1.0
    for i, (r, s) in enumerate(zip(board_ranks_t, board_suits[:5])):
        features[(i+2)*6] = r / 12.0
        features[(i+2)*6 + 1 + s] = 1.0
        features[(i+2)*6 + 5] = 1.0

    d = 42
    h_ranks = list(hole_ranks_t)
    h_suits = list(hole_suits[:2])

    features[d+0] = 1.0 if h_ranks[0] == h_ranks[1] else 0.0
    features[d+1] = 1.0 if h_suits[0] == h_suits[1] else 0.0
    features[d+2] = max(h_ranks) / 12.0
    features[d+3] = min(h_ranks) / 12.0
    features[d+4] = abs(h_ranks[0] - h_ranks[1]) / 12.0
    features[d+5] = len(board_ranks_t) / 5.0

    if board_ranks_t:
        b_ranks = list(board_ranks_t)
        b_suits = list(board_suits)
        all_ranks = h_ranks + b_ranks
        all_suits = h_suits + b_suits

        features[d+6] = max(b_ranks) / 12.0
        features[d+7] = min(b_ranks) / 12.0
        features[d+8] = (max(b_ranks) - min(b_ranks)) / 12.0

        brc = Counter(b_ranks)
        features[d+9] = max(brc.values()) / 4.0
        features[d+10] = sum(1 for v in brc.values() if v >= 2) / 3.0

        bsc = Counter(b_suits)
        features[d+11] = max(bsc.values()) / 5.0
        dom_suit = bsc.most_common(1)[0][0]
        features[d+12] = sum(1 for s in h_suits if s == dom_suit) / 2.0

        asc = Counter(all_suits)
        features[d+13] = max(asc.values()) / 7.0
        features[d+14] = 1.0 if max(asc.values()) >= 4 else 0.0
        features[d+15] = 1.0 if max(asc.values()) >= 5 else 0.0

        def _max_consec(ranks):
            if not ranks: return 0
            mc, c = 1, 1
            for i in range(1, len(ranks)):
                if ranks[i] - ranks[i-1] == 1: c += 1; mc = max(mc, c)
                else: c = 1
            return mc

        unique_all = sorted(set(all_ranks))
        if 12 in unique_all: unique_all = sorted(set([-1] + unique_all))
        unique_board = sorted(set(b_ranks))
        if 12 in unique_board: unique_board = sorted(set([-1] + unique_board))

        features[d+16] = _max_consec(unique_all) / 5.0
        features[d+17] = _max_consec(unique_board) / 5.0

        board_set = set(b_ranks)
        if 12 in board_set: board_set.add(-1)
        hero_set = set(h_ranks)
        if 12 in hero_set: hero_set.add(-1)
        hero_str = 0.0
        for start in range(-1, 10):
            window = set(range(start, start + 5))
            combined = len(window & (board_set | hero_set))
            hero_adds = combined - len(window & board_set)
            if combined >= 4 and hero_adds > 0:
                hero_str = 1.0 if combined == 5 else 0.5
                if combined == 5: break
        features[d+18] = hero_str

        features[d+19] = 1.0 if max(b_ranks) in h_ranks else 0.0
        features[d+20] = 1.0 if (h_ranks[0] == h_ranks[1] and h_ranks[0] > max(b_ranks)) else 0.0
        features[d+21] = sum(1 for r in h_ranks if r in b_ranks) / 2.0
        features[d+22] = sum(1 for r in h_ranks if r > max(b_ranks)) / 2.0
        features[d+23] = 1.0 if len(set(b_suits)) == 1 else 0.0
        features[d+24] = len(board_ranks_t) / 5.0

        for si in range(4):
            hero_has = sum(1 for s in h_suits if s == si)
            board_has = bsc.get(si, 0)
            features[d+25+si] = (hero_has + board_has) / 7.0

        arc = Counter(all_ranks)
        features[d+29] = sum(1 for v in arc.values() if v >= 2) / 6.0
        features[d+30] = sum(1 for v in arc.values() if v >= 3) / 3.0
        features[d+31] = sum(1 for v in arc.values() if v >= 4) / 2.0

        features[d+32] = sum(1 for r in b_ranks if r <= 4) / max(len(b_ranks), 1)
        features[d+33] = sum(1 for r in b_ranks if 5 <= r <= 8) / max(len(b_ranks), 1)
        features[d+34] = sum(1 for r in b_ranks if r >= 9) / max(len(b_ranks), 1)

        hero_on_board = sum(1 for r in h_ranks if r in b_ranks)
        if hero_on_board == 1:
            kicker = [r for r in h_ranks if r not in b_ranks]
            features[d+35] = kicker[0] / 12.0 if kicker else 0.0
        else:
            features[d+35] = max(h_ranks) / 12.0

        features[d+36] = 1.0 if len(board_ranks) == 3 else 0.0
        features[d+37] = 1.0 if len(board_ranks) == 4 else 0.0
        features[d+38] = 1.0 if len(board_ranks) == 5 else 0.0
        features[d+39] = 1.0 if len(board_ranks) == 0 else 0.0

    with torch.no_grad():
        inp = torch.from_numpy(features).unsqueeze(0)
        if _EQUITY_DEVICE and _EQUITY_DEVICE != 'cpu':
            inp = inp.to(_EQUITY_DEVICE)
        pred = _EQUITY_MODEL(inp).cpu().numpy()[0]

    return pred


class FastEnhancedPokerEngine(FastPokerEngine):
    """
    Drop-in replacement for EnhancedPokerEngine using the fast pokers backend.
    Produces identical feature vectors for the same game state.
    """

    BEHAVIORAL_STATS_DIM = 8
    OPPONENT_PROFILE_DIM = 5

    def __init__(self, num_players: int = 6,
                 add_position_features: bool = True,
                 add_pot_odds_features: bool = True,
                 add_hand_strength_features: bool = True,
                 add_stack_features: bool = True,
                 add_equity_features: bool = True,
                 add_street_features: bool = True,
                 add_game_state_features: bool = True,
                 add_betting_history_features: bool = True,
                 legacy_game_state: bool = False,
                 behavioral_stats: bool = False,
                 opponent_profile: bool = False,
                 rake_pct: float = 0.0,
                 rake_cap: float = 0.0):
        self.add_position_features = add_position_features
        self.add_pot_odds_features = add_pot_odds_features
        self.add_hand_strength_features = add_hand_strength_features
        self.add_stack_features = add_stack_features
        self.add_equity_features = add_equity_features and _TREYS_AVAILABLE
        self.add_street_features = add_street_features
        self.add_game_state_features = add_game_state_features
        self.add_betting_history_features = add_betting_history_features
        self.legacy_game_state = legacy_game_state  # 3 dims instead of 6
        self.add_behavioral_stats = behavioral_stats
        self.add_opponent_profile = opponent_profile
        self._opponent_profile = None  # 5-dim array, set via set_opponent_profile()
        self.rake_pct = rake_pct  # e.g. 0.05 = 5%
        self.rake_cap = rake_cap  # max rake in chips (0 = no cap)
        self.use_equity_model = (_EQUITY_MODEL is not None)  # auto-detect

        self.num_players = num_players
        self._bh = None  # behavioral tracking state (must be before super().__init__)
        self.enhanced_obs_size = self._calculate_enhanced_size()
        super().__init__(num_players)

    def _calculate_enhanced_size(self) -> int:
        size = 0
        if self.add_position_features:
            size += self.num_players + 2
        if self.add_pot_odds_features:
            size += 3
        if self.add_hand_strength_features:
            size += 5
        if self.add_stack_features:
            size += 2
        if self.add_equity_features:
            size += 9
        if self.add_street_features:
            size += 4
        if self.add_game_state_features:
            size += 3 if self.legacy_game_state else 6
        if self.add_betting_history_features:
            size += 5
        if self.add_behavioral_stats:
            size += self.BEHAVIORAL_STATS_DIM
        if self.add_opponent_profile:
            size += self.OPPONENT_PROFILE_DIM
        return size

    def get_observation_space_size(self) -> int:
        return self.enhanced_obs_size

    def get_observation(self) -> np.ndarray:
        if self._done or not self.current_agent:
            return np.zeros(self.enhanced_obs_size, dtype=np.float32)

        features = []
        if self.add_position_features:
            features.append(self._get_position_features())
        if self.add_pot_odds_features:
            features.append(self._get_pot_odds_features())
        if self.add_hand_strength_features:
            features.append(self._get_hand_strength_features())
        if self.add_stack_features:
            features.append(self._get_stack_features())
        if self.add_equity_features:
            features.append(self._get_equity_features())
        if self.add_street_features:
            features.append(self._get_street_features())
        if self.add_game_state_features:
            features.append(self._get_game_state_features())
        if self.add_betting_history_features:
            features.append(self._get_betting_history_features())
        if self.add_behavioral_stats:
            features.append(self._get_behavioral_stats_features())
        if self.add_opponent_profile:
            features.append(self._get_opponent_profile_features())

        if features:
            return np.concatenate(features).astype(np.float32)
        return np.zeros(self.enhanced_obs_size, dtype=np.float32)

    # ── Opponent profile (cross-hand, set externally) ──────────────────────

    def set_opponent_profile(self, profile: np.ndarray):
        """Set cross-hand opponent profile (5 dims). Persists until changed."""
        self._opponent_profile = profile

    def _get_opponent_profile_features(self) -> np.ndarray:
        """Return opponent profile or zeros if not set."""
        if self._opponent_profile is not None:
            return self._opponent_profile.astype(np.float32)
        return np.zeros(self.OPPONENT_PROFILE_DIM, dtype=np.float32)

    # ── Helper: quick access to current state ────────────────────────────────

    def _cur_idx(self):
        return self.agents.index(self.current_agent)

    def _cur_ps(self):
        return self._state.players_state[self._cur_idx()]

    @staticmethod
    def _rank_to_value(rank_str: str) -> int:
        _map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
        }
        return _map.get(rank_str, 0)

    # ── Feature blocks (matching EnhancedPokerEngine exactly) ────────────────

    def _get_position_features(self) -> np.ndarray:
        try:
            agent_idx = self._cur_idx()
            dealer_idx = self._state.button
            rel_pos = (agent_idx - dealer_idx) % self.num_players
            one_hot = np.zeros(self.num_players, dtype=np.float32)
            one_hot[rel_pos] = 1.0
            norm_pos = rel_pos / self.num_players
            is_button = 1.0 if rel_pos == 0 else 0.0
            return np.array([*one_hot, norm_pos, is_button], dtype=np.float32)
        except Exception:
            return np.zeros(self.num_players + 2, dtype=np.float32)

    def _get_pot_odds_features(self) -> np.ndarray:
        try:
            ps = self._cur_ps()
            pot = self._state.pot
            my_chips = ps.stake
            total_chips = self.num_players * 100

            pot_normalized = float(np.clip(pot / max(total_chips, 1), 0.0, 1.0))
            raw_stp = my_chips / max(pot, 1)
            stack_to_pot = float(np.clip(
                math.log1p(raw_stp) / math.log1p(50), 0.0, 1.0))

            max_bet = max(p.bet_chips for p in self._state.players_state)
            # Call amount = difference in current round bets only
            call_amount = max(0, max_bet - ps.bet_chips)
            pot_odds = float(pot / max(pot + call_amount, 1)) if call_amount > 0 else 0.0

            return np.array([pot_normalized, stack_to_pot, pot_odds], dtype=np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)

    def _get_hand_strength_features(self) -> np.ndarray:
        try:
            ps = self._cur_ps()
            cards = ps.hand
            if len(cards) < 2:
                return np.zeros(5, dtype=np.float32)

            r0 = self._rank_to_value(_pkrs_card_rank(cards[0]))
            r1 = self._rank_to_value(_pkrs_card_rank(cards[1]))
            s0 = _pkrs_card_suit(cards[0])
            s1 = _pkrs_card_suit(cards[1])

            is_pair = 1.0 if r0 == r1 else 0.0
            is_suited = 1.0 if s0 == s1 else 0.0
            hi, lo = max(r0, r1), min(r0, r1)
            high_card_norm = (hi - 2) / 12.0
            low_card_norm = (lo - 2) / 12.0
            rank_gap_norm = (hi - lo) / 12.0

            return np.array(
                [is_pair, is_suited, high_card_norm, low_card_norm, rank_gap_norm],
                dtype=np.float32)
        except Exception:
            return np.zeros(5, dtype=np.float32)

    def _get_stack_features(self) -> np.ndarray:
        try:
            ps = self._cur_ps()
            my_stack = ps.stake
            total_chips = sum(
                p.stake + p.pot_chips + p.bet_chips
                for p in self._state.players_state)
            pot = self._state.pot

            m_ratio = min(my_stack / max(pot + 2, 1), 5.0) / 5.0
            relative_stack = my_stack / max(total_chips, 1)

            return np.array([m_ratio, relative_stack], dtype=np.float32)
        except Exception:
            return np.zeros(2, dtype=np.float32)

    def _get_equity_features(self) -> np.ndarray:
        try:
            ps = self._cur_ps()
            hole = ps.hand
            board_pkrs = self._state.public_cards

            if len(hole) < 2:
                return np.zeros(9, dtype=np.float32)

            r1 = self._rank_to_value(_pkrs_card_rank(hole[0]))
            r2 = self._rank_to_value(_pkrs_card_rank(hole[1]))
            hi, lo = max(r1, r2), min(r1, r2)

            # Try GPU equity model first (fast path)
            if self.use_equity_model:
                hole_ranks = [r1, r2]
                hole_suits = [_SUIT_MAP_EQ.get(_pkrs_card_suit(c).lower(), 0) for c in hole[:2]]
                board_ranks = [self._rank_to_value(_pkrs_card_rank(c)) for c in board_pkrs]
                board_suits = [_SUIT_MAP_EQ.get(_pkrs_card_suit(c).lower(), 0) for c in board_pkrs]
                pred = _equity_model_predict(hole_ranks, hole_suits,
                                              board_ranks, board_suits, hole, board_pkrs)
                if pred is not None:
                    return pred

            # Fallback: Monte Carlo equity vs random range (eval7)
            def _to_card_str(c):
                return _pkrs_card_rank(c) + _pkrs_card_suit(c).lower()

            if _EVAL7_AVAILABLE:
                try:
                    e7_hole = [_eval7.Card(_to_card_str(c)) for c in hole[:2]]
                    e7_board = [_eval7.Card(_to_card_str(c)) for c in board_pkrs] if board_pkrs else []
                    n_samples = 500 if not board_pkrs else 150
                    hand_strength = _eval7.py_hand_vs_range_monte_carlo(
                        e7_hole, _EVAL7_RANDOM_RANGE, e7_board, n_samples)
                except Exception:
                    hand_strength = self._approx_preflop_equity(hole)
            else:
                hand_strength = self._approx_preflop_equity(hole)

            if not board_pkrs:
                return np.zeros(9, dtype=np.float32)

            # Postflop: treys for made hand category + texture features
            def _to_treys(c):
                return TreysCard.new(
                    _pkrs_card_rank(c) + _pkrs_card_suit(c).lower())

            treys_hole = [_to_treys(c) for c in hole[:2]]
            treys_board = [_to_treys(c) for c in board_pkrs]

            rank_int = _TREYS_EVALUATOR.evaluate(treys_board, treys_hole)
            rank_class = _TREYS_EVALUATOR.get_rank_class(rank_int)
            made_hand_cat = (9 - rank_class) / 8.0

            # Within-category rank: how good within its hand category?
            lower, upper = _TREYS_RANK_BOUNDS.get(rank_class, (1, 7462))
            if upper > lower:
                hand_rank_within_cat = float((upper - rank_int) / (upper - lower))
            else:
                hand_rank_within_cat = 1.0

            # Top pair
            board_ranks = [self._rank_to_value(_pkrs_card_rank(c))
                           for c in board_pkrs]
            hole_ranks = [r1, r2]
            max_board = max(board_ranks)
            top_pair = 1.0 if max_board in hole_ranks else 0.0

            # Hero flush participation
            board_suit_counts = {}
            for c in board_pkrs:
                s = _pkrs_card_suit(c)
                board_suit_counts[s] = board_suit_counts.get(s, 0) + 1
            dominant_suit = max(board_suit_counts, key=board_suit_counts.get)
            max_board_suited = board_suit_counts[dominant_suit]
            hero_suits = [_pkrs_card_suit(c) for c in hole[:2]]
            if max_board_suited >= 2:
                hero_match = sum(1 for s in hero_suits if s == dominant_suit)
                hero_flush_part = hero_match / 2.0
            else:
                hero_flush_part = 0.0

            # Hero straight participation
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

            # Board pairing level
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
        except Exception:
            return np.zeros(9, dtype=np.float32)

    def _approx_preflop_equity(self, hole) -> float:
        """Fallback preflop equity when eval7 is unavailable."""
        r1 = self._rank_to_value(_pkrs_card_rank(hole[0]))
        r2 = self._rank_to_value(_pkrs_card_rank(hole[1]))
        hi, lo = max(r1, r2), min(r1, r2)
        is_pair = (r1 == r2)
        is_suited = (_pkrs_card_suit(hole[0]) == _pkrs_card_suit(hole[1]))
        if is_pair:
            hs = 0.53 + (hi - 2) / 12.0 * 0.32
        else:
            gap_penalty = max(0, hi - lo - 1) * 0.01
            suited_bonus = 0.03 if is_suited else 0.0
            hs = (0.36 + (hi - 2) / 12.0 * 0.15
                  + (lo - 2) / 12.0 * 0.08
                  + suited_bonus - gap_penalty)
        return float(np.clip(hs, 0.0, 1.0))

    def _get_street_features(self) -> np.ndarray:
        try:
            s = str(self._state.stage)
            if 'Preflop' in s:
                idx = 0
            elif 'Flop' in s:
                idx = 1
            elif 'Turn' in s:
                idx = 2
            elif 'River' in s:
                idx = 3
            else:
                idx = 0
            one_hot = np.zeros(4, dtype=np.float32)
            one_hot[idx] = 1.0
            return one_hot
        except Exception:
            return np.zeros(4, dtype=np.float32)

    def _get_game_state_features(self) -> np.ndarray:
        try:
            ps = self._cur_ps()
            players = self._state.players_state

            alive = sum(1 for p in players if p.active or p.stake == 0)
            players_alive_norm = alive / self.num_players

            pot = self._state.pot
            max_bet = max(p.bet_chips for p in players)
            # Call amount = difference in current round bets only
            call_amount = max(0, max_bet - ps.bet_chips)
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
            my_stake = ps.stake
            can_raise = 1.0 if (my_stake > call_amount and
                                self._state.min_bet <= my_stake - call_amount) else 0.0

            if self.legacy_game_state:
                return np.array(
                    [players_alive_norm, call_amount_norm, aggression_level],
                    dtype=np.float32)
            return np.array(
                [players_alive_norm, call_amount_norm, aggression_level,
                 facing_bet, bet_to_pot, can_raise],
                dtype=np.float32)
        except Exception:
            if self.legacy_game_state:
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _get_betting_history_features(self) -> np.ndarray:
        try:
            ps = self._cur_ps()
            agent_idx = self._cur_idx()
            players = self._state.players_state

            big_blind = 2
            max_bet = max(p.bet_chips for p in players)
            pot = max(self._state.pot, 1)

            my_pot_share = float(np.clip(
                (ps.pot_chips + ps.bet_chips) / pot, 0.0, 1.0))

            # Raise count proxy: how many players have put in more than
            # the minimum required bet this round (indicates raise/re-raise)
            big_blind_amount = 2
            raise_count = sum(
                1 for p in players
                if p.bet_chips > big_blind_amount and p.active
            )
            not_raise_fraction = float(np.clip(
                raise_count / max(self.num_players, 1), 0.0, 1.0))

            facing_3bet_plus = 1.0 if max_bet > 6 * big_blind else 0.0
            pot_was_raised = 1.0 if max_bet > big_blind else 0.0

            # Min opponent stack (include all-in players, exclude only folded)
            # Match old engine: p.status != PlayerStatus.FOLDED
            opp_stacks = [
                p.stake for i, p in enumerate(players)
                if i != agent_idx and (p.active or p.stake == 0)
            ]
            min_opp_stack = min(opp_stacks) if opp_stacks else 0
            min_opp_stack_norm = float(np.clip(min_opp_stack / 100.0, 0.0, 1.0))

            return np.array(
                [my_pot_share, not_raise_fraction, facing_3bet_plus,
                 pot_was_raised, min_opp_stack_norm],
                dtype=np.float32)
        except Exception:
            return np.zeros(5, dtype=np.float32)

    # ── Rake: deduct from winner's reward ────────────────────────────────────

    def _finish(self):
        """Mark hand as done, collect rewards, apply rake to winners."""
        super()._finish()
        if self.rake_pct > 0:
            # Calculate total pot from rewards (winner's gain = pot - their investment)
            total_pot = sum(max(0, r) for r in self._rewards.values())
            if total_pot > 0:
                rake = total_pot * self.rake_pct
                if self.rake_cap > 0:
                    rake = min(rake, self.rake_cap)
                # Deduct rake proportionally from winners
                for agent in self.agents:
                    if self._rewards[agent] > 0:
                        share = self._rewards[agent] / total_pot
                        self._rewards[agent] -= rake * share

    # ── Behavioral stats: within-hand action tracking ───────────────────────

    def reset(self):
        result = super().reset()
        if self.add_behavioral_stats:
            avg_stack = 100.0  # default starting stack
            self._bh = {
                'pot_committed': 0,
                'street_bets': {},           # player_idx → chips bet this street
                'player_agg': {},            # player_idx → aggressive action count
                'player_total': {},          # player_idx → total action count
                'player_fold': {},           # player_idx → fold count
                'sizing_ratios': [],         # (player_idx, ratio) tuples
                'raise_count': 0,
                'street_actions': [],        # (player_idx, action_type) tuples
                'avg_stack': avg_stack,
                'last_stage': 'Preflop',
            }
        return result

    def step(self, action: int):
        if self.add_behavioral_stats and self._bh is not None and not self._done:
            self._bh_track_action(action)
        return super().step(action)

    def _bh_track_action(self, action: int):
        """Track action for behavioral stats before it's applied."""
        bh = self._bh
        if bh is None:
            return

        # Detect street transition
        stage = str(self._state.stage)
        if stage != bh['last_stage']:
            bh['pot_committed'] += sum(bh['street_bets'].values())
            bh['street_bets'] = {}
            bh['street_actions'] = []
            bh['last_stage'] = stage

        pidx = self._state.current_player
        ps = self._state.players_state[pidx]

        # Classify action
        max_bet = max(p.bet_chips for p in self._state.players_state)
        call_amount = max(0, max_bet - ps.bet_chips)

        # action: 0=fold, 1=call/check, 2=raise_half, 3=raise_pot, 4=all_in
        is_aggressive = action >= 2  # raise/all-in
        is_fold = (action == 0)

        bh['player_total'][pidx] = bh['player_total'].get(pidx, 0) + 1
        if is_aggressive:
            bh['player_agg'][pidx] = bh['player_agg'].get(pidx, 0) + 1
            bh['raise_count'] += 1
        if is_fold:
            bh['player_fold'][pidx] = bh['player_fold'].get(pidx, 0) + 1

        # Track bet sizing for aggressive actions
        if is_aggressive:
            current_pot = bh['pot_committed'] + sum(bh['street_bets'].values())
            if current_pot > 0:
                # Estimate bet amount based on action type
                if action == 2:  # raise half
                    est_bet = current_pot * 0.5
                elif action == 3:  # raise pot
                    est_bet = current_pot
                elif action == 4:  # all-in
                    est_bet = ps.stake
                else:
                    est_bet = current_pot * 0.5
                ratio = est_bet / max(current_pot, 1)
                bh['sizing_ratios'].append((pidx, ratio))

        # Track street bets
        if action == 1:  # call
            bh['street_bets'][pidx] = bh['street_bets'].get(pidx, 0) + call_amount
        elif is_aggressive:
            # Approximate: raise adds to street bets
            bh['street_bets'][pidx] = max_bet + (max_bet if action == 3 else max_bet // 2)

        bh['street_actions'].append((pidx, 'agg' if is_aggressive else ('fold' if is_fold else 'passive')))

    def _get_behavioral_stats_features(self) -> np.ndarray:
        """Compute 8-dim behavioral stats from tracked within-hand actions.

        Mirrors src/agent/obs_builder.py behavioral stats:
          0. opp_aggression_ratio
          1. opp_fold_ratio
          2. avg_bet_sizing (normalized /3)
          3. max_bet_sizing (normalized /3)
          4. num_raises_norm (/6)
          5. street_aggression
          6. opp_check_ratio
          7. pot_escalation
        """
        feats = np.zeros(self.BEHAVIORAL_STATS_DIM, dtype=np.float32)
        bh = self._bh
        if bh is None:
            return feats

        try:
            hero_idx = self._cur_idx()

            # Opponent totals (exclude hero)
            opp_total = sum(c for p, c in bh['player_total'].items() if p != hero_idx)
            opp_agg = sum(c for p, c in bh['player_agg'].items() if p != hero_idx)
            opp_fold = sum(c for p, c in bh['player_fold'].items() if p != hero_idx)
            opp_sizing = [r for p, r in bh['sizing_ratios'] if p != hero_idx]

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
                agg_count = sum(1 for _, a in sa if a == 'agg')
                feats[5] = agg_count / len(sa)

            # 6: opp check ratio on current street
            opp_street = [(p, a) for p, a in sa if p != hero_idx]
            if opp_street:
                check_count = sum(1 for _, a in opp_street if a == 'passive')
                feats[6] = check_count / len(opp_street)

            # 7: pot escalation (total pot / avg starting stack)
            total_pot = bh['pot_committed'] + sum(bh['street_bets'].values())
            feats[7] = min(total_pot / max(bh['avg_stack'], 1), 1.0)

        except Exception:
            pass

        return feats
