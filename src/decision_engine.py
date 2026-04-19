"""
Unified Decision Engine — single source of truth for all poker decisions.

Every consumer (runtime agent, GUI, validation, analysis) uses this class.
No decision logic should exist outside this module.

Pipeline:
  1. Preflop range override (if enabled and street == preflop)
  2. Model routing (facing_bet / checked_to / 3bet)
  3. Strong-hand fold guard (cap fold on trips+)
  4. All-in quality filter (suppress weak all-ins)
  5. Re-raise classifier (learned raise decision)
  6. Style tuning (soft probability adjustments)
  7. Action sampling (argmax + raise randomization)

Usage:
    from src.decision_engine import DecisionEngine, DecisionConfig, SpotContext

    engine = DecisionEngine()
    engine.load_models(facing_bet='models/fb.pt', checked_to='models/ct.pt')

    ctx = SpotContext(obs=obs, legal_actions=[0,1,2], facing_bet=True, street='flop')
    action, strategy = engine.decide(ctx)
"""

import sys
import os
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger('decision_engine')


# ─────────────────────────────────────────────────────────────────────────────
# Feature index constants (shared across entire codebase)
# ─────────────────────────────────────────────────────────────────────────────

IDX_POT_NORM = 8       # obs[8]  = pot / (6 * 100)
IDX_POT_ODDS = 10      # obs[10] = pot / (pot + call)   — 0 when not facing bet
IDX_EQUITY = 18         # obs[18] = hand_strength (equity vs random)
IDX_MADE_HAND = 19      # obs[19] = made_hand_cat (0.0=highcard .. 0.875=quads)
IDX_FLUSH_DRAW = 20     # obs[20] = hero_flush_participation (0/0.5/1.0)
IDX_STRAIGHT_DRAW = 21  # obs[21] = hero_straight_participation (0/0.5/1.0)
IDX_TOP_PAIR = 22       # obs[22] = top_pair (0 or 1)
IDX_BOARD_PAIR = 23     # obs[23] = board_pairing_level (0/0.33/0.67/1.0)
IDX_BOARD_FLUSH = 24    # obs[24] = board_flush_texture
IDX_RANK_WITHIN_CAT = 26  # obs[26] = hand_rank_within_cat (0=worst..1=best in category)
IDX_STACK_TO_POT = 9    # obs[9]  = effective_stack / pot (SPR)
IDX_IS_RIVER = 30       # obs[30] = is_river flag

# Made hand categories
HAND_HIGH_CARD = 0.0
HAND_ONE_PAIR = 0.125
HAND_TWO_PAIR = 0.25
HAND_TRIPS = 0.375
HAND_STRAIGHT = 0.5
HAND_FLUSH = 0.625
HAND_FULL_HOUSE = 0.75
HAND_QUADS = 0.875

# Action indices
FOLD = 0
CALL = 1
RAISE_HALF = 2
RAISE_POT = 3
ALL_IN = 4
RAISE_MIN = 5

NUM_ACTIONS = 6  # model outputs 5 (0-4), action 5 only used by preflop ranges

ACTION_NAMES = ['Fold', 'Call', 'RaiseHalf', 'RaisePot', 'AllIn', 'RaiseMin']

# 3bet pot threshold
_3BET_POT_BB = 12


# ─────────────────────────────────────────────────────────────────────────────
# SpotContext — everything needed to make a decision
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpotContext:
    """Game context for a single decision point.

    Callers construct this from their own data source (GameState, hand history,
    engine observation, etc.) and pass it to DecisionEngine.
    """
    obs: np.ndarray
    legal_actions: List[int]
    facing_bet: bool
    street: str = 'postflop'  # 'preflop', 'flop', 'turn', 'river'

    # Preflop-only (optional, for range override)
    hole_cards: Optional[List[str]] = None
    rel_pos: int = 0
    num_limpers: int = 0
    num_callers: int = 0  # cold-callers of the open (excludes opener)
    effective_stack: float = 100.0
    num_players: int = 6
    pot_bb: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DecisionConfig — what to enable/disable
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DecisionConfig:
    """Configuration for the decision pipeline.

    All consumers use this. The agent uses the defaults (everything on).
    Validation/analysis can disable guards for raw model comparison.
    """
    # Guards
    strong_hand_guard: bool = True
    allin_filter: bool = True
    style_tuning: bool = True

    # Preflop
    use_preflop_ranges: bool = True

    # Re-raise classifier
    reraise_classifier_path: str = ''
    reraise_threshold: float = 0.25

    # Sampling
    raise_randomize: bool = True  # 80/20 raise size randomization

    # Model temperature (used when loading, not at inference)
    temperature: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Strong-hand fold guard
# ─────────────────────────────────────────────────────────────────────────────

# (min_made_hand_cat, max_fold_prob)
_FOLD_CAPS = [
    (HAND_FULL_HOUSE, 0.00),  # full house+ → never fold
    (HAND_FLUSH,      0.01),  # flush → 1% max fold
    (HAND_STRAIGHT,   0.02),  # straight → 2% max fold
    (HAND_TRIPS,      0.03),  # set/trips → 3% max fold
]


def _apply_strong_hand_guard(strategy: np.ndarray, obs: np.ndarray,
                              legal_actions: List[int]) -> np.ndarray:
    """Cap fold probability for strong made hands."""
    if FOLD not in legal_actions:
        return strategy
    made_hand = obs[IDX_MADE_HAND] if len(obs) > IDX_MADE_HAND else 0.0

    max_fold = 1.0
    for min_cat, cap in _FOLD_CAPS:
        if made_hand >= min_cat - 0.01:
            max_fold = cap
            break

    if max_fold >= 1.0 or strategy[FOLD] <= max_fold:
        return strategy

    excess = strategy[FOLD] - max_fold
    strategy = strategy.copy()
    strategy[FOLD] = max_fold

    non_fold = [a for a in legal_actions if a != FOLD]
    non_fold_sum = sum(strategy[a] for a in non_fold)
    if non_fold_sum > 0:
        for a in non_fold:
            strategy[a] += excess * (strategy[a] / non_fold_sum)
    elif non_fold:
        for a in non_fold:
            strategy[a] = excess / len(non_fold)
    return strategy


# ─────────────────────────────────────────────────────────────────────────────
# All-in quality filter
# ─────────────────────────────────────────────────────────────────────────────

def _apply_allin_filter(strategy: np.ndarray, obs: np.ndarray,
                         legal_actions: List[int]) -> np.ndarray:
    """Suppress all-in with weak hands (< two pair and no draw)."""
    if ALL_IN not in legal_actions or strategy[ALL_IN] < 0.01:
        return strategy
    if len(obs) <= IDX_STRAIGHT_DRAW:
        return strategy

    made_hand = float(obs[IDX_MADE_HAND])
    if made_hand >= HAND_TWO_PAIR - 0.01:
        return strategy  # two pair+ — allow
    has_flush_draw = float(obs[IDX_FLUSH_DRAW]) > 0.0
    has_straight_draw = float(obs[IDX_STRAIGHT_DRAW]) > 0.0
    if has_flush_draw or has_straight_draw:
        return strategy  # semi-bluff — allow

    # Weak hand: redistribute all-in to other actions
    strategy = strategy.copy()
    excess = strategy[ALL_IN]
    strategy[ALL_IN] = 0.0
    non_allin = [a for a in legal_actions if a != ALL_IN]
    non_allin_sum = sum(strategy[a] for a in non_allin)
    if non_allin_sum > 0:
        for a in non_allin:
            strategy[a] += excess * (strategy[a] / non_allin_sum)
    elif non_allin:
        for a in non_allin:
            strategy[a] = excess / len(non_allin)
    return strategy


def _promote_to_allin(strategy: np.ndarray, obs: np.ndarray,
                       legal_actions: List[int],
                       shift_frac: float = 0.5) -> np.ndarray:
    """Promote RaisePot → AllIn on river with near-nut hands.

    Deep CFR model has no all-in head (action_size=4). At deep SPR this leaves
    value behind when we hold a monster on the river. Shifts `shift_frac` of
    RaisePot mass to AllIn only when hand is near-nuts within category and
    board texture doesn't threaten us.
    """
    if ALL_IN not in legal_actions:
        return strategy
    if len(obs) <= IDX_IS_RIVER:
        return strategy
    if float(obs[IDX_IS_RIVER]) < 0.5:
        return strategy
    # Need meaningful SPR, otherwise RaisePot ≈ AllIn already
    if float(obs[IDX_STACK_TO_POT]) < 0.75:
        return strategy
    # Straight+ only
    made_hand = float(obs[IDX_MADE_HAND])
    if made_hand < HAND_STRAIGHT - 0.01:
        return strategy
    # Near-nuts within category
    if float(obs[IDX_RANK_WITHIN_CAT]) < 0.85:
        return strategy
    # Texture safety: flush possible and we don't have one → skip
    if float(obs[IDX_BOARD_FLUSH]) > 0.75 and made_hand < HAND_FLUSH - 0.01:
        return strategy
    # Board paired and we don't have FH+ → opponent can have boat
    if float(obs[IDX_BOARD_PAIR]) > 0.5 and made_hand < HAND_FULL_HOUSE - 0.01:
        return strategy
    rp_mass = float(strategy[RAISE_POT])
    if rp_mass < 0.1:
        return strategy
    strategy = strategy.copy()
    shift = rp_mass * shift_frac
    strategy[RAISE_POT] -= shift
    strategy[ALL_IN] += shift
    return strategy


# ─────────────────────────────────────────────────────────────────────────────
# Style tuning
# ─────────────────────────────────────────────────────────────────────────────

_STYLE = {
    # Raise sizing: upgrade Raise½ → RaisePot with strong hands
    'raise_size_upgrade': 0.35,
    'raise_size_min_hand': HAND_TWO_PAIR,

    # Facing bet: add some raise probability (model never re-raises)
    # NOTE: skipped when reraise classifier is loaded
    'facing_bet_raise_boost': 0.20,
    'facing_bet_raise_min_hand': HAND_TWO_PAIR,

    # Facing bet: fold boost — now handled by _facing_bet_fold_boost()
    # (old flat 10% boost for air only was too weak — WTSD 37-42%)

    # Checked-to: bet with strong hands (value bet) — scales by street
    # Flop: full boost (CBet), Turn: moderate, River: thin value only
    'value_bet_boost_flop': 0.12,
    'value_bet_boost_turn': 0.04,
    'value_bet_boost_river': 0.03,
    'value_bet_min_hand': HAND_ONE_PAIR,

    # Checked-to: add bluff frequency with air
    'bluff_boost': 0.12,
    'bluff_max_hand': 0.01,

    # Checked-to: add semi-bluff with draws
    'semi_bluff_boost': 0.12,
}


def _facing_bet_fold_boost(obs: np.ndarray, made_hand: float,
                           has_draw: bool, street: str) -> float:
    """Compute fold probability boost when facing a bet.

    Considers:
    - Hand strength (air folds most, bottom pair folds a lot, top pair less)
    - Bet size (bigger bet = fold more)
    - Street (later streets = fold more, draws lose equity)
    - Draws (still have equity on flop/turn, not river)
    """
    # Street multiplier: fold more on later streets
    street_mult = {'flop': 0.5, 'Flop': 0.5,
                   'turn': 0.8, 'Turn': 0.8,
                   'river': 1.0, 'River': 1.0}.get(street, 0.5)

    # Bet size: obs[35] = bet_to_pot = call / (pot - call), normalized /2
    # Higher = bigger bet = should fold more
    bet_size = float(obs[35]) if len(obs) > 35 else 0.3
    bet_mult = 0.5 + bet_size  # range ~0.5 (min bet) to ~1.5 (overbet)

    # Base fold boost by hand strength
    if made_hand < 0.01:
        # Pure air: fold a lot
        if has_draw and street in ('flop', 'Flop'):
            return 0.0  # don't fold draws on flop
        base = 0.25
    elif made_hand < 0.125:
        # High card (A-high, K-high): weak, fold most of the time
        if has_draw and street in ('flop', 'Flop', 'turn', 'Turn'):
            return 0.0  # draws with overcards have equity
        base = 0.20
    elif made_hand < 0.20:
        # Bottom/middle pair: fold facing big bets on later streets
        if street in ('flop', 'Flop'):
            return 0.0  # keep calling on flop (could improve)
        base = 0.15
    elif made_hand < 0.25:
        # Top pair: only fold facing big bets on river
        if street not in ('river', 'River'):
            return 0.0
        base = 0.08
    else:
        # Two pair+: don't boost folding (strong hand guard handles this)
        return 0.0

    return min(base * street_mult * bet_mult, 0.35)


def _apply_style_tuning(strategy: np.ndarray, obs: np.ndarray,
                         legal_actions: List[int], facing_bet: bool,
                         skip_facing_bet_raise: bool = False,
                         street: str = '') -> np.ndarray:
    """Soft post-inference adjustments to fix model tendencies."""
    if len(obs) <= IDX_MADE_HAND:
        return strategy

    made_hand = float(obs[IDX_MADE_HAND])
    has_flush_draw = float(obs[IDX_FLUSH_DRAW]) > 0.0 if len(obs) > IDX_FLUSH_DRAW else False
    has_straight_draw = float(obs[IDX_STRAIGHT_DRAW]) > 0.0 if len(obs) > IDX_STRAIGHT_DRAW else False
    has_draw = has_flush_draw or has_straight_draw

    strategy = strategy.copy()

    if facing_bet:
        # Boost re-raise with strong hands (skipped when reraise classifier is active)
        if not skip_facing_bet_raise and made_hand >= _STYLE['facing_bet_raise_min_hand']:
            boost = _STYLE['facing_bet_raise_boost']
            if RAISE_HALF in legal_actions and strategy[CALL] > boost * 2:
                strategy[RAISE_HALF] += boost * 0.6
                strategy[RAISE_POT] += boost * 0.3
                strategy[ALL_IN] += boost * 0.1
                strategy[CALL] -= boost

        # Fold boost: scale by hand strength, bet size, and street.
        # The model under-folds everything (trained on showdown-only data).
        if FOLD in legal_actions and strategy[CALL] > 0.05:
            boost = _facing_bet_fold_boost(obs, made_hand, has_draw, street)
            if boost > 0 and strategy[CALL] > boost:
                strategy[FOLD] += boost
                strategy[CALL] -= boost
    else:
        # Value bet boost: bet with one pair+ when checked to — scales by street
        # Flop: bet often (CBet), Turn: moderate, River: selective
        if street in ('flop', 'Flop'):
            vb_boost = _STYLE['value_bet_boost_flop']
        elif street in ('turn', 'Turn'):
            vb_boost = _STYLE['value_bet_boost_turn']
        else:
            vb_boost = _STYLE['value_bet_boost_river']

        if (made_hand >= _STYLE['value_bet_min_hand']
                and RAISE_HALF in legal_actions
                and strategy[CALL] > vb_boost * 1.5):
            strategy[RAISE_HALF] += vb_boost * 0.7
            strategy[RAISE_POT] += vb_boost * 0.3 if RAISE_POT in legal_actions else 0
            strategy[CALL] -= vb_boost

        # Raise sizing upgrade: Raise½ → RaisePot with strong hands
        if (made_hand >= _STYLE['raise_size_min_hand']
                and RAISE_POT in legal_actions and strategy[RAISE_HALF] > 0.1):
            upgrade = strategy[RAISE_HALF] * _STYLE['raise_size_upgrade']
            strategy[RAISE_HALF] -= upgrade
            strategy[RAISE_POT] += upgrade

        # Bluff boost: bet with air when checked to (flop only — model
        # already bluffs 40-50% on turn/river, boosting makes it too high)
        if (street in ('flop', 'Flop')
                and made_hand < _STYLE['bluff_max_hand']
                and not has_draw and RAISE_HALF in legal_actions
                and strategy[CALL] > _STYLE['bluff_boost'] * 2):
            strategy[RAISE_HALF] += _STYLE['bluff_boost']
            strategy[CALL] -= _STYLE['bluff_boost']

        # Semi-bluff boost: bet with draws when checked to
        if (has_draw and made_hand < HAND_ONE_PAIR
                and RAISE_HALF in legal_actions
                and strategy[CALL] > _STYLE['semi_bluff_boost'] * 2):
            boost = _STYLE['semi_bluff_boost']
            strategy[RAISE_HALF] += boost
            strategy[CALL] -= boost

    # Renormalize
    total = sum(strategy[a] for a in legal_actions)
    if total > 0:
        for a in range(len(strategy)):
            if a in legal_actions:
                strategy[a] /= total
            else:
                strategy[a] = 0.0

    return strategy


# ─────────────────────────────────────────────────────────────────────────────
# DecisionEngine
# ─────────────────────────────────────────────────────────────────────────────

class DecisionEngine:
    """Single source of truth for all poker decisions.

    Handles model routing, guards, style tuning, preflop ranges, and sampling.
    Every consumer (agent, validation, analysis, GUI) uses this class.
    """

    def __init__(self, config: DecisionConfig = None):
        self.config = config or DecisionConfig()

        self._model_fb = None       # facing_bet model
        self._model_ct = None       # checked_to model
        self._model_fb_3bet = None  # facing_bet 3bet pot model
        self._model_single = None   # fallback single model

        self._obs_size_fb = 0
        self._obs_size_ct = 0
        self._obs_size_fb_3bet = 0

        self._reraise_clf = None
        self._reraise_obs_size = 0

        self._preflop_ranges: Dict[int, object] = {}

    # ── Model loading ────────────────────────────────────────────────────

    def load_models(self, facing_bet: str = '', checked_to: str = '',
                    facing_bet_3bet: str = '', single: str = '',
                    reraise: str = '') -> None:
        """Load models from checkpoint paths. Empty string = skip."""
        if facing_bet:
            self._model_fb, self._obs_size_fb = self._load_one(facing_bet)
            logger.info(f"Loaded facing_bet: {facing_bet} (obs={self._obs_size_fb})")
        if checked_to:
            self._model_ct, self._obs_size_ct = self._load_one(
                checked_to, exclude_fold=True)
            logger.info(f"Loaded checked_to: {checked_to} (obs={self._obs_size_ct})")
        if facing_bet_3bet:
            self._model_fb_3bet, self._obs_size_fb_3bet = self._load_one(facing_bet_3bet)
            logger.info(f"Loaded 3bet facing_bet: {facing_bet_3bet}")
        if single:
            self._model_single, _ = self._load_one(single)
            logger.info(f"Loaded single model: {single}")
        if reraise or self.config.reraise_classifier_path:
            rr_path = reraise or self.config.reraise_classifier_path
            if rr_path:
                self._load_reraise(rr_path)

    def _load_one(self, path: str, exclude_fold: bool = False):
        """Load a CQL/Supervised/DeepCFR checkpoint. Returns (adapter, obs_size)."""
        from src.models import load_model
        adapter, obs_size = load_model(path, exclude_fold=exclude_fold)
        return adapter, obs_size

    def _load_reraise(self, path: str) -> None:
        """Load the binary re-raise classifier."""
        import torch
        from scripts.train_reraise import ReRaiseClassifier

        checkpoint = torch.load(path, map_location='cpu')
        obs_size = checkpoint['obs_size']
        hidden = checkpoint.get('hidden_layers', [256, 128, 64])
        dropout = checkpoint.get('dropout', 0.15)

        clf = ReRaiseClassifier(obs_size, hidden, dropout)
        clf.load_state_dict(checkpoint['model_state_dict'])
        clf.eval()

        self._reraise_clf = clf
        self._reraise_obs_size = obs_size
        logger.info(f"Reraise classifier loaded: {path} "
                    f"(threshold={self.config.reraise_threshold})")

    def load_preflop_ranges(self, num_players_list: List[int] = None) -> None:
        """Load GTO preflop ranges."""
        if not self.config.use_preflop_ranges:
            return
        try:
            from src.preflop_ranges import PreflopRanges
            for np_ in (num_players_list or [2, 3, 4, 5, 6]):
                try:
                    self._preflop_ranges[np_] = PreflopRanges(num_players=np_)
                except Exception:
                    pass
            logger.info(f"Preflop ranges loaded: {list(self._preflop_ranges.keys())}")
        except Exception as e:
            logger.warning(f"Could not load preflop ranges: {e}")

    # ── Core decision pipeline ───────────────────────────────────────────

    def get_strategy(self, ctx: SpotContext) -> np.ndarray:
        """Get strategy array — full pipeline, no sampling.

        This is the main entry point for all consumers that need raw strategy.
        """
        legal = ctx.legal_actions
        strategy = np.zeros(NUM_ACTIONS, dtype=np.float32)

        # 1. Preflop range override
        if (ctx.street == 'preflop' and self._preflop_ranges
                and self.config.use_preflop_ranges and ctx.hole_cards):
            pf_strategy = self._preflop_strategy(ctx)
            if pf_strategy is not None:
                return pf_strategy

        # 2. Model routing + inference
        model, target_obs_size = self._route_model(ctx)
        if model is None:
            # Fallback: check if free, fold if facing bet
            if not ctx.facing_bet and CALL in legal:
                strategy[CALL] = 1.0
            elif FOLD in legal:
                strategy[FOLD] = 1.0
            elif legal:
                strategy[legal[0]] = 1.0
            return strategy

        # Pad observation to model's expected size
        obs = _pad_obs(ctx.obs, target_obs_size)

        # Model legal actions: exclude fold for checked_to, exclude action 5
        if not ctx.facing_bet and self._model_ct is not None:
            model_legal = [a for a in legal if a != FOLD and a < 5] or [CALL]
        else:
            model_legal = [a for a in legal if a < 5] or [CALL]

        raw = model.get_strategy(obs, 0, model_legal)
        if isinstance(raw, dict):
            for a, p in raw.items():
                if a < NUM_ACTIONS:
                    strategy[a] = p
        elif isinstance(raw, (list, np.ndarray)):
            for i, p in enumerate(raw):
                if i < NUM_ACTIONS:
                    strategy[i] = float(p)

        # 3. Strong hand guard
        if self.config.strong_hand_guard:
            strategy = _apply_strong_hand_guard(strategy, ctx.obs, legal)

        # 4. All-in quality filter
        if self.config.allin_filter:
            strategy = _apply_allin_filter(strategy, ctx.obs, legal)

        # 5. Re-raise classifier
        skip_fb_raise = False
        if ctx.facing_bet and self._reraise_clf is not None:
            strategy = self._apply_reraise(strategy, ctx.obs, legal)
            skip_fb_raise = True

        # 6. Style tuning
        if self.config.style_tuning:
            strategy = _apply_style_tuning(strategy, ctx.obs, legal,
                                            ctx.facing_bet, skip_fb_raise,
                                            street=ctx.street)

        return strategy

    def decide(self, ctx: SpotContext) -> Tuple[int, List[float]]:
        """Get strategy + sample action. Full pipeline.

        Returns (action_index, strategy_list_of_6).
        """
        strategy = self.get_strategy(ctx)
        strategy_list = [float(strategy[i]) for i in range(NUM_ACTIONS)]
        action = self._sample_action(strategy_list, ctx.legal_actions)
        return action, strategy_list

    # ── Model routing ────────────────────────────────────────────────────

    def _route_model(self, ctx: SpotContext):
        """Pick the right model based on game context.

        Returns (model, target_obs_size).
        """
        is_3bet = ctx.pot_bb > _3BET_POT_BB if ctx.pot_bb > 0 else (
            self._is_3bet_from_obs(ctx.obs))

        if ctx.facing_bet:
            if is_3bet and self._model_fb_3bet:
                return self._model_fb_3bet, self._obs_size_fb_3bet
            if self._model_fb:
                return self._model_fb, self._obs_size_fb
        else:
            if self._model_ct:
                return self._model_ct, self._obs_size_ct

        # Fallback to single model
        if self._model_single:
            return self._model_single, getattr(self._model_single, 'obs_size', 50)

        # Try any loaded model
        for m, sz in [(self._model_fb, self._obs_size_fb),
                      (self._model_ct, self._obs_size_ct)]:
            if m:
                return m, sz

        return None, 0

    def _is_3bet_from_obs(self, obs: np.ndarray) -> bool:
        """Detect 3bet pot from observation."""
        if len(obs) > IDX_POT_NORM:
            pot_bb = obs[IDX_POT_NORM] * 600
            return pot_bb > _3BET_POT_BB
        return False

    # ── Preflop ranges ───────────────────────────────────────────────────

    def _preflop_strategy(self, ctx: SpotContext) -> Optional[np.ndarray]:
        """Get preflop strategy from GTO ranges."""
        try:
            pf = self._preflop_ranges.get(ctx.num_players)
            if pf is None:
                pf = self._preflop_ranges.get(6)
            if pf is None:
                return None

            pot_bb = ctx.pot_bb
            if pot_bb <= 0 and ctx.facing_bet:
                pot_bb = 3.5  # reasonable default

            strategy = pf.get_strategy(
                hole_cards=ctx.hole_cards, rel_pos=ctx.rel_pos,
                pot_bb=pot_bb, legal_actions=ctx.legal_actions,
                num_limpers=ctx.num_limpers, num_callers=ctx.num_callers,
                effective_stack=ctx.effective_stack)

            if strategy is not None:
                result = np.zeros(NUM_ACTIONS, dtype=np.float32)
                for i in range(min(len(strategy), NUM_ACTIONS)):
                    result[i] = float(strategy[i])
                return result
        except Exception as e:
            logger.warning(f"Preflop range failed: {e}")
        return None

    # ── Re-raise classifier ──────────────────────────────────────────────

    def _apply_reraise(self, strategy: np.ndarray, obs: np.ndarray,
                        legal_actions: List[int]) -> np.ndarray:
        """Use reraise classifier to upgrade call → raise."""
        import torch

        best = max(legal_actions, key=lambda a: strategy[a])
        if best != CALL:
            return strategy
        raise_actions = [a for a in legal_actions if a >= RAISE_HALF]
        if not raise_actions:
            return strategy

        # Pad/truncate obs for classifier
        clf_obs = obs
        if len(clf_obs) < self._reraise_obs_size:
            padded = np.zeros(self._reraise_obs_size, dtype=np.float32)
            padded[:len(clf_obs)] = clf_obs
            clf_obs = padded
        elif len(clf_obs) > self._reraise_obs_size:
            clf_obs = clf_obs[:self._reraise_obs_size]

        with torch.no_grad():
            logit = self._reraise_clf(
                torch.FloatTensor(clf_obs).unsqueeze(0)).item()
            p_raise = 1.0 / (1.0 + np.exp(-logit))

        if p_raise < self.config.reraise_threshold:
            return strategy

        move = min(p_raise, 0.6) * strategy[CALL]
        if move < 0.01:
            return strategy

        strategy = strategy.copy()
        strategy[CALL] -= move

        # Distribute among raise sizes by hand strength
        made_hand = float(obs[IDX_MADE_HAND]) if len(obs) > IDX_MADE_HAND else 0
        if made_hand >= HAND_TRIPS:
            r_weights = {RAISE_HALF: 0.2, RAISE_POT: 0.5, ALL_IN: 0.3}
        elif made_hand >= HAND_TWO_PAIR:
            r_weights = {RAISE_HALF: 0.3, RAISE_POT: 0.5, ALL_IN: 0.2}
        else:
            r_weights = {RAISE_HALF: 0.6, RAISE_POT: 0.3, ALL_IN: 0.1}

        total_w = sum(r_weights.get(a, 0) for a in raise_actions)
        if total_w > 0:
            for a in raise_actions:
                strategy[a] += move * r_weights.get(a, 0) / total_w

        logger.debug(f"Reraise: P={p_raise:.2f}, moved {move:.2f} to raises")
        return strategy

    # ── Action sampling ──────────────────────────────────────────────────

    def _sample_action(self, strategy: List[float],
                        legal_actions: List[int]) -> int:
        """Argmax with optional 80/20 raise randomization."""
        best_action = legal_actions[0]
        best_prob = -1.0
        for a in legal_actions:
            if strategy[a] > best_prob:
                best_prob = strategy[a]
                best_action = a

        # 80% keep model's raise, 20% randomize among other raise sizes
        if self.config.raise_randomize and best_action >= RAISE_HALF:
            raise_actions = [a for a in legal_actions
                             if a >= RAISE_HALF and strategy[a] > 0.01]
            if len(raise_actions) > 1 and random.random() > 0.80:
                others = [a for a in raise_actions if a != best_action]
                if others:
                    probs = [strategy[a] for a in others]
                    total = sum(probs)
                    if total > 0:
                        r = random.random() * total
                        cum = 0.0
                        for a, p in zip(others, probs):
                            cum += p
                            if r <= cum:
                                best_action = a
                                break

        return best_action

    # ── Info ─────────────────────────────────────────────────────────────

    @property
    def has_models(self) -> bool:
        return any(m is not None for m in
                   [self._model_fb, self._model_ct, self._model_single])

    def info(self) -> str:
        parts = []
        if self._model_fb:
            parts.append(f'FB(obs={self._obs_size_fb})')
        if self._model_ct:
            parts.append(f'CT(obs={self._obs_size_ct})')
        if self._model_fb_3bet:
            parts.append(f'3BET(obs={self._obs_size_fb_3bet})')
        if self._reraise_clf:
            parts.append(f'RR(thr={self.config.reraise_threshold})')
        if self._preflop_ranges:
            parts.append(f'PF({list(self._preflop_ranges.keys())})')
        return ', '.join(parts) or 'no models loaded'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class _SupervisedWrapper:
    """Wraps a StrategyNetwork to mimic the get_strategy interface."""

    def __init__(self, model, obs_size, action_size, num_players, iteration=0):
        import torch
        self.model = model
        self.obs_size = obs_size
        self.action_size = action_size
        self.num_players = num_players
        self.iteration = iteration
        self.strategy_networks = [model] * num_players
        self.advantage_networks = [model] * num_players

    def get_strategy(self, observation, player_idx, legal_actions):
        import torch
        obs_t = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(obs_t).numpy()[0]
        masked = np.full(self.action_size, -1e9, dtype=np.float32)
        for a in legal_actions:
            if a < self.action_size:
                masked[a] = logits[a]
        masked -= masked.max()
        exp_s = np.exp(masked)
        strategy = exp_s / exp_s.sum()
        result = np.zeros(self.action_size, dtype=np.float32)
        for a in legal_actions:
            if a < self.action_size:
                result[a] = strategy[a]
        return result


def _pad_obs(obs: np.ndarray, target_size: int) -> np.ndarray:
    """Pad observation to target size, computing equity_vs_odds for the last dim."""
    if len(obs) >= target_size:
        return obs[:target_size]
    padded = np.zeros(target_size, dtype=np.float32)
    padded[:len(obs)] = obs
    if len(obs) >= 19 and target_size > len(obs):
        pot_odds = obs[IDX_POT_ODDS]
        equity = obs[IDX_EQUITY]
        padded[target_size - 1] = np.clip(equity - (1.0 - pot_odds), -1.0, 1.0)
    return padded
