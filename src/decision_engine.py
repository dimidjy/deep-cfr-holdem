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
from typing import Dict, List, Optional, Tuple, Union

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
IDX_HIGH_CARD = 13      # obs[13] = high_card_norm ((hi-2)/12, A=1.0, J≈0.75)

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
# 6 reserved for fast_fold (agent pre-turn fold, not a normal action)
RAISE_THIRD = 7  # 1/3-pot bet

NUM_ACTIONS = 6  # model outputs 5 (0-4), action 5 only used by preflop ranges

ACTION_NAMES = ['Fold', 'Call', 'RaiseHalf', 'RaisePot', 'AllIn', 'RaiseMin',
                'FastFold', 'RaiseThird']

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

    # Preflop lookup support (data-driven preflop strategy)
    action_history: Optional[str] = None  # e.g. 'open', 'vs_raise', 'vs_3bet'
    num_to_act: int = 0     # players still to act behind hero, clamped {0..4}
    call_amount_bb: float = 0.0  # for nearest-preset sizing fallback

    # OUTPUT side-channel: populated by DecisionEngine when the preflop
    # lookup chooses an explicit BB raise size to be typed (rather than a
    # preset button). None for postflop, unused fallback paths, or
    # non-raise actions. Read by the action executor.
    custom_size_bb_out: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# DecisionConfig — what to enable/disable
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DecisionConfig:
    """Configuration for the decision pipeline.

    Pipeline = preflop lookup (data-driven cohort) → model inference →
    postflop sizing override → sampling. The legacy strong_hand_guard /
    allin_filter / style_tuning heuristics were removed once the cohort
    lookup + postflop sizing made them redundant.
    """
    # Preflop
    use_preflop_ranges: bool = True   # static GTO range fallback
    use_preflop_lookup: bool = True   # data-driven cohort lookup (preferred)
    preflop_db_path: str = 'data/preflop_lookup.sqlite'
    # Minimum cohort play_rate to trust the lookup for a spot.
    # Below this, the spot is treated as fold (cohort folds it most of the
    # time → showdown-only data under-counts it).
    preflop_min_play_rate: float = 0.25
    # Tighter thresholds for facing aggression: NL5 population 3-bets and
    # 4-bets value-heavy, so defending the cohort's regular range loses to
    # dominated showdowns. Bumping the gate folds marginal hands like
    # AQo/AJs/77/JTs vs 3-bets and TT/AQs vs 4-bets, which a winning-reg
    # cohort defends against balanced 3-bet ranges but NL5 randoms don't
    # 3-bet wide enough to justify.
    preflop_min_play_rate_vs_3bet: float = 0.40
    preflop_min_play_rate_vs_4bet: float = 0.50
    # Cohort sample minimums for each lookup level. Higher values push more
    # spots to the next level (or static fallback) — useful for tightening
    # ranges when the cohort's calling-station bias is leaking through.
    preflop_min_samples_exact: int = 5      # L0 (hole, pos, hist, stack, nta)
    preflop_min_samples_no_nta: int = 3     # L1 (drop num_to_act)
    preflop_min_samples_no_stack: int = 5   # L2 (drop stack_bucket)

    # Postflop sizing (data-driven cohort sizes for bets/raises)
    use_postflop_sizing: bool = True
    postflop_db_path: str = 'data/postflop_sizing.sqlite'
    # Per-SPR-bucket threshold: cohort shove% at/above which a model raise is
    # promoted to ALL_IN. Keys MUST match SPR_BUCKETS labels in
    # src/postflop_spot_key.py. Stratified by SPR because deep-stack avg
    # all_in_pct is 0.4-2.5% (no spot ever crosses any threshold) while
    # SPR<2 averages 71.9% — a scalar threshold can't capture both regimes.
    # Scalar float still accepted for back-compat.
    postflop_all_in_threshold: Union[float, Dict[str, float]] = field(
        default_factory=lambda: {
            '<2': 0.25, '2-5': 0.30, '5-15': 0.45, '15+': 0.55,
        })
    # Sample floor specifically for ALL_IN promotion. Decoupled from the
    # general postflop-lookup min_samples (5) used for sizing-percentile
    # overrides, because the binary all_in_pct metric is high-variance — at
    # n=5 a single decision swings the rate by 20pp. Sizing percentiles are
    # robust at smaller n; promotion needs a tighter floor.
    postflop_min_samples_for_promotion: int = 20
    # Histogram-based bet-size sampling: when True, draw the bet size from
    # the cohort's empirical bucketed distribution (per spot) instead of
    # the deterministic p25/p75 percentile. Adds variance and reproduces
    # bimodal cohorts (e.g. spots that split between half- and pot-sized
    # bets). Falls back to percentile path on sparse spots.
    postflop_use_histogram_sampling: bool = True
    postflop_min_samples_for_histogram: int = 20

    # Re-raise classifier
    reraise_classifier_path: str = ''
    reraise_threshold: float = 0.25

    # Sampling
    raise_randomize: bool = True  # 80/20 raise size randomization
    mix_raise_sizes: bool = True  # when top-2 are Raise½ + RaisePot, sample
                                  # between them proportional to their probs
                                  # (e.g. 30%/40% → pick half 43%, pot 57%)
    bet_over_check: bool = True   # when argmax is Check/Call but
                                  # (Raise½ + RaisePot) > Check, override to
                                  # a small bet (Raise½) — "trust the model
                                  # that betting is correct when combined
                                  # raise probability beats check"
    # Threshold-sample: when > 0, replaces argmax. Zero out any legal action
    # with prob < cap, renormalize the remainder, then sample from that
    # distribution. Bypasses all post-argmax overrides (bet_over_check,
    # mix_raise_sizes, raise_randomize). Used to test whether stochastic
    # play above a confidence floor outperforms strict argmax. 0 = disabled.
    action_sample_cap: float = 0.0

    # Low-stakes postflop sampling: in small short-handed pots (≤3 active,
    # pot ≤8 BB), sample the action stochastically from the model's full
    # probability distribution instead of taking argmax. The variance is
    # cheap (small pot), and it preserves CFR's mixed-strategy intent —
    # prevents the agent from being readable in spots that are common but
    # not high-stakes.
    postflop_sample_low_pot: bool = True
    postflop_sample_max_players: int = 3
    postflop_sample_max_pot_bb: float = 5.0

    # Model temperature (used when loading, not at inference)
    temperature: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# All-in promotion (river-only, optional — separate from removed allin_filter)
# ─────────────────────────────────────────────────────────────────────────────

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
# DecisionEngine
# ─────────────────────────────────────────────────────────────────────────────

class DecisionEngine:
    """Single source of truth for all poker decisions.

    Handles model routing, preflop lookup/ranges, postflop sizing override,
    and sampling. Every consumer (agent, validation, analysis, GUI) uses
    this class.
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
        # Postflop sizing lookup (lazy-loaded by load_preflop_ranges()).
        # None when DB missing or disabled — postflop sizing becomes a no-op.
        self._postflop_sizing = None

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
        """Load preflop strategy: data-driven `PreflopLookup` (preferred) wrapping
        static `PreflopRanges` as fallback. Falls back to ranges-only if the
        lookup DB is missing or `use_preflop_lookup` is disabled.
        """
        if not self.config.use_preflop_ranges and not self.config.use_preflop_lookup:
            return
        try:
            from src.preflop_ranges import PreflopRanges
            ranges_per_n: Dict[int, 'PreflopRanges'] = {}
            for np_ in (num_players_list or [2, 3, 4, 5, 6]):
                try:
                    ranges_per_n[np_] = PreflopRanges(num_players=np_)
                except Exception:
                    pass

            preflop_lookup_loaded = False
            if self.config.use_preflop_lookup:
                try:
                    from pathlib import Path as _Path
                    from src.preflop_lookup import PreflopLookup
                    if _Path(self.config.preflop_db_path).exists():
                        # Wrap each per-N range in a lookup. The lookup loads
                        # the DB once; we just re-hand it different fallbacks.
                        for np_, fb in ranges_per_n.items():
                            self._preflop_ranges[np_] = PreflopLookup(
                                self.config.preflop_db_path, fallback=fb,
                                min_samples_exact=(
                                    self.config.preflop_min_samples_exact),
                                min_samples_no_nta=(
                                    self.config.preflop_min_samples_no_nta),
                                min_samples_no_stack=(
                                    self.config.preflop_min_samples_no_stack),
                                min_play_rate=self.config.preflop_min_play_rate,
                                min_play_rate_vs_3bet=(
                                    self.config.preflop_min_play_rate_vs_3bet),
                                min_play_rate_vs_4bet=(
                                    self.config.preflop_min_play_rate_vs_4bet))
                        logger.info(
                            f"Preflop lookup loaded from "
                            f"{self.config.preflop_db_path} "
                            f"(num_players={list(self._preflop_ranges.keys())})")
                        preflop_lookup_loaded = True
                    else:
                        logger.warning(
                            f"Preflop lookup DB not found at "
                            f"{self.config.preflop_db_path} — "
                            f"falling back to static ranges only.")
                except Exception as e:
                    logger.warning(f"Could not load preflop lookup: {e} — "
                                   f"falling back to static ranges.")

            if not preflop_lookup_loaded:
                # No lookup installed → use plain ranges
                self._preflop_ranges.update(ranges_per_n)
                logger.info(f"Preflop ranges (static) loaded: "
                            f"{list(self._preflop_ranges.keys())}")
        except Exception as e:
            logger.warning(f"Could not load preflop strategy: {e}")

        # ── Postflop sizing lookup ──────────────────────────────────────
        if self.config.use_postflop_sizing:
            try:
                from pathlib import Path as _Path
                from src.postflop_sizing import PostflopSizingLookup
                if _Path(self.config.postflop_db_path).exists():
                    self._postflop_sizing = PostflopSizingLookup(
                        self.config.postflop_db_path,
                        all_in_promotion_threshold=self.config.postflop_all_in_threshold,
                        min_samples_for_promotion=self.config.postflop_min_samples_for_promotion,
                        use_histogram_sampling=self.config.postflop_use_histogram_sampling,
                        min_samples_for_histogram=self.config.postflop_min_samples_for_histogram,
                    )
                    logger.info(
                        f"Postflop sizing loaded from "
                        f"{self.config.postflop_db_path} "
                        f"(spots={self._postflop_sizing.num_spots})")
                else:
                    logger.warning(
                        f"Postflop sizing DB not found at "
                        f"{self.config.postflop_db_path} — "
                        f"postflop sizing disabled (preset sizes only).")
            except Exception as e:
                logger.warning(
                    f"Could not load postflop sizing: {e} — "
                    f"postflop sizing disabled.")

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

        # 3. Re-raise classifier (optional)
        if ctx.facing_bet and self._reraise_clf is not None:
            strategy = self._apply_reraise(strategy, ctx.obs, legal)

        return strategy

    def decide(self, ctx: SpotContext) -> Tuple[int, List[float]]:
        """Get strategy + sample action. Full pipeline.

        Returns (action_index, strategy_list_of_6).
        """
        strategy = self.get_strategy(ctx)
        strategy_list = [float(strategy[i]) for i in range(NUM_ACTIONS)]
        # Stochastic sampling for low-stakes postflop spots: small short-
        # handed pots use the model's full probability distribution instead
        # of argmax. Keeps the agent unreadable in cheap spots where
        # variance is fine; preserves CFR's mixed-strategy intent.
        if (self.config.postflop_sample_low_pot
                and ctx.street != 'preflop'
                and ctx.num_players <= self.config.postflop_sample_max_players
                and ctx.pot_bb <= self.config.postflop_sample_max_pot_bb):
            action = self._sample_from_distribution(
                strategy_list, ctx.legal_actions)
        else:
            action = self._sample_action(strategy_list, ctx.legal_actions, ctx.obs)
        return action, strategy_list

    @staticmethod
    def _sample_from_distribution(strategy: List[float],
                                  legal_actions: List[int]) -> int:
        """Sample an action proportional to model probabilities, restricted to
        legal actions. Falls back to argmax if all legal probabilities are 0.
        """
        # Restrict to legal actions and renormalize.
        probs = []
        actions = []
        for a in legal_actions:
            p = strategy[a] if 0 <= a < len(strategy) else 0.0
            if p > 0:
                actions.append(a)
                probs.append(p)
        if not actions:
            # All-zero strategy on legal slots — fall back to argmax over legal.
            best, best_p = legal_actions[0], -1.0
            for a in legal_actions:
                p = strategy[a] if 0 <= a < len(strategy) else 0.0
                if p > best_p:
                    best, best_p = a, p
            return best
        total = sum(probs)
        norm = [p / total for p in probs]
        return int(np.random.choice(actions, p=norm))

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
        """Get preflop strategy from PreflopLookup (data-driven) or
        PreflopRanges (static fallback). Side-effect: writes
        `ctx.custom_size_bb_out` when the lookup chose an explicit BB size
        for a raise; the action executor reads that field to type the size.
        """
        try:
            pf = self._preflop_ranges.get(ctx.num_players)
            if pf is None:
                pf = self._preflop_ranges.get(6)
            if pf is None:
                return None

            pot_bb = ctx.pot_bb
            if pot_bb <= 0 and ctx.facing_bet:
                pot_bb = 3.5  # reasonable default

            # PreflopLookup returns (strategy, custom_size_bb); legacy
            # PreflopRanges returns just strategy. Support both.
            from src.preflop_lookup import PreflopLookup as _PreflopLookup
            if isinstance(pf, _PreflopLookup):
                strategy, custom_size_bb = pf.get_strategy(
                    hole_cards=ctx.hole_cards, rel_pos=ctx.rel_pos,
                    pot_bb=pot_bb, legal_actions=ctx.legal_actions,
                    num_limpers=ctx.num_limpers, num_callers=ctx.num_callers,
                    effective_stack=ctx.effective_stack,
                    num_players=ctx.num_players,
                    action_history=ctx.action_history,
                    num_to_act=ctx.num_to_act,
                    call_amount_bb=ctx.call_amount_bb,
                )
                ctx.custom_size_bb_out = custom_size_bb
            else:
                strategy = pf.get_strategy(
                    hole_cards=ctx.hole_cards, rel_pos=ctx.rel_pos,
                    pot_bb=pot_bb, legal_actions=ctx.legal_actions,
                    num_limpers=ctx.num_limpers, num_callers=ctx.num_callers,
                    effective_stack=ctx.effective_stack)
                ctx.custom_size_bb_out = None

            if strategy is not None:
                result = np.zeros(NUM_ACTIONS, dtype=np.float32)
                for i in range(min(len(strategy), NUM_ACTIONS)):
                    result[i] = float(strategy[i])
                return result
        except Exception as e:
            logger.warning(f"Preflop strategy failed: {e}")
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
                        legal_actions: List[int],
                        obs: Optional[np.ndarray] = None) -> int:
        """Pick an action from model output.

        Default: argmax.

        With `mix_raise_sizes`: when the top 2 actions are Raise½ (2) and
        RaisePot (3), sample between them proportional to their probabilities.
        E.g., strategy [30% half, 40% pot] → pick half 43% of hands, pot 57%.
        This preserves CFR's mixed-strategy intent for raise-sizing spots.

        With `raise_randomize`: additional 80/20 randomization among all raise
        sizes (legacy behavior).

        With `bet_over_check`: if `obs` provided, converts Check to small bet
        when model wants to bet (split across sizes) AND hand has semi-bluff
        equity (draws) or overcard air. Without `obs`, override is skipped.
        """
        # Safely pull probability even if action index exceeds strategy length
        # (happens for agent-side overrides like RaiseThird=7 that the model
        # doesn't output — treat as probability 0)
        def _prob(a: int) -> float:
            return strategy[a] if 0 <= a < len(strategy) else 0.0

        # ── Threshold sampling (research mode) ──────────────────────────
        # When action_sample_cap > 0, sample from the model's distribution
        # restricted to legal actions whose prob ≥ cap. Bypasses all the
        # heuristic overrides below — used to A/B-test stochastic play vs
        # argmax. Falls back to argmax-over-legal if no legal action meets
        # the cap (avoids picking uniformly from very-low-prob actions).
        cap = float(getattr(self.config, 'action_sample_cap', 0.0))
        if cap > 0:
            kept = [(a, _prob(a)) for a in legal_actions if _prob(a) >= cap]
            if kept:
                acts = [a for a, _ in kept]
                probs = np.array([p for _, p in kept], dtype=np.float64)
                s = probs.sum()
                if s > 0:
                    probs /= s
                    return int(np.random.choice(acts, p=probs))
            # No legal action ≥ cap → fall through to argmax fallback below.

        best_action = legal_actions[0]
        best_prob = -1.0
        for a in legal_actions:
            if _prob(a) > best_prob:
                best_prob = _prob(a)
                best_action = a

        # ── Check → Bet override ─────────────────────────────────────────
        # When argmax is Check/Call (action 1) but the COMBINED probability
        # of Raise½ + RaisePot exceeds Check's probability, the model actually
        # prefers betting (just split across two sizes). Override to a small bet
        # — but only for hands with semi-bluff equity (flush/straight draw) or
        # air with strong overcards. Without this filter, marginal made hands
        # (middle pair etc.) get small-bet-spewed where checking is correct.
        # Prefer RaiseThird (1/3 pot), fall back to RaiseMin
        # (~2BB), then Raise½ if neither available.
        if (self.config.bet_over_check
                and best_action == CALL
                and RAISE_HALF in legal_actions
                and RAISE_POT in legal_actions
                and obs is not None
                and len(obs) > IDX_STRAIGHT_DRAW):
            combined_raise = strategy[RAISE_HALF] + strategy[RAISE_POT]
            if combined_raise > strategy[CALL]:
                has_flush_draw = float(obs[IDX_FLUSH_DRAW]) > 0.0
                has_straight_draw = float(obs[IDX_STRAIGHT_DRAW]) > 0.0
                made_hand = float(obs[IDX_MADE_HAND])
                high_card = float(obs[IDX_HIGH_CARD]) if len(obs) > IDX_HIGH_CARD else 0.0
                # Air with J+ high card = bluff candidate with backdoor equity
                has_high_cards_air = (made_hand < HAND_ONE_PAIR - 0.01
                                      and high_card >= 0.75)
                if has_flush_draw or has_straight_draw or has_high_cards_air:
                    if RAISE_THIRD in legal_actions:
                        best_action = RAISE_THIRD
                    elif RAISE_MIN in legal_actions:
                        best_action = RAISE_MIN
                    else:
                        best_action = RAISE_HALF

        # ── Raise-size mixing (Raise½ vs RaisePot) ───────────────────────
        # When those two are the top 2 by probability, sample between them
        # proportionally. Keeps model's mixed-strategy intent for sizing.
        if (self.config.mix_raise_sizes
                and RAISE_HALF in legal_actions
                and RAISE_POT in legal_actions):
            # Find the two largest probabilities among legal actions
            # (use _prob to safely handle actions beyond strategy length)
            legal_sorted = sorted(legal_actions, key=lambda a: -_prob(a))
            if len(legal_sorted) >= 2:
                top1, top2 = legal_sorted[0], legal_sorted[1]
                top_pair = {top1, top2}
                if top_pair == {RAISE_HALF, RAISE_POT}:
                    p_half = _prob(RAISE_HALF)
                    p_pot = _prob(RAISE_POT)
                    total = p_half + p_pot
                    if total > 0:
                        best_action = (RAISE_HALF
                                        if random.random() * total < p_half
                                        else RAISE_POT)

        # 80% keep model's raise, 20% randomize among other raise sizes
        if self.config.raise_randomize and best_action >= RAISE_HALF:
            raise_actions = [a for a in legal_actions
                             if a >= RAISE_HALF and _prob(a) > 0.01]
            if len(raise_actions) > 1 and random.random() > 0.80:
                others = [a for a in raise_actions if a != best_action]
                if others:
                    probs = [_prob(a) for a in others]
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
