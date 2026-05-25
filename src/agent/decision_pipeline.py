"""Unified staged decision pipeline — the single source of truth for how a
poker decision is built from a model strategy.

Both the runtime agent (`src/agent/policy_core.py`) and self-play
(`scripts/analysis/self_play.py`) drive this one ordered pipeline, so they no
longer diverge. Each stage is a thin wrapper that delegates to the existing,
already-tested logic in `src/decision_engine.py` / `src/postflop_sizing.py`;
this module owns only the ORDER and the small relocated guards. The ordered
list in `build_pipeline()` is the place to read to see "what overrides what".

Conceptual grouping the stages map to:
  1. Model decision         → ModelStrategyStage
  2. Conditional action      → ArchetypeSkew / BluffSuppression / EVSearch /
     override (fold/call/       AllInPromote / AllInCallFilter / SampleAction /
     half/pot/allin)           BlockBetPromo
  3. Database override        → PostflopSizingStage (cohort sizing + all-in
                                promotion)
  4. Sizing override          → PostflopSizingStage (size) + RaiseThirdSizing

The pipeline operates purely on a `DecisionState` bundle — it never imports
`GameState`, so self-play (which has no GameState snapshot) fills
the same bundle from its simulator.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import logging
import numpy as np

from src.decision_engine import (
    NUM_ACTIONS,
    FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN, RAISE_THIRD,
    IDX_MADE_HAND, IDX_TOP_PAIR, IDX_EQUITY,
    IDX_BOARD_PAIR, IDX_BOARD_FLUSH, IDX_BOARD_STRAIGHT,
    HAND_ONE_PAIR, HAND_TRIPS, HAND_FLUSH, HAND_FULL_HOUSE,
    _promote_to_allin,
)

logger = logging.getLogger('src.agent.decision_pipeline')


# ─────────────────────────────────────────────────────────────────────────────
# DecisionState — the mutable bundle threaded through every stage (pure data)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DecisionState:
    # Inputs (set by the driver before run)
    ctx: object                       # SpotContext (obs, legal_actions, street, rel_pos, ...)
    engine: object                    # DecisionEngine — stages call get_strategy/_sample_action/...
    obs: np.ndarray
    legal: List[int]
    street: str

    # Game-state-derived scalars the relocated guards read. Each driver
    # computes these from its OWN data source (live GameState / sim engine).
    call_amount_bb: float = 0.0
    my_stack_bb: float = 0.0          # hero remaining stack
    effective_stack_bb: float = 100.0
    min_raise_bb: float = 1.0
    pot_total: float = 0.0            # raw pot (RAISE_THIRD + postflop sizing pot_bb)
    community_cards: Optional[List[str]] = None
    hole_cards: Optional[List[str]] = None
    num_active: int = 6
    was_preflop_raiser: bool = False  # driver computes (live heuristic vs sim seat)
    facing_bet_size_pct: Optional[float] = None

    # Mutated by stages
    strategy: Optional[np.ndarray] = None
    strategy_list: Optional[List[float]] = None
    action: Optional[int] = None
    custom_size_bb: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Stage interface
# ─────────────────────────────────────────────────────────────────────────────
class DecisionStage(ABC):
    name: str = 'stage'

    @abstractmethod
    def applies(self, st: DecisionState) -> bool:
        """Cheap gate (street / config / legal). run() is called only if True."""

    @abstractmethod
    def run(self, st: DecisionState) -> None:
        """Mutate `st` in place. Only invoked when applies() is True."""


# ── 1. Model decision ────────────────────────────────────────────────────────
class ModelStrategyStage(DecisionStage):
    """Raw strategy: preflop lookup OR model inference (+ re-raise classifier).
    Delegates entirely to DecisionEngine.get_strategy."""
    name = 'model_strategy'

    def applies(self, st: DecisionState) -> bool:
        return True

    def run(self, st: DecisionState) -> None:
        st.strategy = st.engine.get_strategy(st.ctx)
        # Preflop lookup may have set an explicit BB raise size (side-channel).
        st.custom_size_bb = st.ctx.custom_size_bb_out


# ── 2. Conditional action overrides ──────────────────────────────────────────
class ArchetypeSkewStage(DecisionStage):
    """Optional: skew the strategy toward an opponent archetype."""
    name = 'archetype_skew'

    def __init__(self, skew_fn):
        self._skew_fn = skew_fn

    def applies(self, st: DecisionState) -> bool:
        return self._skew_fn is not None and st.street != 'preflop'

    def run(self, st: DecisionState) -> None:
        st.strategy = self._skew_fn(st.strategy, st.legal)


class BluffSuppressionStage(DecisionStage):
    """Optional: suppress bluff raises with weak hands (multiway / HU air)."""
    name = 'bluff_suppression'

    def __init__(self, use_bluff_guard: bool):
        self._on = bool(use_bluff_guard)

    def applies(self, st: DecisionState) -> bool:
        return self._on and st.street != 'preflop' and len(st.obs) > 19

    def run(self, st: DecisionState) -> None:
        # Verbatim relocation of default_core bluff guard, reading st fields.
        strategy, obs, legal = st.strategy, st.obs, st.legal
        made_hand = float(obs[19])
        n_active = st.num_active
        not_facing_bet = st.call_amount_bb == 0

        if n_active >= 3 and made_hand < 0.23 and not_facing_bet:
            raise_mass = sum(strategy[a] for a in [2, 3, 4, 5] if a < len(strategy))
            if raise_mass > 0.1:
                shift = raise_mass * 0.8
                for a in [2, 3, 4, 5]:
                    if a < len(strategy):
                        strategy[a] *= 0.2
                strategy[1] += shift
                s = sum(strategy[a] for a in legal if a < len(strategy))
                if s > 0:
                    for a in range(len(strategy)):
                        strategy[a] /= s
        elif n_active == 2 and made_hand < 0.115 and not_facing_bet:
            has_draw = False
            if len(obs) > 21:
                has_draw = obs[20] > 0.4 or obs[21] > 0.4
            if not has_draw:
                raise_mass = sum(strategy[a] for a in [2, 3, 4, 5] if a < len(strategy))
                if raise_mass > 0.3:
                    shift = raise_mass * 0.5
                    for a in [2, 3, 4, 5]:
                        if a < len(strategy):
                            strategy[a] *= 0.5
                    strategy[1] += shift
                    s = sum(strategy[a] for a in legal if a < len(strategy))
                    if s > 0:
                        for a in range(len(strategy)):
                            strategy[a] /= s


class EVSearchStage(DecisionStage):
    """Optional: Monte-Carlo EV search override (off by default)."""
    name = 'ev_search'

    def __init__(self, use_search: bool):
        self._on = bool(use_search)

    def applies(self, st: DecisionState) -> bool:
        return self._on and st.street != 'preflop'

    def run(self, st: DecisionState) -> None:
        try:
            from src.agent.search import search_override
            hole = st.hole_cards
            if hole:
                board = st.community_cards or []
                search_model = None
                try:
                    m, _ = st.engine._route_model(st.ctx)
                    if m:
                        search_model = m
                except Exception:
                    pass
                search_legal = [a for a in st.legal if a <= 3]
                if search_legal:
                    st.strategy, _evs = search_override(
                        st.strategy, hole, board,
                        pot_bb=st.pot_total,
                        call_amount_bb=st.call_amount_bb,
                        stack_bb=st.my_stack_bb,
                        street=st.street,
                        legal_actions=search_legal,
                        n_active=st.num_active,
                        min_ev_diff=0.5,
                        model=search_model,
                        n_sims=100,
                    )
        except Exception:
            pass  # search failure = use model as-is


class AllInPromoteStage(DecisionStage):
    """Optional: river near-nut RaisePot → AllIn (model has no all-in head)."""
    name = 'allin_promote'

    def __init__(self, allin_promote: bool, frac: float):
        self._on = bool(allin_promote)
        self._frac = float(frac)

    def applies(self, st: DecisionState) -> bool:
        return self._on and st.street == 'river'

    def run(self, st: DecisionState) -> None:
        st.strategy = _promote_to_allin(st.strategy, st.obs, st.legal,
                                        shift_frac=self._frac)


class AllInCallFilterStage(DecisionStage):
    """Optional: fold big/shove-sized bets unless set+ on a safe board.
    Verbatim relocation of default_core._apply_allin_call_filter."""
    name = 'allin_call_filter'

    def __init__(self, use_filter: bool, frac: float):
        self._on = bool(use_filter)
        self._frac = float(frac)

    def applies(self, st: DecisionState) -> bool:
        return self._on and st.street != 'preflop'

    def run(self, st: DecisionState) -> None:
        strategy, obs, legal = st.strategy, st.obs, st.legal
        if CALL not in legal or FOLD not in legal:
            return
        if len(obs) <= IDX_BOARD_FLUSH:
            return
        if st.my_stack_bb <= 0:
            return
        committed = st.call_amount_bb / (st.call_amount_bb + st.my_stack_bb)
        if committed < self._frac:
            return

        made_hand = float(obs[IDX_MADE_HAND])
        board_flush = float(obs[IDX_BOARD_FLUSH])
        board_pair = float(obs[IDX_BOARD_PAIR])

        weak = made_hand < HAND_TRIPS - 0.01
        flush_board_no_flush = (board_flush > 0.75
                                and made_hand < HAND_FLUSH - 0.01)
        paired_board_no_boat = (board_pair > 0.5
                                and made_hand < HAND_FULL_HOUSE - 0.01)
        if not (weak or flush_board_no_flush or paired_board_no_boat):
            return  # set+ on safe board → allow call

        call_mass = float(strategy[CALL])
        if call_mass < 0.05:
            return
        strategy = strategy.copy()
        strategy[FOLD] += call_mass
        strategy[CALL] = 0.0
        s = sum(strategy[a] for a in legal if a < len(strategy))
        if s > 0:
            for a in range(len(strategy)):
                strategy[a] = strategy[a] / s
        st.strategy = strategy


class SampleActionStage(DecisionStage):
    """Snapshot strategy → list, then sample the action. Live path uses plain
    argmax-based `_sample_action`; self-play sets `low_pot_sampling=True` to keep
    its stochastic small-pot sampling (relocated from DecisionEngine.decide)."""
    name = 'sample_action'

    def __init__(self, low_pot_sampling: bool = False):
        self._low_pot = bool(low_pot_sampling)

    def applies(self, st: DecisionState) -> bool:
        return True

    def run(self, st: DecisionState) -> None:
        st.strategy_list = [float(st.strategy[i]) for i in range(NUM_ACTIONS)]
        eng, ctx = st.engine, st.ctx
        cfg = eng.config
        if (self._low_pot
                and getattr(cfg, 'postflop_sample_low_pot', False)
                and ctx.street != 'preflop'
                and ctx.num_players <= cfg.postflop_sample_max_players
                and ctx.pot_bb <= cfg.postflop_sample_max_pot_bb):
            st.action = eng._sample_from_distribution(st.strategy_list, st.legal)
        else:
            st.action = eng._sample_action(st.strategy_list, st.legal, st.obs)


class BlockBetPromoStage(DecisionStage):
    """Flop: TP+ on a wet board → upgrade RAISE_HALF to RAISE_POT (so the sizing
    lookup returns cohort p75). Verbatim relocation."""
    name = 'block_bet_promo'

    def applies(self, st: DecisionState) -> bool:
        return (st.street == 'flop' and st.action == RAISE_HALF
                and len(st.obs) > IDX_BOARD_STRAIGHT)

    def run(self, st: DecisionState) -> None:
        obs = st.obs
        made_hand = float(obs[IDX_MADE_HAND])
        top_pair = float(obs[IDX_TOP_PAIR])
        equity = float(obs[IDX_EQUITY])
        board_flush = float(obs[IDX_BOARD_FLUSH])
        board_straight = float(obs[IDX_BOARD_STRAIGHT])
        is_tp_plus = (
            top_pair >= 0.5
            or made_hand >= HAND_ONE_PAIR + 0.05
            or (made_hand >= HAND_ONE_PAIR - 0.01 and equity >= 0.65))
        is_wet_board = board_flush >= 0.33 or board_straight >= 0.4
        if is_tp_plus and is_wet_board:
            st.action = RAISE_POT
            logger.info(
                f"Block-bet promo: TP+ on wet flop "
                f"(top_pair={top_pair:.2f} made={made_hand:.2f} "
                f"eq={equity:.2f} flush_tx={board_flush:.2f} "
                f"straight_tx={board_straight:.2f}) "
                f"→ RAISE_HALF promoted to RAISE_POT")


# ── 3 + 4. Database override + sizing override ───────────────────────────────
class PostflopSizingStage(DecisionStage):
    """Data-driven cohort sizing: replace the preset's implied size with the
    cohort size, and promote raise→ALL_IN when the cohort shoves the spot.
    Delegates to engine._postflop_sizing.adjust; builds spot_inputs from st."""
    name = 'postflop_sizing'

    def applies(self, st: DecisionState) -> bool:
        return (st.street in ('flop', 'turn', 'river')
                and getattr(st.engine, '_postflop_sizing', None) is not None)

    @staticmethod
    def _build_inputs(st: DecisionState) -> dict:
        # Mirrors default_core._build_postflop_sizing_inputs exactly, reading
        # st fields. was_preflop_raiser / facing_bet_size_pct are precomputed
        # by the driver; pot floor matches the original.
        pot_bb = st.pot_total if st.pot_total > 0 else 0.01
        return {
            'street': st.street,
            'obs': st.obs,
            'hole_cards': st.hole_cards,
            'community_cards': st.community_cards or [],
            'rel_pos': st.ctx.rel_pos if st.ctx else 0,
            'num_active': st.num_active,
            'effective_stack_bb': st.effective_stack_bb,
            'pot_bb': pot_bb,
            'was_preflop_raiser': st.was_preflop_raiser,
            'facing_bet_size_pct': st.facing_bet_size_pct,
            'facing_check_raise': False,
            'call_amount_bb': st.call_amount_bb,
        }

    def run(self, st: DecisionState) -> None:
        try:
            spot_inputs = self._build_inputs(st)
            new_action, custom_size = st.engine._postflop_sizing.adjust(
                st.action, spot_inputs)
            if new_action != st.action:
                st.action = new_action
                st.strategy_list = [0.0] * NUM_ACTIONS
                if st.action < NUM_ACTIONS:
                    st.strategy_list[st.action] = 1.0
                logger.info(
                    f"Postflop sizing: action promoted to ALL_IN "
                    f"(custom_size={custom_size:.1f}bb)")
            if custom_size is not None:
                st.custom_size_bb = custom_size
                logger.info(
                    f"Postflop sizing override → custom raise size "
                    f"{custom_size:.2f}bb ({st.street})")
        except Exception as exc:
            logger.warning(f"Postflop sizing override failed: {exc}")


class RaiseThirdSizingStage(DecisionStage):
    """RAISE_THIRD → typed 1/3-pot bet (the client's 1/3-pot preset button is a
    min-bet, so we compute + type the size). Verbatim relocation."""
    name = 'raise_third_sizing'

    def applies(self, st: DecisionState) -> bool:
        return st.action == RAISE_THIRD and st.custom_size_bb is None

    def run(self, st: DecisionState) -> None:
        pot_bb = float(st.pot_total or 0.0)
        call_bb = float(st.call_amount_bb or 0.0)
        min_raise_bb = float(st.min_raise_bb or 0.0) or 1.0
        third_pot = call_bb + (pot_bb + call_bb) / 3.0
        st.custom_size_bb = max(round(third_pot * 2) / 2, min_raise_bb)
        logger.info(
            f"RAISE_THIRD → custom 1/3-pot bet {st.custom_size_bb:.2f}bb "
            f"(pot={pot_bb:.2f} call={call_bb:.2f} "
            f"min_raise={min_raise_bb:.2f})"
            f"")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline + factory
# ─────────────────────────────────────────────────────────────────────────────
class DecisionPipeline:
    def __init__(self, stages: List[DecisionStage]):
        self._stages = stages

    def run(self, st: DecisionState) -> DecisionState:
        for stage in self._stages:
            if stage.applies(st):
                stage.run(st)
        return st

    def describe(self) -> List[str]:
        """Ordered stage names — the single-glance view of the pipeline."""
        return [s.name for s in self._stages]


def build_pipeline(engine, config: dict,
                   low_pot_sampling: bool = False) -> DecisionPipeline:
    """Declare the ONE ordered decision pipeline. This list is the single
    source of truth for what overrides what, and in what order.

    `low_pot_sampling`: False for the runtime agent (plain argmax sampling),
    True for self-play (preserves its stochastic small-pot sampling).
    """
    skew_fn = None
    archetype_name = config.get('archetype_skew', '')
    if archetype_name:
        from src.archetypes import get_archetype_skew
        skew_fn = get_archetype_skew(archetype_name)

    return DecisionPipeline([
        # 1. Model decision
        ModelStrategyStage(),
        # 2. Conditional action overrides
        ArchetypeSkewStage(skew_fn),
        BluffSuppressionStage(config.get('use_bluff_guard', False)),
        EVSearchStage(config.get('use_search', False)),
        AllInPromoteStage(config.get('allin_promote', False),
                          float(config.get('allin_promote_frac', 0.5))),
        AllInCallFilterStage(config.get('allin_call_filter', False),
                             float(config.get('allin_call_frac', 0.7))),
        SampleActionStage(low_pot_sampling=low_pot_sampling),
        BlockBetPromoStage(),
        # 3 + 4. Database + sizing override
        PostflopSizingStage(),
        RaiseThirdSizingStage(),
    ])
