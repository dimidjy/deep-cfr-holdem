"""Run a model over `spot_library` spots and report what it decides.

Two answers per spot, because they differ and the difference is the point:

  * **raw**      — the network's own strategy, nothing applied
  * **pipeline** — what the agent actually plays after the staged decision
                   pipeline (`src/agent/decision_pipeline.py`): lookups,
                   all-in guards, sizing overrides and sampling

A model can look fine raw and behave badly once the pipeline runs, or the
reverse. Showing only one hides which layer is responsible for a decision.

Both use the same `GameState` construction as `scripts/inspect_model_spots.py`
so results are comparable with that script.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.agent.game_state import CardInfo, GameState, PlayerInfo
from src.spot_library import Spot

ACTION_NAMES = ['Fold', 'Call', 'Raise½', 'RaisePot', 'AllIn', 'RaiseMin',
                '—', 'Raise⅓']
NUM_ACTIONS = 6


def _card(s: str) -> CardInfo:
    return CardInfo(s[0].upper(), s[1].lower())


def build_game_state(spot: Spot) -> GameState:
    """Seat hero either on the button (IP) or in the big blind (OOP).

    Only `num_players` seats are active; the rest are folded out, which is how
    a shorter-handed pot actually reaches the flop.
    """
    n = spot.num_players
    btn_seat = 0
    my_seat = 0 if spot.position == 'IP' else min(2, n - 1)

    players: List[PlayerInfo] = []
    for seat in range(6):
        active = seat < n
        is_hero = seat == my_seat
        # The player we are facing posts the bet we are up against.
        bet = spot.call_bb if (active and not is_hero and seat == (
            my_seat + 1) % max(n, 1)) else 0.0
        players.append(PlayerInfo(
            seat=seat, name=f'p{seat}',
            stack=spot.stack_bb if active else 0.0,
            is_active=active, is_sitting_out=not active,
            bet_current_round=bet if spot.call_bb > 0 else 0.0,
            hole_cards=[_card(c) for c in spot.hole] if is_hero else None,
            has_button=(seat == btn_seat)))

    # You cannot fold when nothing is bet — checking is free. Offering FOLD
    # in a checked-to spot lets the model put mass on an impossible action
    # and inflates every fold statistic.
    legal = [0, 1, 2, 3, 4] if spot.call_bb > 0 else [1, 2, 3, 4]

    return GameState(
        players=players, my_seat=my_seat,
        community_cards=[_card(c) for c in spot.board],
        pot_total=spot.pot_bb, street=spot.street, button_seat=btn_seat,
        small_blind=0.5, big_blind=1.0,
        legal_actions=legal,
        num_active_players=n,
        call_amount=spot.call_bb, min_raise=2.0)


@dataclass
class SpotResult:
    spot: Dict
    raw: List[float]
    pipeline: List[float]
    raw_action: int
    pipeline_action: int
    equity: Optional[float]
    made_hand: Optional[str]
    error: str = ''

    def to_dict(self) -> Dict:
        return {
            'spot': self.spot,
            'raw': [round(x, 4) for x in self.raw],
            'pipeline': [round(x, 4) for x in self.pipeline],
            'raw_action': self.raw_action,
            'pipeline_action': self.pipeline_action,
            'raw_name': ACTION_NAMES[self.raw_action]
            if 0 <= self.raw_action < len(ACTION_NAMES) else '?',
            'pipeline_name': ACTION_NAMES[self.pipeline_action]
            if 0 <= self.pipeline_action < len(ACTION_NAMES) else '?',
            'equity': round(self.equity, 3) if self.equity is not None else None,
            'made_hand': self.made_hand,
            'error': self.error,
        }


# Pipeline settings used when the caller does not override them. These are
# the knobs the tournament-selected configuration runs with.
DEFAULT_SETTINGS = {
    'equity_model': 'models/equity_model_v3.pt',
    'temperature': 1.0,
    'use_preflop_ranges': True,
    'use_preflop_lookup': True,
    'allin_promote': True,
    'allin_promote_frac': 0.7,
    'allin_call_filter': True,
    'allin_call_frac': 0.7,
    'postflop_promotion_requires_strong_hand': True,
    'action_sample_cap': 0.0,
    'cbet_action_sample_cap': 0.4,
}
DEFAULT_MODEL = 'models/best/checkpoint_iter_4000.pt'

# Base 6-max observation width before the optional blocks
# (position 8 + pot 3 + hand 5 + stack 2 + equity 9 + street 4 + game 6 + bet 5).
_BASE_OBS = 42


class PolicyRunner:
    """DecisionEngine + observation builder + decision pipeline for a single
    model. Construction loads the checkpoint."""

    def __init__(self, model_path: str = '', settings: Optional[Dict] = None):
        from src.decision_engine import DecisionConfig, DecisionEngine
        from src.agent.decision_pipeline import build_pipeline
        from src.agent.obs_builder import ObservationBuilder

        cfg = dict(DEFAULT_SETTINGS, **(settings or {}))
        dc_keys = set(DecisionConfig.__dataclass_fields__)
        self.engine = DecisionEngine(DecisionConfig(
            **{k: v for k, v in cfg.items() if k in dc_keys}))

        if cfg.get('equity_model'):
            try:
                import torch
                from src.engine.fast_enhanced_engine import load_equity_model
                load_equity_model(cfg['equity_model'],
                                  device='cuda' if torch.cuda.is_available() else 'cpu')
            except Exception:
                pass  # falls back to Monte Carlo equity

        model_path = model_path or DEFAULT_MODEL
        self.engine.load_models(facing_bet=model_path, checked_to=model_path)
        self.engine.load_preflop_ranges()

        # The checkpoint's input width says which optional obs blocks it was
        # trained with: behavioral stats (+8, +3 with betting line), then
        # equity-vs-odds (+1).
        extra = max(self.engine._obs_size_fb, self.engine._obs_size_ct) - _BASE_OBS
        use_bh = extra >= 8
        if use_bh:
            extra -= 11 if extra >= 11 else 8
        self.obs_builder = ObservationBuilder(
            6, behavioral_stats=use_bh, equity_vs_odds=extra >= 1)
        self.pipeline = build_pipeline(self.engine, cfg, low_pot_sampling=False)

    def decide(self, gs: GameState):
        """Returns (action_idx, strategy) for one postflop snapshot, or None
        when the observation cannot be built."""
        from src.decision_engine import SpotContext
        from src.agent.decision_pipeline import DecisionState

        obs = self.obs_builder.build(gs)
        if obs is None:
            return None
        legal = list(gs.legal_actions or [0, 1])
        my = gs.my_player
        call = float(gs.call_amount)
        pot = float(gs.pot_total or 0.0)
        ctx = SpotContext(
            obs=obs, legal_actions=legal, facing_bet=call > 0,
            street=gs.street, rel_pos=gs.compute_rel_pos(),
            pot_bb=pot, num_players=int(gs.num_active_players),
            effective_stack=float(my.stack + call) if my else 100.0,
            call_amount_bb=call)
        st = DecisionState(
            ctx=ctx, engine=self.engine, obs=obs, legal=legal,
            street=gs.street,
            call_amount_bb=call,
            my_stack_bb=float(my.stack) if my else 0.0,
            effective_stack_bb=float(my.stack + call) if my else 100.0,
            min_raise_bb=float(gs.min_raise or 0.0) or 1.0,
            pot_total=pot,
            community_cards=[str(c) for c in gs.community_cards],
            hole_cards=[str(c) for c in gs.my_hole_cards or []] or None,
            num_active=int(gs.num_active_players),
            facing_bet_size_pct=(call / pot) if call > 0.01 and pot > 0.01 else None,
        )
        self.pipeline.run(st)
        return st.action, st.strategy_list


class SpotEvaluator:
    """Holds a configured PolicyRunner. Construction loads the model, so keep
    one instance per model path rather than rebuilding per spot."""

    def __init__(self, runner: PolicyRunner) -> None:
        self._runner = runner

    @classmethod
    def for_model(cls, model_path: str = '',
                  settings: Optional[Dict] = None) -> 'SpotEvaluator':
        return cls(PolicyRunner(model_path, settings))

    # ── evaluation ─────────────────────────────────────────────────────────
    def evaluate(self, spot: Spot) -> SpotResult:
        gs = build_game_state(spot)
        blank = [0.0] * NUM_ACTIONS
        try:
            obs = self._runner.obs_builder.build(gs)
        except Exception as exc:
            return SpotResult(spot.to_dict(), blank, blank, -1, -1,
                              None, None, f'obs build failed: {exc}')

        equity = made = None
        if obs is not None:
            from src.obs_schema import IDX_EQUITY, IDX_MADE_HAND, made_hand_label
            if len(obs) > IDX_MADE_HAND:
                equity = float(obs[IDX_EQUITY])
                made = made_hand_label(float(obs[IDX_MADE_HAND]))

        raw = self._raw_strategy(obs, gs)
        pipeline, action = self._pipeline_strategy(gs)
        raw_action = int(np.argmax(raw)) if any(raw) else -1
        return SpotResult(spot.to_dict(), raw, pipeline, raw_action, action,
                          equity, made)

    def strategy_for_vector(self, obs, legal: Optional[List[int]] = None
                            ) -> List[float]:
        """Model strategy for a hand-supplied observation vector.

        Bypasses GameState entirely so a vector can be edited feature by
        feature to see what the decision actually depends on.
        """
        import numpy as _np
        out = [0.0] * NUM_ACTIONS
        legal = legal or [0, 1, 2, 3, 4]
        try:
            engine = self._runner.engine
            model = (getattr(engine, '_model_fb', None)
                     or getattr(engine, '_model_ct', None))
            if model is None:
                return out
            arr = _np.asarray(obs, dtype=_np.float32)
            probs = model.get_strategy(arr, 0, list(legal))
            for i in range(min(len(probs), NUM_ACTIONS)):
                out[i] = float(probs[i])
        except Exception:
            pass
        return out

    def obs_for_spot(self, spot: Spot):
        """The observation vector a spot produces, for seeding the explorer."""
        gs = build_game_state(spot)
        try:
            obs = self._runner.obs_builder.build(gs)
            return (list(map(float, obs)) if obs is not None else None,
                    list(gs.legal_actions))
        except Exception:
            return None, [0, 1, 2, 3, 4]

    def _raw_strategy(self, obs, gs: GameState) -> List[float]:
        out = [0.0] * NUM_ACTIONS
        try:
            engine = self._runner.engine
            model = (getattr(engine, '_model_fb', None)
                     or getattr(engine, '_model_ct', None))
            if model is None or obs is None:
                return out
            probs = model.get_strategy(obs, 0, list(gs.legal_actions))
            for i in range(min(len(probs), NUM_ACTIONS)):
                out[i] = float(probs[i])
        except Exception:
            pass
        return out

    def _pipeline_strategy(self, gs: GameState) -> tuple:
        """PolicyRunner.decide() returns (action_idx, strategy), or None when
        the obs builder fails."""
        out = [0.0] * NUM_ACTIONS
        try:
            result = self._runner.decide(gs)
            if result is None:
                return out, -1
            action, strategy = int(result[0]), result[1]
            if strategy is not None:
                for i in range(min(len(strategy), NUM_ACTIONS)):
                    out[i] = float(strategy[i])
            elif 0 <= action < NUM_ACTIONS:
                out[action] = 1.0
            return out, action
        except Exception:
            return out, -1


def summarise(results: List[SpotResult]) -> Dict:
    """Aggregate fold / call / aggression rates, overall and per category."""
    def bucket(probs: List[float]) -> Dict[str, float]:
        total = sum(probs) or 1.0
        return {
            'fold': probs[0] / total,
            'call': probs[1] / total,
            'aggr': sum(probs[2:]) / total,
        }

    per_cat: Dict[str, Dict] = {}
    overall = {'fold': 0.0, 'call': 0.0, 'aggr': 0.0, 'n': 0}
    for r in results:
        if r.error or not any(r.pipeline):
            continue
        b = bucket(r.pipeline)
        cat = r.spot['category']
        slot = per_cat.setdefault(cat, {
            'label': r.spot['category_label'], 'rank': r.spot['rank'],
            'fold': 0.0, 'call': 0.0, 'aggr': 0.0, 'n': 0})
        for k in ('fold', 'call', 'aggr'):
            slot[k] += b[k]
            overall[k] += b[k]
        slot['n'] += 1
        overall['n'] += 1

    for slot in per_cat.values():
        n = max(slot['n'], 1)
        for k in ('fold', 'call', 'aggr'):
            slot[k] = round(slot[k] / n, 4)
    n = max(overall['n'], 1)
    for k in ('fold', 'call', 'aggr'):
        overall[k] = round(overall[k] / n, 4)

    return {'overall': overall,
            'by_category': sorted(per_cat.values(), key=lambda s: s['rank'])}
