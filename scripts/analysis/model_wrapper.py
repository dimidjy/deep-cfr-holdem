"""
Shared model wrapper for analysis scripts.

Thin wrapper around DecisionEngine that provides the get_action/get_strategy
interface expected by analyzers. Uses the same decision pipeline as the live
agent (model routing, guards, style tuning).

Usage:
    from scripts.analysis.model_wrapper import ModelWrapper
    model = ModelWrapper(
        model_fb_path='models/cql_6p_facing_bet_v9.pt',
        model_ct_path='models/cql_6p_checked_to_v5.pt',
        model_fb_3bet_path='models/cql_6p_facing_bet_3bet_v2.pt',
        use_guards=True,   # same pipeline as agent
    )
    action, strategy = model.get_action(obs, legal_actions)
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.decision_engine import (
    DecisionEngine, DecisionConfig, SpotContext,
    IDX_POT_ODDS, IDX_POT_NORM,
    FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN,
    ACTION_NAMES, _3BET_POT_BB,
)


class ModelWrapper:
    """Unified model wrapper with DecisionEngine for analysis scripts.

    By default uses the full agent pipeline (guards, style tuning).
    Set use_guards=False for raw model comparison.
    """

    def __init__(
        self,
        model_fb_path: str,
        model_ct_path: str,
        model_fb_3bet_path: Optional[str] = None,
        device: str = 'cpu',
        use_guards: bool = True,
    ):
        dc = DecisionConfig(
            strong_hand_guard=use_guards,
            allin_filter=use_guards,
            style_tuning=use_guards,
            use_preflop_ranges=False,  # analysis doesn't use preflop ranges
            raise_randomize=False,  # deterministic for analysis
        )
        self._engine = DecisionEngine(dc)
        self._engine.load_models(
            facing_bet=model_fb_path,
            checked_to=model_ct_path,
            facing_bet_3bet=model_fb_3bet_path or '',
        )
        self.device = device
        self.use_guards = use_guards
        print(f'[ModelWrapper] DecisionEngine: {self._engine.info()} '
              f'(guards={"ON" if use_guards else "OFF"})')

    def is_facing_bet(self, obs: np.ndarray) -> bool:
        return float(obs[IDX_POT_ODDS]) > 0.001 if len(obs) > IDX_POT_ODDS else False

    def is_3bet_pot(self, obs: np.ndarray) -> bool:
        pot_bb = obs[IDX_POT_NORM] * 600 if len(obs) > IDX_POT_NORM else 0
        return pot_bb > _3BET_POT_BB

    def get_action(
        self,
        obs: np.ndarray,
        legal_actions: List[int],
        player_idx: int = 0,
    ) -> Tuple[int, np.ndarray]:
        """Get model's preferred action and full strategy."""
        facing_bet = self.is_facing_bet(obs)
        pot_bb = obs[IDX_POT_NORM] * 600 if len(obs) > IDX_POT_NORM else 0

        ctx = SpotContext(
            obs=obs, legal_actions=legal_actions,
            facing_bet=facing_bet, street='postflop',
            pot_bb=pot_bb)

        strategy = self._engine.get_strategy(ctx)
        best_action = max(legal_actions, key=lambda a: strategy[a])
        return best_action, strategy

    def get_strategy(
        self,
        obs: np.ndarray,
        legal_actions: List[int],
        player_idx: int = 0,
    ) -> np.ndarray:
        """Get model's full strategy."""
        _, strategy = self.get_action(obs, legal_actions, player_idx)
        return strategy

    def info(self) -> str:
        return self._engine.info()
