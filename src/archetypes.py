"""
Opponent archetypes for exploitation training.

Each archetype is a strategy skew function that modifies model output
to simulate a specific player type. Used during Deep CFR fine-tuning
so the model learns to exploit different tendencies.

Archetype wrappers are NOT separate models — they take the base model's
strategy output and multiply action probabilities to simulate a play style.
"""

import numpy as np
from typing import List, Callable, Dict

# Action indices (must match decision_engine.py)
FOLD = 0
CALL = 1
RAISE_HALF = 2
RAISE_POT = 3
ALL_IN = 4

OPPONENT_PROFILE_DIM = 5

# ── Archetype target profiles ────────────────────────────────────────────────
# Format: [VPIP, PFR, AF_norm, Fold2CBet, WTSD]
#   AF_norm = AF / (AF + 1)  →  squashes unbounded AF into [0, 1]
#   e.g. AF=2.0 → 0.667,  AF=0.5 → 0.333,  AF=4.5 → 0.818
#
# These are TARGET profiles. The actual stats produced by the skew functions
# depend on the base model and must be calibrated via self-play measurement.
# ─────────────────────────────────────────────────────────────────────────────

ARCHETYPE_PROFILES: Dict[str, np.ndarray] = {
    'calling_station': np.array([0.55, 0.10, 0.33, 0.20, 0.45], dtype=np.float32),
    'nit':             np.array([0.12, 0.10, 0.67, 0.75, 0.22], dtype=np.float32),
    'maniac':          np.array([0.60, 0.45, 0.82, 0.15, 0.38], dtype=np.float32),
    'weak_tight':      np.array([0.18, 0.14, 0.60, 0.65, 0.28], dtype=np.float32),
    'gto':             np.array([0.22, 0.18, 0.71, 0.45, 0.30], dtype=np.float32),
}

ARCHETYPE_NAMES = list(ARCHETYPE_PROFILES.keys())


# ── Strategy skew functions ──────────────────────────────────────────────────
# Each function takes a strategy array (5 actions) and list of legal action
# indices, returns a modified strategy. Multipliers are initial guesses —
# calibrate by running self_play.py --archetype <name> and comparing stats
# to target profiles above.
# ─────────────────────────────────────────────────────────────────────────────

def _apply_skew(strategy: np.ndarray, legal: List[int],
                fold_mult: float, call_mult: float, raise_mult: float) -> np.ndarray:
    """Apply multiplicative skew to strategy, renormalize over legal actions."""
    s = strategy.copy()
    s[FOLD] *= fold_mult
    s[CALL] *= call_mult
    s[RAISE_HALF] *= raise_mult
    s[RAISE_POT] *= raise_mult
    s[ALL_IN] *= raise_mult
    # Zero out illegal actions
    mask = np.zeros(len(s), dtype=np.float32)
    mask[legal] = 1.0
    s *= mask
    total = s.sum()
    if total > 0:
        s /= total
    else:
        s[legal] = 1.0 / len(legal)
    return s


def calling_station_skew(strategy: np.ndarray, legal: List[int]) -> np.ndarray:
    """Loose-passive: calls everything, rarely folds or raises."""
    return _apply_skew(strategy, legal,
                       fold_mult=0.15, call_mult=2.5, raise_mult=0.4)


def nit_skew(strategy: np.ndarray, legal: List[int]) -> np.ndarray:
    """Ultra-tight: folds most hands, only plays premiums aggressively."""
    return _apply_skew(strategy, legal,
                       fold_mult=2.5, call_mult=0.6, raise_mult=0.8)


def maniac_skew(strategy: np.ndarray, legal: List[int]) -> np.ndarray:
    """Loose-aggressive: raises constantly, rarely folds or flat-calls."""
    return _apply_skew(strategy, legal,
                       fold_mult=0.3, call_mult=0.4, raise_mult=3.0)


def weak_tight_skew(strategy: np.ndarray, legal: List[int]) -> np.ndarray:
    """Tight-passive: folds a lot to aggression, rarely raises postflop."""
    return _apply_skew(strategy, legal,
                       fold_mult=2.0, call_mult=0.8, raise_mult=0.5)


def gto_skew(strategy: np.ndarray, legal: List[int]) -> np.ndarray:
    """No adjustment — returns strategy unchanged (self-play baseline)."""
    return strategy


_SKEW_FUNCTIONS: Dict[str, Callable] = {
    'calling_station': calling_station_skew,
    'nit':             nit_skew,
    'maniac':          maniac_skew,
    'weak_tight':      weak_tight_skew,
    'gto':             gto_skew,
}


def get_archetype_skew(name: str) -> Callable:
    """Get the strategy skew function for an archetype."""
    if name not in _SKEW_FUNCTIONS:
        raise ValueError(f'Unknown archetype: {name!r}. '
                         f'Available: {list(_SKEW_FUNCTIONS.keys())}')
    return _SKEW_FUNCTIONS[name]


def get_archetype_profile(name: str) -> np.ndarray:
    """Get the 5-dim opponent profile for an archetype."""
    if name not in ARCHETYPE_PROFILES:
        raise ValueError(f'Unknown archetype: {name!r}. '
                         f'Available: {list(ARCHETYPE_PROFILES.keys())}')
    return ARCHETYPE_PROFILES[name].copy()


def random_archetype() -> str:
    """Pick a random archetype name (uniform over all types including GTO)."""
    return ARCHETYPE_NAMES[np.random.randint(len(ARCHETYPE_NAMES))]
