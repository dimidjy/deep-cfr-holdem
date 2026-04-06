"""
Per-player opponent stat tracking for play.

Tracks VPIP, PFR, AF, Fold-to-CBet, and WTSD across hands.
Persists to JSON files so stats survive between sessions.
Stats are computed from actions only — no hole cards needed.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

from .archetypes import OPPONENT_PROFILE_DIM


@dataclass
class PlayerProfile:
    """Raw action counters for a single opponent."""
    name: str
    hands: int = 0

    # Preflop
    vpip_count: int = 0          # voluntarily put $ in pot
    pfr_count: int = 0           # preflop raise

    # Postflop aggression
    postflop_bets: int = 0       # bet or raise actions
    postflop_calls: int = 0      # call actions

    # Continuation bet
    cbet_opps: int = 0           # opportunities to CBet
    cbet_count: int = 0          # times they CBet
    fold_to_cbet_opps: int = 0   # times they faced a CBet
    fold_to_cbet_count: int = 0  # times they folded to CBet

    # Showdown
    saw_flop: int = 0
    went_to_showdown: int = 0

    def to_profile(self) -> np.ndarray:
        """Convert raw counters to 5-dim normalized profile.

        Format: [VPIP, PFR, AF_norm, Fold2CBet, WTSD]
        All values [0, 1].
        """
        vpip = self.vpip_count / max(self.hands, 1)
        pfr = self.pfr_count / max(self.hands, 1)

        # AF = (bets + raises) / calls.  Normalize: AF/(AF+1) → [0, 1]
        af = self.postflop_bets / max(self.postflop_calls, 1)
        af_norm = af / (af + 1.0)

        fold_to_cbet = self.fold_to_cbet_count / max(self.fold_to_cbet_opps, 1)
        wtsd = self.went_to_showdown / max(self.saw_flop, 1)

        return np.array([vpip, pfr, af_norm, fold_to_cbet, wtsd],
                        dtype=np.float32)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'PlayerProfile':
        """Deserialize from dict."""
        return cls(**d)


class OpponentTracker:
    """Track and persist per-player statistics across hands.

    Usage:
        tracker = OpponentTracker()
        profile = tracker.get_profile('player_name')  # 5-dim np.ndarray
        tracker.get_player('player_name').vpip_count += 1
        tracker.save('player_name')
    """

    MIN_HANDS = 30       # below this → return zeros (GTO fallback)
    CONFIDENCE_HANDS = 100  # full confidence at this many hands

    def __init__(self, data_dir: str = 'data/opponent_stats'):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._players: Dict[str, PlayerProfile] = {}

    def get_player(self, name: str) -> PlayerProfile:
        """Get or create a player profile (in-memory)."""
        if name not in self._players:
            self._players[name] = self._load_or_create(name)
        return self._players[name]

    def get_profile(self, name: str) -> np.ndarray:
        """Get 5-dim opponent profile with confidence blending.

        Below MIN_HANDS: returns zeros (GTO fallback).
        Between MIN_HANDS and CONFIDENCE_HANDS: linearly blends toward real stats.
        Above CONFIDENCE_HANDS: returns raw stats.
        """
        p = self.get_player(name)
        if p.hands < self.MIN_HANDS:
            return np.zeros(OPPONENT_PROFILE_DIM, dtype=np.float32)

        raw = p.to_profile()
        blend = min(1.0, p.hands / self.CONFIDENCE_HANDS)
        return raw * blend

    def get_aggregate_profile(self, names: list) -> np.ndarray:
        """Average profile across multiple opponents.

        Used in multiway pots where we face multiple villains.
        Returns zeros if no valid profiles exist.
        """
        profiles = [self.get_profile(n) for n in names]
        profiles = [p for p in profiles if p.sum() > 0]
        if not profiles:
            return np.zeros(OPPONENT_PROFILE_DIM, dtype=np.float32)
        return np.mean(profiles, axis=0).astype(np.float32)

    def save(self, name: str):
        """Persist player stats to JSON."""
        if name not in self._players:
            return
        path = self._data_dir / f'{self._safe_name(name)}.json'
        with open(path, 'w') as f:
            json.dump(self._players[name].to_dict(), f, indent=2)

    def save_all(self):
        """Persist all tracked players."""
        for name in self._players:
            self.save(name)

    def _load_or_create(self, name: str) -> PlayerProfile:
        """Load from disk or create new profile."""
        path = self._data_dir / f'{self._safe_name(name)}.json'
        if path.exists():
            with open(path, 'r') as f:
                return PlayerProfile.from_dict(json.load(f))
        return PlayerProfile(name=name)

    @staticmethod
    def _safe_name(name: str) -> str:
        """Sanitize player name for use as filename."""
        return ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
