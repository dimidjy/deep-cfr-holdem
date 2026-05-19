"""Runtime opponent-range lookup with Monte Carlo equity vs the empirical range.

Mirrors `src/postflop_sizing.py` patterns: lazy SQLite connection, in-memory
cache of (spot → list of (hand_class, count)) for hot loop performance.

Usage:
    lookup = OppRangeLookup('data/opp_range.sqlite')
    equity = lookup.equity(
        hero_hole=['Ks','Td'], community=['6h','9s','Kh','4h'],
        street='turn', board_texture='monotone', position='OOP',
        spr_bucket='2-5', action_context='face_check_raise',
    )
    # → 0.42 (empirical equity vs hands opps actually show up with here)
    # or None if spot has < MIN_SAMPLES_PER_SPOT observations.
"""
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.opp_range_spot_key import (
    OppRangeSpotKey, MIN_SAMPLES_PER_SPOT,
)

try:
    import eval7 as _eval7
    _EVAL7_AVAILABLE = True
except ImportError:
    _EVAL7_AVAILABLE = False


_RANK_CHARS = '23456789TJQKA'  # eval7 Card.rank is 0-indexed (2=0 .. A=12)


def _hand_class_from_cards(c1, c2) -> str:
    """Map an `eval7.Card` pair to its 169-class string ('AKs', 'T9o', 'QQ')."""
    r1, r2 = c1.rank, c2.rank
    s1, s2 = c1.suit, c2.suit
    if r1 == r2:
        return _RANK_CHARS[r1] + _RANK_CHARS[r2]
    if r1 < r2:
        r1, r2 = r2, r1
    return _RANK_CHARS[r1] + _RANK_CHARS[r2] + ('s' if s1 == s2 else 'o')


class OppRangeLookup:
    """Thread-safe lookup of empirical opp ranges + equity computation."""

    def __init__(self, db_path: str,
                 min_samples: int = MIN_SAMPLES_PER_SPOT):
        self.db_path = db_path
        self.min_samples = min_samples
        self._cache: Optional[Dict[Tuple, List[Tuple[str, int]]]] = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._cache is not None:
            return
        with self._lock:
            if self._cache is not None:
                return
            cache: Dict[Tuple, List[Tuple[str, int]]] = {}
            if not Path(self.db_path).exists():
                self._cache = cache
                return
            conn = sqlite3.connect(self.db_path)
            try:
                rows = conn.execute(
                    "SELECT street, board_texture, position, spr_bucket, "
                    "action_context, hand_class, count "
                    "FROM opp_range_lookup").fetchall()
            except sqlite3.Error:
                rows = []
            finally:
                conn.close()
            for r in rows:
                key = (r[0], r[1], r[2], r[3], r[4])
                cache.setdefault(key, []).append((r[5], r[6]))
            self._cache = cache

    def get_range_combos(self, spot: OppRangeSpotKey
                          ) -> Optional[List[Tuple[str, int]]]:
        """Return [(hand_class, count), ...] for the spot if data exists
        AND total count ≥ min_samples. Otherwise None.
        """
        self._ensure_loaded()
        combos = self._cache.get(spot.as_tuple())
        if not combos:
            return None
        if sum(c for _, c in combos) < self.min_samples:
            return None
        return combos

    def build_handrange(self, spot: OppRangeSpotKey):
        """Return an `eval7.HandRange` with custom per-combo weights from
        empirical counts. Returns None if data insufficient / eval7 missing.

        Note: eval7's HandRange string syntax does NOT support per-class
        weights (`AA:0.5` fails to parse). We build an unweighted range from
        the present classes, then overwrite `.hands[i][1]` (weight).

        Per-combo weight = (class count / total) / N_combos_in_class.
        This ensures eval7's per-combo MC sampling matches the empirical
        class-level frequencies (pairs have 6 combos, suited 4, offsuit 12 —
        without normalizing, offsuit classes get 2-3× over-sampled).
        """
        if not _EVAL7_AVAILABLE:
            return None
        combos = self.get_range_combos(spot)
        if combos is None:
            return None
        total = sum(c for _, c in combos)
        weight_by_class = {hc: count / total for hc, count in combos}
        try:
            rng = _eval7.HandRange(','.join(weight_by_class.keys()))
        except Exception:
            return None
        # Count combos-per-class in the built range (so we can divide)
        class_combo_count = {}
        for (cards, _w) in rng.hands:
            hc = _hand_class_from_cards(cards[0], cards[1])
            class_combo_count[hc] = class_combo_count.get(hc, 0) + 1
        # Per-combo weight = class freq / N combos in class
        new_hands = []
        for (cards, _w) in rng.hands:
            hc = _hand_class_from_cards(cards[0], cards[1])
            n = max(class_combo_count.get(hc, 1), 1)
            w = weight_by_class.get(hc, 0.0) / n
            new_hands.append((cards, w))
        try:
            rng.hands = new_hands
        except Exception:
            return None
        return rng

    def equity(self, hero_hole: List[str], community: List[str],
                spot: OppRangeSpotKey, n_samples: int = 300
                ) -> Optional[float]:
        """MC equity for hero_hole vs the empirical range at this spot.

        Returns None if the spot is under-sampled (callers fall back to
        the hardcoded tier ranges in `src/range_equity.py`).
        """
        rng = self.build_handrange(spot)
        if rng is None:
            return None
        if len(hero_hole) < 2:
            return None
        try:
            e7_hole = [_eval7.Card(c) for c in hero_hole[:2]]
            e7_board = [_eval7.Card(c) for c in community] if community else []
            return float(_eval7.py_hand_vs_range_monte_carlo(
                e7_hole, rng, e7_board, n_samples))
        except Exception:
            return None

    def build_handrange_board_aware(self, spot: OppRangeSpotKey,
                                      community: List[str]):
        """Like `build_handrange`, but re-weights each hand_class by how
        well it fits the current board. Made hands and strong draws get
        boosted; air gets downweighted.

        Concretely: weight = empirical_count * fit_score(hand_class, board).

        Falls back to plain build_handrange if treys unavailable or no
        community cards.
        """
        if not _EVAL7_AVAILABLE:
            return None
        if not community:
            return self.build_handrange(spot)
        combos = self.get_range_combos(spot)
        if combos is None:
            return None
        # Re-weight by board fit
        from src.range_equity import _hand_class_fit_score
        adj_weights = {}
        for hand_class, count in combos:
            fit = _hand_class_fit_score(hand_class, community)
            w = count * fit
            if w > 0:
                adj_weights[hand_class] = w
        if not adj_weights:
            return None
        # Normalize for downstream sanity (eval7 doesn't strictly need this
        # but keeps weights bounded ≤ 1)
        total = sum(adj_weights.values())
        weight_by_class = {hc: w / total for hc, w in adj_weights.items()}
        try:
            rng = _eval7.HandRange(','.join(weight_by_class.keys()))
        except Exception:
            return None
        # Count combos-per-class then divide so per-combo weight reflects
        # the class-level empirical frequency (same fix as build_handrange).
        class_combo_count = {}
        for (cards, _w) in rng.hands:
            hc = _hand_class_from_cards(cards[0], cards[1])
            class_combo_count[hc] = class_combo_count.get(hc, 0) + 1
        new_hands = []
        for (cards, _w) in rng.hands:
            hc = _hand_class_from_cards(cards[0], cards[1])
            n = max(class_combo_count.get(hc, 1), 1)
            w = weight_by_class.get(hc, 0.0) / n
            new_hands.append((cards, w))
        try:
            rng.hands = new_hands
        except Exception:
            return None
        return rng

    def equity_board_aware(self, hero_hole: List[str], community: List[str],
                            spot: OppRangeSpotKey, n_samples: int = 300
                            ) -> Optional[float]:
        """MC equity vs the board-aware empirical range at this spot.

        Returns None if the spot is under-sampled or no board cards.
        """
        if not community:
            return None
        rng = self.build_handrange_board_aware(spot, community)
        if rng is None:
            return None
        if len(hero_hole) < 2:
            return None
        try:
            e7_hole = [_eval7.Card(c) for c in hero_hole[:2]]
            e7_board = [_eval7.Card(c) for c in community]
            return float(_eval7.py_hand_vs_range_monte_carlo(
                e7_hole, rng, e7_board, n_samples))
        except Exception:
            return None

    def spot_count(self, spot: OppRangeSpotKey) -> int:
        """Total observations at a spot (0 if missing)."""
        self._ensure_loaded()
        combos = self._cache.get(spot.as_tuple())
        if not combos:
            return 0
        return sum(c for _, c in combos)


# Singleton convenience: the DB is small enough (~few MB) that one instance
# per process is fine.
_DEFAULT_DB_PATH = 'data/opp_range.sqlite'
_singleton: Optional[OppRangeLookup] = None
_singleton_lock = threading.Lock()


def get_default_lookup(db_path: str = _DEFAULT_DB_PATH
                       ) -> Optional[OppRangeLookup]:
    """Lazy singleton. Returns None if the DB file is missing."""
    global _singleton
    if _singleton is not None:
        return _singleton if Path(_singleton.db_path).exists() else None
    with _singleton_lock:
        if _singleton is None:
            if not Path(db_path).exists():
                return None
            _singleton = OppRangeLookup(db_path)
    return _singleton
