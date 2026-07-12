"""Build TexasSolver range strings for hero + villain.

TexasSolver's range syntax accepts comma-separated 169-classes ('AA', 'AKs',
'T9o') and specific combos ('AsKs'), each optionally weighted with a colon
('AKs:0.5'). It does NOT understand the '+' shorthand ('22+', 'A2s+') — that
crashes the binary with a "format not recognize" SIGABRT — so every '+' range
must be expanded to explicit classes here.

Two sources, in priority order (see the GTO plan):
  1. Empirical: `src/opp_range_lookup.py` → per-spot hand-class counts from the
     showdown HH archive (`data/opp_range.sqlite`). Class weight is normalized
     so the per-combo weight reflects the class's empirical frequency.
  2. Tier fallback: the hardcoded aggression-tier ranges in
     `src/range_equity.py` (which DO use '+'), expanded to explicit classes.
"""
from typing import Dict, List, Optional, Tuple

from src.opp_range_spot_key import OppRangeSpotKey
from src.range_equity import _TIER_RANGE_STRS

_RANK_ORDER = '23456789TJQKA'
_RANK_IDX = {r: i for i, r in enumerate(_RANK_ORDER)}


def _n_combos(hand_class: str) -> int:
    """Number of card combinations a 169-class represents."""
    if len(hand_class) == 2:        # pair, e.g. 'AA'
        return 6
    if hand_class.endswith('s'):    # suited
        return 4
    return 12                       # offsuit


def build_range_string_from_combos(combos: List[Tuple[str, int]]) -> str:
    """[(hand_class, count), ...] -> 'AA:1.0,AKs:0.83,...' for TexasSolver.

    The per-class weight is `(count/total) / n_combos(class)` so that each
    individual combo's weight reflects the class's empirical frequency
    (offsuit classes have 12 combos vs 6 for pairs — without dividing they'd
    be over-represented). Weights are then scaled so the max is 1.0 to keep
    them in a sane range; only relative weights matter to the solver.
    """
    total = sum(c for _, c in combos)
    if total <= 0:
        return ''
    per_combo: Dict[str, float] = {}
    for hc, cnt in combos:
        if cnt <= 0:
            continue
        per_combo[hc] = (cnt / total) / _n_combos(hc)
    if not per_combo:
        return ''
    mx = max(per_combo.values()) or 1.0
    parts = []
    for hc, w in per_combo.items():
        wn = w / mx
        if wn < 1e-4:
            continue
        parts.append(f'{hc}:{wn:.4f}')
    return ','.join(parts)


def _expand_token(token: str) -> List[str]:
    """Expand one range token (possibly '+'-suffixed) to explicit classes.

    'AA'   -> ['AA']
    '22+'  -> ['22','33',...,'AA']
    'A2s+' -> ['A2s','A3s',...,'AKs']   (lower card climbs up to high-1)
    'K9o+' -> ['K9o','KTo','KJo','KQo']
    'JTo'  -> ['JTo']
    """
    token = token.strip()
    if not token:
        return []
    plus = token.endswith('+')
    if plus:
        token = token[:-1]
    if len(token) < 2:
        return []
    r1, r2 = token[0].upper(), token[1].upper()
    suit = token[2].lower() if len(token) >= 3 else ''
    if r1 not in _RANK_IDX or r2 not in _RANK_IDX:
        return []

    if r1 == r2:
        # Pair. '22+' -> all pairs from r1 up to AA.
        if not plus:
            return [r1 + r1]
        return [_RANK_ORDER[i] * 2 for i in range(_RANK_IDX[r1], len(_RANK_ORDER))]

    # Non-pair: keep the higher rank fixed, climb the lower rank.
    hi, lo = (r1, r2) if _RANK_IDX[r1] > _RANK_IDX[r2] else (r2, r1)
    suffix = suit if suit in ('s', 'o') else 'o'
    if not plus:
        return [hi + lo + suffix]
    out = []
    for i in range(_RANK_IDX[lo], _RANK_IDX[hi]):   # lo .. hi-1
        out.append(hi + _RANK_ORDER[i] + suffix)
    return out


def expand_range_to_explicit(range_str: str) -> str:
    """Expand a '+'-using eval7-style range string to explicit classes.

    'KK+,AKs' -> 'KK,AA,AKs'. De-duplicates while preserving order.
    """
    seen = set()
    out: List[str] = []
    for tok in range_str.split(','):
        for hc in _expand_token(tok):
            if hc not in seen:
                seen.add(hc)
                out.append(hc)
    return ','.join(out)


def tier_range_string(aggression_tier: int) -> str:
    """Explicit (no '+') range string for an aggression tier (0..3)."""
    tier = max(0, min(int(aggression_tier), len(_TIER_RANGE_STRS) - 1))
    return expand_range_to_explicit(_TIER_RANGE_STRS[tier])


def _range_has_class(range_str: str, hand_class: str) -> bool:
    """True if `hand_class` is already a token (ignoring weight) in range_str."""
    for tok in range_str.split(','):
        name = tok.split(':')[0].strip()
        if name == hand_class:
            return True
    return False


def _ensure_class_weight_floor(range_str: str, hand_class: Optional[str],
                               floor: float = 0.1) -> str:
    """Guarantee hero's hand CLASS survives the solver's weight cutoff.

    TexasSolver drops combos whose weight falls below an internal cutoff, so
    a hand the pool rarely shows down (tiny empirical weight, e.g. 0.0018) can
    vanish from the solved tree — leaving hero's actual hand unsolvable. We
    raise the class weight to `floor` (or add the class at `floor` if absent).

    Why the class and not the exact combo: TexasSolver throws a runtime_error
    if a range lists both a class ('T4o') and a specific combo of it
    ('Th4c') — duplicate combo definition. Boosting the class keeps all its
    combos (including hero's) without that conflict. The weight only affects
    how much hero's hand influences the opponent's response, never hero's own
    per-combo strategy, so a small floor is safe.
    """
    if not hand_class:
        return range_str
    parts: List[str] = []
    found = False
    for tok in range_str.split(','):
        tok = tok.strip()
        if not tok:
            continue
        name, _, w = tok.partition(':')
        name = name.strip()
        if name == hand_class:
            found = True
            try:
                wv = float(w) if w else 1.0
            except ValueError:
                wv = 1.0
            parts.append(f'{name}:{max(wv, floor):.4f}')
        else:
            parts.append(tok)
    if not found:
        parts.append(f'{hand_class}:{floor:.4f}')
    return ','.join(parts)


def build_hero_and_villain_ranges(
        hero_spot: OppRangeSpotKey,
        villain_spot: OppRangeSpotKey,
        lookup,
        hero_hand_class: Optional[str],
        aggression_tier: int = 1,
        hero_hand_combo: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Return (hero_range_str, villain_range_str, source_tag).

    source_tag ∈ {'empirical', 'tier_fallback', 'mixed'} — empirical when both
    sides come from the DB, tier_fallback when both fall back to the hardcoded
    tier ranges, mixed otherwise. Hero's exact hand class is force-added so the
    solve always contains hero's row.
    """
    hero_combos = lookup.get_range_combos(hero_spot) if lookup else None
    vill_combos = lookup.get_range_combos(villain_spot) if lookup else None

    if hero_combos:
        hero_str = build_range_string_from_combos(hero_combos)
        hero_src = 'empirical'
    else:
        hero_str = tier_range_string(aggression_tier)
        hero_src = 'tier'
    if not hero_str:                         # empirical built to empty — fall back
        hero_str = tier_range_string(aggression_tier)
        hero_src = 'tier'

    if vill_combos:
        vill_str = build_range_string_from_combos(vill_combos)
        vill_src = 'empirical'
    else:
        vill_str = tier_range_string(aggression_tier)
        vill_src = 'tier'
    if not vill_str:
        vill_str = tier_range_string(aggression_tier)
        vill_src = 'tier'

    # Guarantee hero's hand class survives the solver's weight cutoff.
    hero_str = _ensure_class_weight_floor(hero_str, hero_hand_class)

    if hero_src == vill_src == 'empirical':
        tag = 'empirical'
    elif hero_src == vill_src == 'tier':
        tag = 'tier_fallback'
    else:
        tag = 'mixed'
    return hero_str, vill_str, tag
