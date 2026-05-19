"""Range-conditioned equity for obs builders.

Both runtime (src/agent/obs_builder.py) and training
(src/hand_history/obs_builder.py, src/engine/fast_enhanced_engine.py) obs
builders import from here so threat-tier logic and ranges stay in sync.

The model's existing `hand_strength` is equity vs a uniform-random opp range,
which over-estimates equity when the opponent has shown aggression on a
wet board. This module adds a second equity number computed against a
narrower range selected by recent aggression signals.

Tiers (rough hand-strength percentile):
- 0: random (~100%, full 1326 combos) — no aggression
- 1: wide value (~40%)               — single bet/raise, average sizing
- 2: tight value (~15%)              — two raises OR wet+large bet
- 3: nutted (~5%)                    — 3bet+ OR jam OR wet+huge bet
"""

from typing import List, Tuple

try:
    import eval7 as _eval7
    _EVAL7_AVAILABLE = True
except ImportError:
    _EVAL7_AVAILABLE = False

try:
    from treys import Card as _TreysCard, Evaluator as _TreysEvaluator
    _TREYS_EVALUATOR = _TreysEvaluator()
    _TREYS_AVAILABLE = True
except ImportError:
    _TREYS_AVAILABLE = False


_RANK_ORDER = '23456789TJQKA'
_RANK_VAL = {c: i for i, c in enumerate(_RANK_ORDER)}   # '2'→0 .. 'A'→12
_SUITS = ('s', 'h', 'd', 'c')


def _build_random_range_str() -> str:
    combos = []
    for i, r1 in enumerate(_RANK_ORDER):
        for j, r2 in enumerate(_RANK_ORDER):
            if i < j:
                combos.append(f'{r2}{r1}s')
                combos.append(f'{r2}{r1}o')
            elif i == j:
                combos.append(f'{r1}{r2}')
    return ','.join(combos)


_TIER_RANGE_STRS = [
    _build_random_range_str(),
    # ~40% — open-raise / loose 3bet-call / std cbet range
    "22+,A2s+,A8o+,K2s+,K9o+,Q2s+,Q9o+,J6s+,JTo,T7s+,T9o,96s+,98o,86s+,76s,65s,54s",
    # ~15% — 3bet value / strong cbet / barrel range
    "66+,A9s+,ATo+,K9s+,KJo+,Q9s+,QTo+,J9s+,JTo,T9s,98s,87s",
    # ~5% — 4bet+ value / river-jam range
    "99+,AJs+,AQo+,KQs",
]

if _EVAL7_AVAILABLE:
    THREAT_TIER_RANGES = [_eval7.HandRange(s) for s in _TIER_RANGE_STRS]
else:
    THREAT_TIER_RANGES = [None] * len(_TIER_RANGE_STRS)


def select_threat_tier(num_raises_norm: float = 0.0,
                       max_bet_sizing: float = 0.0,
                       facing_3bet_plus: float = 0.0,
                       pot_was_raised: float = 0.0,
                       board_flush_texture: float = 0.0,
                       board_straight_texture: float = 0.0,
                       opp_aggressed_this_hand: float = 0.0) -> int:
    """Pick tier 0..3 from per-street aggression signals.

    `num_raises_norm` and `max_bet_sizing` accumulate hand-wide (preflop
    contributions never reset), so using them caused 91% of postflop spots
    to map to tier 3 in realistic play. Both are accepted for API
    compatibility but ignored.

    `opp_aggressed_this_hand` (1.0 if opp has bet/raised at any prior point
    this hand) lifts tier from 0 → 1 when the current street is checked
    through. Carries multi-street aggression context that pot_was_raised
    loses when bets get collected into the pot between streets.
    """
    if pot_was_raised < 0.5:
        # No bet THIS street, but did opp show aggression earlier?
        if opp_aggressed_this_hand >= 0.5:
            tier = 1                   # prior aggression → opp's range narrow
        else:
            tier = 0                   # truly passive
    elif facing_3bet_plus >= 0.5:
        tier = 3                       # 3bet+ pressure this street
    else:
        tier = 1                       # single bet/raise this street

    board_wet = max(board_flush_texture, board_straight_texture) > 0.4
    if board_wet and tier > 0:
        tier = min(tier + 1, 3)        # wet board → tighter implied range

    return tier


def compute_threat_range_equity(hole_cards: List[str],
                                community_cards: List[str],
                                tier: int,
                                n_samples: int = 300) -> float:
    """MC equity for hole_cards vs the tier's range given community_cards.

    Returns 0.0 on any failure (eval7 unavailable, parse error, no cards).
    """
    if not _EVAL7_AVAILABLE or tier < 0 or tier >= len(THREAT_TIER_RANGES):
        return 0.0
    rng = THREAT_TIER_RANGES[tier]
    if rng is None:
        return 0.0
    if len(hole_cards) < 2:
        return 0.0
    try:
        e7_hole = [_eval7.Card(c) for c in hole_cards[:2]]
        e7_board = [_eval7.Card(c) for c in community_cards] if community_cards else []
        return float(_eval7.py_hand_vs_range_monte_carlo(
            e7_hole, rng, e7_board, n_samples))
    except Exception:
        return 0.0


def get_threat_range_equity(hole_cards: List[str],
                             community_cards: List[str],
                             tier: int,
                             spot_key=None,
                             db_lookup=None,
                             n_samples: int = 300,
                             board_aware: bool = True,
                             tier_safety_floor: bool = True) -> float:
    """DB-first equity computation with tier safety floor.

    Resolution order:
    1. **Board-aware DB**: take DB's hand-class distribution at this spot,
       re-weight each class by how well it fits the current board (made
       hands + draws get boosted, air drops), then MC equity vs that
       filtered range. Requires `db_lookup`, `spot_key`, `community_cards`,
       and `board_aware=True`.
    2. **Plain DB**: empirical distribution at this spot (no board filter).
    3. **Tier fallback**: hardcoded range bucket by aggression tier.

    With `tier_safety_floor=True` (default), return `min(DB, tier)`. The DB
    bucket is per-street and ignores multi-street aggression history, so it
    can over-estimate equity in spots where opp's hand-wide aggression
    implies a tight value range (e.g. river vs ck-raise-flop opponent).
    The tier selector reads hand-wide aggression signals and gives the
    realistic equity ceiling in those spots; capping the DB by the tier
    prevents the model from seeing over-confident threat-range values.
    """
    eq_tier = None
    if tier_safety_floor:
        eq_tier = compute_threat_range_equity(
            hole_cards, community_cards, tier, n_samples)

    if (board_aware and db_lookup is not None and spot_key is not None
            and community_cards and _TREYS_AVAILABLE):
        eq = db_lookup.equity_board_aware(
            hole_cards, community_cards, spot_key, n_samples=n_samples)
        if eq is not None:
            return min(eq, eq_tier) if eq_tier is not None else eq
    if db_lookup is not None and spot_key is not None:
        eq = db_lookup.equity(hole_cards, community_cards, spot_key,
                                n_samples=n_samples)
        if eq is not None:
            return min(eq, eq_tier) if eq_tier is not None else eq
    if eq_tier is not None:
        return eq_tier
    return compute_threat_range_equity(hole_cards, community_cards, tier,
                                         n_samples)


# ──────────────────────────────────────────────────────────────────────
# Board-aware range filtering
# ──────────────────────────────────────────────────────────────────────


def _class_to_combos(hand_class: str) -> List[Tuple[str, str]]:
    """Expand a 169-class string into card-string pairs.

    'AA'  → 6 combos (e.g. ('As','Ah'), ...)
    'AKs' → 4 combos (one per suit)
    'AKo' → 12 combos
    """
    if len(hand_class) < 2:
        return []
    r1, r2 = hand_class[0], hand_class[1]
    if r1 == r2:
        # Pair: enumerate unique 2-card suit combos
        return [(r1 + _SUITS[i], r1 + _SUITS[j])
                for i in range(4) for j in range(i + 1, 4)]
    if len(hand_class) == 3 and hand_class[2] == 's':
        return [(r1 + s, r2 + s) for s in _SUITS]
    # offsuit ('o' or missing)
    return [(r1 + s1, r2 + s2)
            for s1 in _SUITS for s2 in _SUITS if s1 != s2]


def _combo_fit_score(combo: Tuple[str, str],
                     board: List[str],
                     board_set: set,
                     board_ranks: List[int],
                     board_suit_counts: dict,
                     top_board_rank: int,
                     treys_board) -> float:
    """How well does this 2-card combo fit the board?

    Score scale:
      2.0  — made hand (top pair / overpair / two pair / set / straight / flush+)
      1.5  — combo draw (flush draw + straight draw, or weak made + draw)
      1.3  — flush draw
      1.2  — underpair or weak pair (uses board pair below top)
      1.0  — open-ended straight draw
      0.7  — gutshot
      0.5  — overcards only (both cards above top board rank)
      0.1  — total air
    """
    c1, c2 = combo
    if c1 in board_set or c2 in board_set:
        return 0.0  # conflict — combo impossible

    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    v1, v2 = _RANK_VAL[r1], _RANK_VAL[r2]

    # treys eval for made-hand classification
    th = [_TreysCard.new(c[0] + c[1].lower()) for c in combo]
    rank_int = _TREYS_EVALUATOR.evaluate(treys_board, th)
    rank_class = _TREYS_EVALUATOR.get_rank_class(rank_int)

    # Trips or better, two pair → strong made
    if rank_class <= 7:
        return 2.0

    # rank_class == 8: one pair. Distinguish top pair vs weaker.
    if rank_class == 8:
        # pocket pair?
        if v1 == v2:
            if v1 >= top_board_rank:
                return 2.0  # overpair
            return 1.2      # underpair
        # Pair uses a board card — check if top pair
        if v1 == top_board_rank or v2 == top_board_rank:
            return 2.0      # top pair (any kicker)
        return 1.2          # middle / bottom pair

    # rank_class == 9: high card → look for draws
    has_fd = _has_flush_draw_combo(s1, s2, board_suit_counts)
    has_oesd, has_gutshot = _has_straight_draw_combo(v1, v2, board_ranks)

    if has_fd and has_oesd:
        return 1.5
    if has_fd:
        return 1.3
    if has_oesd:
        return 1.0
    if has_gutshot:
        return 0.7

    # Overcards
    if v1 > top_board_rank and v2 > top_board_rank:
        return 0.5

    return 0.1


def _has_flush_draw_combo(s1: str, s2: str, board_suit_counts: dict) -> bool:
    """4 to a flush (either own suit appears 3 times on board, OR suited
    hand with 2 on board)."""
    if s1 == s2:
        # Suited: 2 own + N board same suit; FD requires ≥2 on board
        return board_suit_counts.get(s1, 0) >= 2
    # Offsuit: each card individually needs 3 board cards of that suit
    return (board_suit_counts.get(s1, 0) >= 3
            or board_suit_counts.get(s2, 0) >= 3)


def _has_straight_draw_combo(v1: int, v2: int,
                              board_ranks: List[int]) -> Tuple[bool, bool]:
    """Detect open-ended straight draw (OESD) or gutshot.

    Uses the union of hole + board ranks, checks for 4-in-a-row (OESD)
    or 4-of-5 with one gap (gutshot). Wheel-aware (A treated as 1 too).
    """
    ranks = set(board_ranks)
    ranks.add(v1)
    ranks.add(v2)
    if 12 in ranks:        # Ace can also be low
        ranks.add(-1)

    oesd = False
    gutshot = False
    # Scan windows of 5 consecutive rank values
    for lo in range(-1, 9):  # -1..8 (windows from -1..3 to 8..12)
        # Count ranks in [lo, lo+4]
        present = sum(1 for r in range(lo, lo + 5) if r in ranks)
        if present >= 4:
            # 4 present in a 5-window → at most 1 gap → straight draw
            # OESD if the gap is at either end (i.e. 4 consecutive)
            # → check 4-window
            consecutive_4 = any(
                all(r in ranks for r in range(lo + k, lo + k + 4))
                for k in range(2)
            )
            if consecutive_4:
                # but need to confirm both hole cards involved (otherwise
                # already on board → not a draw for THIS hand)
                if v1 in ranks and v2 in ranks:
                    # at least one hole card must be inside the window
                    if (lo <= v1 <= lo + 4) or (lo <= v2 <= lo + 4):
                        oesd = True
                        break
            elif (lo <= v1 <= lo + 4) or (lo <= v2 <= lo + 4):
                gutshot = True
    return oesd, gutshot


def _hand_class_fit_score(hand_class: str,
                           board: List[str]) -> float:
    """Average combo fit score for a 169-class on the given board.

    Returns 0.0..2.0. Higher = combo class fits the board well (made
    hands + strong draws).
    """
    if not _TREYS_AVAILABLE or not board:
        return 1.0  # neutral score if can't evaluate
    combos = _class_to_combos(hand_class)
    if not combos:
        return 1.0
    board_set = set(board)
    board_ranks = [_RANK_VAL[c[0]] for c in board]
    top_rank = max(board_ranks)
    board_suit_counts = {}
    for c in board:
        s = c[1]
        board_suit_counts[s] = board_suit_counts.get(s, 0) + 1
    try:
        treys_board = [_TreysCard.new(c[0] + c[1].lower()) for c in board]
    except Exception:
        return 1.0

    scores = []
    for combo in combos:
        sc = _combo_fit_score(combo, board, board_set, board_ranks,
                               board_suit_counts, top_rank, treys_board)
        if sc > 0:
            scores.append(sc)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
