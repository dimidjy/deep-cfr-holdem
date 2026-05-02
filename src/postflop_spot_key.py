"""Shared spot-key derivation for the postflop sizing lookup.

Single source of truth used by both the offline build script
(`scripts/build_postflop_lookup.py`) and the runtime sizing module
(`src/postflop_sizing.py`). Build/runtime divergence on key derivation
is the #1 production-bug source — keep ALL key construction here.

A spot is identified by 7 fields:

    street          'flop' / 'turn' / 'river'
    hand_bucket     11 categories from high_card → full_plus
    draw_type       6 categories: none/flush_draw/oesd/gutshot/combo_draw/made_hand
    board_texture   5 categories: dry/two_tone/paired/monotone/wet_paired
    position        IP / OOP   (collapsed from rel_pos)
    spr_bucket      <2 / 2-5 / 5-15 / 15+
    action_context  cbet/probe/donk/face_bet_small/face_bet_med/face_bet_large/face_check_raise

Total spot space ≈ 55K — dense enough on a 20M-hand archive.
"""
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# ─── Constants reused from decision_engine for consistency ──────────────
# (We re-declare here to avoid importing decision_engine in the build path.)
IDX_MADE_HAND = 19
IDX_FLUSH_DRAW = 20
IDX_STRAIGHT_DRAW = 21
IDX_TOP_PAIR = 22
IDX_BOARD_PAIR = 23
IDX_BOARD_FLUSH = 24
IDX_BOARD_STRAIGHT = 25
IDX_RANK_WITHIN_CAT = 26

HAND_HIGH_CARD = 0.0
HAND_ONE_PAIR = 0.125
HAND_TWO_PAIR = 0.25
HAND_TRIPS = 0.375
HAND_STRAIGHT = 0.5
HAND_FLUSH = 0.625
HAND_FULL_HOUSE = 0.75
HAND_QUADS = 0.875

# Sample-count thresholds at the runtime lookup (tunable per-instance)
MIN_SAMPLES_PER_SPOT = 5

# Hand-bucket enum values
HB_HIGH_CARD = 'high_card'
HB_LOW_PAIR = 'low_pair'
HB_MID_PAIR = 'mid_pair'
HB_TOP_PAIR_LOW = 'top_pair_low'
HB_TOP_PAIR_HIGH = 'top_pair_high'
HB_OVERPAIR = 'overpair'
HB_TWO_PAIR = 'two_pair'
HB_TRIPS_SET = 'trips_set'
HB_STRAIGHT = 'straight'
HB_FLUSH = 'flush'
HB_FULL_PLUS = 'full_plus'
HAND_BUCKET_VALUES = (
    HB_HIGH_CARD, HB_LOW_PAIR, HB_MID_PAIR, HB_TOP_PAIR_LOW,
    HB_TOP_PAIR_HIGH, HB_OVERPAIR, HB_TWO_PAIR, HB_TRIPS_SET,
    HB_STRAIGHT, HB_FLUSH, HB_FULL_PLUS,
)

# Draw-type enum values
DT_NONE = 'none'
DT_FLUSH_DRAW = 'flush_draw'
DT_OESD = 'oesd'
DT_GUTSHOT = 'gutshot'
DT_COMBO_DRAW = 'combo_draw'
DT_MADE_HAND = 'made_hand'
DRAW_TYPE_VALUES = (
    DT_NONE, DT_FLUSH_DRAW, DT_OESD, DT_GUTSHOT, DT_COMBO_DRAW, DT_MADE_HAND,
)

# Board-texture enum values
BT_DRY = 'dry'
BT_TWO_TONE = 'two_tone'
BT_PAIRED = 'paired'
BT_MONOTONE = 'monotone'
BT_WET_PAIRED = 'wet_paired'
BOARD_TEXTURE_VALUES = (
    BT_DRY, BT_TWO_TONE, BT_PAIRED, BT_MONOTONE, BT_WET_PAIRED,
)

# Action-context enum values
AC_CBET = 'cbet'
AC_PROBE = 'probe'
AC_DONK = 'donk'
AC_FACE_BET_SMALL = 'face_bet_small'
AC_FACE_BET_MED = 'face_bet_med'
AC_FACE_BET_LARGE = 'face_bet_large'
AC_FACE_CHECK_RAISE = 'face_check_raise'
ACTION_CONTEXT_VALUES = (
    AC_CBET, AC_PROBE, AC_DONK,
    AC_FACE_BET_SMALL, AC_FACE_BET_MED, AC_FACE_BET_LARGE,
    AC_FACE_CHECK_RAISE,
)

SPR_BUCKETS = (
    (2.0, '<2'),
    (5.0, '2-5'),
    (15.0, '5-15'),
    (float('inf'), '15+'),
)


@dataclass(frozen=True)
class PostflopSpotKey:
    street: str
    hand_bucket: str
    draw_type: str
    board_texture: str
    position: str
    spr_bucket: str
    action_context: str

    def to_tuple(self) -> Tuple[str, str, str, str, str, str, str]:
        return (self.street, self.hand_bucket, self.draw_type,
                self.board_texture, self.position, self.spr_bucket,
                self.action_context)


# ─── Bucket classifiers ─────────────────────────────────────────────────

def _approx(a: float, b: float, tol: float = 0.05) -> bool:
    return abs(a - b) <= tol


def classify_hand_bucket(obs) -> str:
    """11-category hand bucket from obs features."""
    cat = float(obs[IDX_MADE_HAND])
    top = float(obs[IDX_TOP_PAIR])
    rank = float(obs[IDX_RANK_WITHIN_CAT])

    if cat >= HAND_FULL_HOUSE - 0.01:
        return HB_FULL_PLUS
    if cat >= HAND_FLUSH - 0.01:
        return HB_FLUSH
    if cat >= HAND_STRAIGHT - 0.01:
        return HB_STRAIGHT
    if cat >= HAND_TRIPS - 0.01:
        return HB_TRIPS_SET
    if cat >= HAND_TWO_PAIR - 0.01:
        return HB_TWO_PAIR
    if cat >= HAND_ONE_PAIR - 0.01:
        # PAIR — distinguish top_pair / overpair / mid_pair / low_pair
        # using top_pair flag and rank_within_cat
        if top > 0.5:
            return HB_TOP_PAIR_HIGH if rank >= 0.5 else HB_TOP_PAIR_LOW
        # No top pair: must be pocket pair (overpair or under-pair) or
        # bottom/middle pair from a board-card. rank_within_cat is the
        # best signal we have without raw cards.
        if rank >= 0.6:
            return HB_OVERPAIR
        if rank >= 0.3:
            return HB_MID_PAIR
        return HB_LOW_PAIR
    return HB_HIGH_CARD


def _has_flush_draw(hole: List[str], community: List[str]) -> bool:
    """Hero has 4+ to a flush, not yet completed."""
    if len(community) < 3 or len(hole) < 2:
        return False
    suits = [c[1].lower() for c in hole + community]
    counts = {}
    for s in suits:
        counts[s] = counts.get(s, 0) + 1
    # 4 to a flush AND we have at least 1 of that suit in our hole
    hero_suits = {c[1].lower() for c in hole}
    for s, n in counts.items():
        if n == 4 and s in hero_suits:
            return True
    return False


_RANK_ORDER = '23456789TJQKA'


def _rank_idx(card: str) -> int:
    return _RANK_ORDER.index(card[0].upper())


def _has_oesd_or_gutshot(hole: List[str], community: List[str]
                         ) -> Tuple[bool, bool]:
    """Returns (has_oesd, has_gutshot). OESD = 8 outs, gutshot = 4 outs.

    Skips river entirely (no draws on river by definition).
    """
    if len(community) < 3 or len(community) >= 5 or len(hole) < 2:
        return (False, False)
    # Build sorted set of unique rank indices across hole + board
    ranks = sorted({_rank_idx(c) for c in hole + community})
    hero_rank_set = {_rank_idx(c) for c in hole}

    best_outs = 0
    # For every possible 5-card straight target (5 consecutive ranks),
    # count how many of those ranks are already covered.
    # If 4 covered, hero needs 1 → straight draw. Whether it's OESD or
    # gutshot depends on whether the missing rank is at the end (gutshot
    # if interior, OESD if end). Easier: count straight outs across all
    # possible target straights — if any target gives 8+ unique outs → OESD.
    outs_by_target: List[int] = []
    # Include wheel A-2-3-4-5 (Ace as 1)
    for low in range(0, 11):
        target = set(range(low, low + 5))
        covered = target & set(ranks)
        if len(covered) == 4:
            # We're 1 card away from this straight — count outs (= 1 rank)
            missing = (target - set(ranks)).pop()
            # 4 outs for this straight (4 cards of the missing rank)
            outs_by_target.append(missing)
    # OESD also considers wheel if A is in hand
    if 12 in ranks:  # Ace
        target = {0, 1, 2, 3, 12}
        covered = target & set(ranks)
        if len(covered) == 4:
            outs_by_target.append((target - set(ranks)).pop())

    # Number of distinct missing-rank slots at distance 1 = number of out-ranks
    # OESD: 2 missing ranks (at both ends) = 8 outs
    # Gutshot: 1 missing rank = 4 outs
    n_targets = len(set(outs_by_target))
    has_draw = n_targets >= 1
    if not has_draw:
        return (False, False)
    # Hero must use at least one card from hole to qualify
    # (otherwise it's a board-only draw, not hero's draw)
    target_uses_hero = False
    for low in range(-4, 13):
        target = set(range(max(0, low), min(13, low + 5)))
        if len(target) < 5:
            continue
        if len(target - set(ranks)) == 1 and target & hero_rank_set:
            target_uses_hero = True
            break
    if not target_uses_hero:
        return (False, False)
    if n_targets >= 2:
        return (True, False)   # OESD (8 outs)
    return (False, True)       # gutshot (4 outs)


def classify_draw_type(obs, hole_cards: Optional[List[str]],
                       community_cards: Optional[List[str]]) -> str:
    """6-category draw type. Made hands return 'made_hand'.

    obs: feature vector (used for the made-hand short-circuit)
    hole_cards / community_cards: optional. When provided, distinguish
    OESD from gutshot precisely. When None, gutshots collapse to OESD.
    """
    cat = float(obs[IDX_MADE_HAND])
    if cat >= HAND_STRAIGHT - 0.01:
        return DT_MADE_HAND   # straight or better — drawing is moot

    # On river, no draws by definition
    if community_cards and len(community_cards) >= 5:
        return DT_NONE

    has_flush_draw = False
    has_oesd = False
    has_gutshot = False
    if hole_cards and community_cards:
        has_flush_draw = _has_flush_draw(hole_cards, community_cards)
        has_oesd, has_gutshot = _has_oesd_or_gutshot(hole_cards, community_cards)
    else:
        # Fallback: feature-based bucketing (collapses gutshot+oesd)
        has_flush_draw = float(obs[IDX_FLUSH_DRAW]) >= 0.5
        if float(obs[IDX_STRAIGHT_DRAW]) >= 0.5:
            has_oesd = True   # cannot distinguish from gutshot

    has_straight_draw = has_oesd or has_gutshot
    if has_flush_draw and has_straight_draw:
        return DT_COMBO_DRAW
    if has_flush_draw:
        return DT_FLUSH_DRAW
    if has_oesd:
        return DT_OESD
    if has_gutshot:
        return DT_GUTSHOT
    return DT_NONE


def classify_board_texture(obs) -> str:
    """5-category board texture from obs features."""
    paired = float(obs[IDX_BOARD_PAIR]) > 0.1
    flush_tx = float(obs[IDX_BOARD_FLUSH])
    straight_tx = float(obs[IDX_BOARD_STRAIGHT])

    monotone = flush_tx >= 0.95
    has_flush_texture = 0.33 <= flush_tx < 0.95
    has_straight_texture = straight_tx >= 0.4
    has_draws = has_flush_texture or has_straight_texture

    if monotone:
        return BT_MONOTONE
    if paired and has_draws:
        return BT_WET_PAIRED
    if paired:
        return BT_PAIRED
    if has_draws:
        return BT_TWO_TONE
    return BT_DRY


def classify_position(rel_pos: int, num_active: int,
                      num_to_act_after_hero_postflop: Optional[int] = None
                      ) -> str:
    """IP if hero acts last on the street, OOP otherwise.

    When `num_to_act_after_hero_postflop` is provided (preferred), use it
    directly. Otherwise infer from rel_pos: BTN (rel_pos=0) is IP, others
    OOP. Note that postflop action order is SB→BB→...→BTN; rel_pos=0
    (BTN) acts last when present.
    """
    if num_to_act_after_hero_postflop is not None:
        return 'IP' if num_to_act_after_hero_postflop == 0 else 'OOP'
    return 'IP' if rel_pos == 0 else 'OOP'


def classify_spr(effective_stack_bb: float, pot_bb: float) -> str:
    """SPR bucket. Postflop pot must be > 0; else clamp to 15+."""
    if pot_bb <= 0.01:
        return '15+'
    spr = effective_stack_bb / pot_bb
    for upper, label in SPR_BUCKETS:
        if spr < upper:
            return label
    return SPR_BUCKETS[-1][1]


def classify_action_context(was_preflop_raiser: bool,
                            position_is_oop: bool,
                            facing_bet_size_pct: Optional[float],
                            facing_check_raise: bool,
                            ) -> str:
    """Determine action context from running street state.

    Args:
        was_preflop_raiser: True if hero was the last preflop aggressor
        position_is_oop: True if hero is OOP this street
        facing_bet_size_pct: bet size as fraction of pot when hero faces
            a bet (None when hero is first to act / not facing a bet)
        facing_check_raise: True when hero bet, opp raised, hero now acting
    """
    if facing_check_raise:
        return AC_FACE_CHECK_RAISE
    if facing_bet_size_pct is not None:
        if facing_bet_size_pct <= 0.40:
            return AC_FACE_BET_SMALL
        if facing_bet_size_pct <= 0.75:
            return AC_FACE_BET_MED
        return AC_FACE_BET_LARGE
    # No bet on the table yet — either first to act, or facing a check
    if was_preflop_raiser:
        return AC_CBET
    if position_is_oop:
        return AC_DONK
    return AC_PROBE


def build_spot_key(*, street: str, obs, hole_cards: Optional[List[str]],
                   community_cards: Optional[List[str]],
                   rel_pos: int, num_active: int,
                   effective_stack_bb: float, pot_bb: float,
                   was_preflop_raiser: bool,
                   facing_bet_size_pct: Optional[float],
                   facing_check_raise: bool = False,
                   num_to_act_after_hero_postflop: Optional[int] = None,
                   ) -> PostflopSpotKey:
    """One-stop key constructor used by both build-time and runtime."""
    hand_bucket = classify_hand_bucket(obs)
    draw_type = classify_draw_type(obs, hole_cards, community_cards)
    board_texture = classify_board_texture(obs)
    position = classify_position(rel_pos, num_active,
                                 num_to_act_after_hero_postflop)
    spr_bucket = classify_spr(effective_stack_bb, pot_bb)
    action_context = classify_action_context(
        was_preflop_raiser=was_preflop_raiser,
        position_is_oop=(position == 'OOP'),
        facing_bet_size_pct=facing_bet_size_pct,
        facing_check_raise=facing_check_raise,
    )
    return PostflopSpotKey(
        street=street,
        hand_bucket=hand_bucket,
        draw_type=draw_type,
        board_texture=board_texture,
        position=position,
        spr_bucket=spr_bucket,
        action_context=action_context,
    )
