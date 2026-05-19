"""Spot key for the empirical opponent-range lookup.

Companion to `src/postflop_spot_key.py`. Where the postflop_sizing key
is *(spot + hero hand bucket) → sizing*, this key is *spot → opponent
hand-class distribution*. We drop hand_bucket and draw_type because the
hand IS what we store (per spot).

Spot fields (5):
    street          'flop' / 'turn' / 'river'
    board_texture   dry / two_tone / paired / monotone / wet_paired
    position        IP / OOP
    spr_bucket      <2 / 2-5 / 5-15 / 15+
    action_context  cbet / probe / donk / face_bet_small / ... / face_check_raise

Total spot space ≈ 3·5·2·4·7 ≈ 840 cells — dense on a 20M-hand archive.
"""
from dataclasses import dataclass
from typing import List, Optional

from src.postflop_spot_key import (
    classify_board_texture, classify_position,
    classify_spr, classify_action_context,
)

# Minimum showdown observations required at a spot before we trust the
# empirical range (vs. falling back to the hardcoded tier ranges).
MIN_SAMPLES_PER_SPOT = 50

_RANK_ORDER = '23456789TJQKA'


@dataclass(frozen=True)
class OppRangeSpotKey:
    street: str
    board_texture: str
    position: str
    spr_bucket: str
    action_context: str

    def as_tuple(self):
        return (self.street, self.board_texture, self.position,
                self.spr_bucket, self.action_context)


def build_opp_range_spot_key(*, street: str, obs,
                              rel_pos: int, num_active: int,
                              effective_stack_bb: float, pot_bb: float,
                              was_preflop_raiser: bool,
                              facing_bet_size_pct: Optional[float],
                              facing_check_raise: bool = False,
                              num_to_act_after_hero_postflop: Optional[int] = None
                              ) -> OppRangeSpotKey:
    """One-stop key constructor used by both build-time and runtime.

    Mirrors the signature of `postflop_spot_key.build_spot_key` minus
    `hole_cards` (no hand bucket / draw type needed here).
    """
    position = classify_position(rel_pos, num_active,
                                  num_to_act_after_hero_postflop)
    return OppRangeSpotKey(
        street=street,
        board_texture=classify_board_texture(obs),
        position=position,
        spr_bucket=classify_spr(effective_stack_bb, pot_bb),
        action_context=classify_action_context(
            was_preflop_raiser=was_preflop_raiser,
            position_is_oop=(position == 'OOP'),
            facing_bet_size_pct=facing_bet_size_pct,
            facing_check_raise=facing_check_raise,
        ),
    )


def hand_class_from_hole(hole_cards: List[str]) -> Optional[str]:
    """Convert hole_cards like ['Ks','Td'] → '169-class' string.

    Returns 'AA', 'KK' for pairs; 'AKs' for suited; 'AKo' for offsuit.
    Higher rank always first. Case-insensitive. Returns None if invalid.
    """
    if len(hole_cards) < 2:
        return None
    c1, c2 = hole_cards[0], hole_cards[1]
    if len(c1) < 2 or len(c2) < 2:
        return None
    r1, s1 = c1[0].upper(), c1[1].lower()
    r2, s2 = c2[0].upper(), c2[1].lower()
    if r1 not in _RANK_ORDER or r2 not in _RANK_ORDER:
        return None
    if r1 == r2:
        return r1 + r2
    # Order high → low.
    if _RANK_ORDER.index(r1) < _RANK_ORDER.index(r2):
        r1, r2 = r2, r1
    return r1 + r2 + ('s' if s1 == s2 else 'o')
