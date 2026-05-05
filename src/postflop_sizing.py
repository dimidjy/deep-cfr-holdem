"""Runtime postflop sizing lookup.

When the model picks RAISE_HALF/RAISE_POT/ALL_IN on a postflop street, this
module queries cohort empirical sizes and returns:
  - a possibly-modified action (e.g. RAISE_POT promoted to ALL_IN when
    cohort shoves the spot frequently), AND
  - a custom_size_bb to type into the bet input field.

If the model didn't pick a raise, or the spot has no cohort data, this
module returns the input action unchanged with no custom size — the action
flows through the existing preset-size path.

The module is purely a sizing-override layer. It never overrides the
strategic decision (FOLD vs CALL vs raise) — only the size of the chosen
raise (and optionally promotes a smaller raise to all-in).
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.postflop_db import connect, load_lookup_into_memory
from src.postflop_spot_key import (
    PostflopSpotKey, MIN_SAMPLES_PER_SPOT, build_spot_key,
)

logger = logging.getLogger('postflop_sizing')

# Action constants (mirror src.agent.game_state to avoid circular imports)
FOLD = 0
CALL = 1
RAISE_HALF = 2
RAISE_POT = 3
ALL_IN = 4
RAISE_MIN = 5
RAISE_CUSTOM = 8

_RAISE_ACTIONS = (RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN)


def _round_to_half_bb(size_bb: float, min_bb: float = 2.0) -> float:
    """Round a raise size to nearest 0.5 BB and floor at min_bb.

    Same rounding as preflop typing (mirrors src/preflop_lookup._round_to_half_bb)
    so the agent's typed values are consistently clean.
    """
    rounded = round(size_bb * 2) / 2
    return max(rounded, min_bb)


class PostflopSizingLookup:
    """Data-driven postflop sizing layer.

    Parameters
    ----------
    db_path : str
        Path to the SQLite postflop sizing DB built by
        `scripts/build_postflop_lookup.py`.
    min_samples : int
        Minimum cohort sample count to trust the sizing percentiles.
        Spots below this leave the model's preset choice unchanged.
    all_in_promotion_threshold : float | Dict[str, float]
        Cohort shove% (0.0-1.0) at or above which we promote the model's
        raise choice to ALL_IN. Either a scalar (applied to all spots) or
        a dict keyed by SPR bucket (`'<2'`, `'2-5'`, `'5-15'`, `'15+'`).
        Per-SPR is preferred because the cohort all_in_pct distribution is
        bimodal (SPR<2 averages ~72%, SPR≥5 averages <3%) — a single scalar
        either over-fires deep or under-fires short.
    min_samples_for_promotion : int
        Sample floor for the binary promotion check, decoupled from
        `min_samples` (which gates sizing percentiles). Higher because
        all_in_pct is high-variance — at n=5 a single decision swings the
        rate by 20pp. Default 20.
    """

    def __init__(self, db_path: str,
                 min_samples: int = MIN_SAMPLES_PER_SPOT,
                 all_in_promotion_threshold: Union[float, Dict[str, float]] = 0.35,
                 min_samples_for_promotion: int = 20):
        if not Path(db_path).exists():
            raise FileNotFoundError(
                f'Postflop sizing DB not found at {db_path}. '
                f'Run scripts/build_postflop_lookup.py to create it.')
        self._min_samples = min_samples
        self._all_in_threshold = all_in_promotion_threshold
        self._min_samples_for_promotion = min_samples_for_promotion
        conn = connect(db_path)
        try:
            self._cache: Dict[Tuple, Dict] = load_lookup_into_memory(conn)
        finally:
            conn.close()
        self._counters = {
            'queries': 0, 'no_data': 0,
            'sizing_override': 0, 'all_in_promotions': 0,
            'promo_below_sample_floor': 0,
            'non_postflop': 0, 'non_raise': 0, 'error_fallback': 0,
        }
        # Periodic-summary cadence — emit one INFO line every N decisions so
        # the user can confirm at a glance that the lookup is firing in
        # production. Mirrors the preflop_lookup pattern.
        self._log_every = 50

        logger.info(
            f"PostflopSizingLookup initialized: db={db_path}, "
            f"spots={len(self._cache)}, min_samples={min_samples}, "
            f"min_samples_for_promotion={min_samples_for_promotion}, "
            f"all_in_threshold={all_in_promotion_threshold}")

    def _threshold_for_spr(self, spr_bucket: str) -> float:
        """Resolve promotion threshold for the given SPR bucket."""
        thr = self._all_in_threshold
        if isinstance(thr, dict):
            # Fall back to a strict 1.01 (unreachable) if bucket missing —
            # safer than defaulting low and accidentally promoting unknown SPRs.
            return float(thr.get(spr_bucket, 1.01))
        return float(thr)

    @property
    def num_spots(self) -> int:
        return len(self._cache)

    @property
    def counters(self) -> Dict[str, int]:
        return dict(self._counters)

    def reset_counters(self) -> None:
        for k in self._counters:
            self._counters[k] = 0

    def _emit_summary_if_due(self) -> None:
        """Emit a periodic INFO line every `_log_every` decisions so the
        user can verify the lookup is firing in production logs."""
        c = self._counters
        total_decisions = (c['queries'] + c['non_postflop']
                           + c['non_raise'] + c['error_fallback'])
        if total_decisions > 0 and total_decisions % self._log_every == 0:
            logger.info(
                f"PostflopSizing usage (last {total_decisions} decisions): "
                f"queries={c['queries']} sized={c['sizing_override']} "
                f"all_in_promo={c['all_in_promotions']} "
                f"no_data={c['no_data']} non_postflop={c['non_postflop']} "
                f"non_raise={c['non_raise']}")

    def adjust(self, action_idx: int, spot_inputs: Dict
               ) -> Tuple[int, Optional[float]]:
        """Given the model's action, query cohort sizing.

        spot_inputs must contain all the keyword args build_spot_key needs:
            street, obs, hole_cards, community_cards, rel_pos, num_active,
            effective_stack_bb, pot_bb, was_preflop_raiser,
            facing_bet_size_pct (None if not facing bet),
            facing_check_raise (default False),
            num_to_act_after_hero_postflop (optional).
        Plus:
            call_amount_bb: BB amount hero needs to put in to call.

        Returns (possibly-modified action_idx, custom_size_bb_or_None).
        - All-in promotion: action becomes ALL_IN with size = effective_stack_bb
        - Sizing override: action unchanged, size from cohort percentile
        - No data / non-applicable: action unchanged, size = None
        """
        if action_idx not in _RAISE_ACTIONS:
            self._counters['non_raise'] += 1
            self._emit_summary_if_due()
            return action_idx, None

        street = spot_inputs.get('street')
        if street not in ('flop', 'turn', 'river'):
            self._counters['non_postflop'] += 1
            self._emit_summary_if_due()
            return action_idx, None

        try:
            spot = build_spot_key(
                street=street,
                obs=spot_inputs['obs'],
                hole_cards=spot_inputs.get('hole_cards'),
                community_cards=spot_inputs.get('community_cards'),
                rel_pos=spot_inputs.get('rel_pos', 0),
                num_active=spot_inputs.get('num_active', 6),
                effective_stack_bb=spot_inputs['effective_stack_bb'],
                pot_bb=spot_inputs['pot_bb'],
                was_preflop_raiser=spot_inputs.get('was_preflop_raiser', False),
                facing_bet_size_pct=spot_inputs.get('facing_bet_size_pct'),
                facing_check_raise=spot_inputs.get('facing_check_raise', False),
                num_to_act_after_hero_postflop=spot_inputs.get(
                    'num_to_act_after_hero_postflop'),
            )
        except Exception as e:
            self._counters['error_fallback'] += 1
            logger.warning(f"PostflopSizingLookup: spot-key build failed: {e}")
            self._emit_summary_if_due()
            return action_idx, None

        self._counters['queries'] += 1
        row = self._cache.get(spot.to_tuple())
        if row is None or row['n_samples'] < self._min_samples:
            self._counters['no_data'] += 1
            n_have = row['n_samples'] if row else 0
            logger.info(
                f"Postflop sizing NO_DATA: spot={spot.to_tuple()} "
                f"n_samples={n_have} < {self._min_samples} → preset size")
            self._emit_summary_if_due()
            return action_idx, None

        pot_bb = spot_inputs['pot_bb']
        call_amount_bb = spot_inputs.get('call_amount_bb', 0.0)
        effective_stack_bb = spot_inputs['effective_stack_bb']

        # ── All-in promotion ─────────────────────────────────────────────
        # Two gates: (1) per-SPR threshold (cohort shove rate at/above which
        # we copy them) and (2) a separate sample floor — the binary
        # all_in_pct is high-variance, so we require more data than the
        # sizing-percentile path uses.
        all_in_pct_frac = float(row['all_in_pct']) / 100.0
        spr_threshold = self._threshold_for_spr(spot.spr_bucket)
        if all_in_pct_frac >= spr_threshold:
            if row['n_samples'] < self._min_samples_for_promotion:
                self._counters['promo_below_sample_floor'] += 1
                logger.info(
                    f"Postflop ALL-IN promotion SUPPRESSED (low samples): "
                    f"{spot.to_tuple()} cohort_shove={row['all_in_pct']:.0f}% "
                    f"n={row['n_samples']} < floor={self._min_samples_for_promotion}")
                # Fall through to sizing-override path below.
            else:
                self._counters['all_in_promotions'] += 1
                logger.info(
                    f"Postflop ALL-IN promotion: {spot.to_tuple()} "
                    f"(cohort shove%={row['all_in_pct']:.0f}% "
                    f">= SPR{spot.spr_bucket} threshold {spr_threshold*100:.0f}%, "
                    f"n={row['n_samples']})")
                self._emit_summary_if_due()
                return ALL_IN, _round_to_half_bb(effective_stack_bb,
                                                 min_bb=pot_bb * 0.10)

        # ── Sizing override ─────────────────────────────────────────────
        # Special case: model picked ALL_IN but cohort didn't promote here.
        # Don't downsize the model's strategic intent — type the effective
        # stack so the bet input gets the shove value typed correctly.
        if action_idx == ALL_IN:
            target_bb = _round_to_half_bb(effective_stack_bb,
                                          min_bb=max(2.0, pot_bb * 0.10))
            self._counters['sizing_override'] += 1
            logger.info(
                f"Postflop ALL-IN typed: {spot.to_tuple()} "
                f"→ {target_bb:.1f}bb (cohort shove% = {row['all_in_pct']:.0f}% "
                f"below SPR{spot.spr_bucket} threshold {spr_threshold*100:.0f}%, "
                f"n={row['n_samples']})")
            self._emit_summary_if_due()
            return action_idx, target_bb

        target_pct = _pick_target_pct(action_idx, row)
        if target_pct is None:
            self._counters['no_data'] += 1
            self._emit_summary_if_due()
            return action_idx, None

        # bet/raise target = target_pct * pot + call_amount (if facing bet,
        # we need to call first then add pct*pot on top).
        post_call_pot = pot_bb + call_amount_bb
        target_bb = call_amount_bb + target_pct * post_call_pot

        # Clamp to [min_raise, effective_stack]
        min_raise_bb = max(2.0, 2.0 * call_amount_bb) if call_amount_bb > 0 else 2.0
        target_bb = max(min_raise_bb, min(target_bb, effective_stack_bb))
        target_bb = _round_to_half_bb(target_bb, min_bb=min_raise_bb)

        self._counters['sizing_override'] += 1
        logger.info(
            f"Postflop sizing OVERRIDE: {spot.to_tuple()} "
            f"action={action_idx} target_pct={target_pct:.2f} → {target_bb:.1f}bb "
            f"(pot={pot_bb:.1f}bb, call={call_amount_bb:.1f}bb, "
            f"n={row['n_samples']})")
        self._emit_summary_if_due()
        return action_idx, target_bb


def _pick_target_pct(action_idx: int, row: Dict) -> Optional[float]:
    """Map model's raise category → cohort percentile.

    Per design doc:
      RAISE_HALF (or RAISE_MIN) → p25 (smaller bet)
      RAISE_POT                 → p75 (bigger bet)
      ALL_IN                    → p75 (large bet — but if all_in_pct didn't
                                       trigger promotion, fall back to p75)
    """
    if action_idx in (RAISE_HALF, RAISE_MIN):
        target = row.get('bet_size_pct_p25')
    elif action_idx == RAISE_POT:
        target = row.get('bet_size_pct_p75')
    elif action_idx == ALL_IN:
        # If we got here, all_in_pct didn't trigger promotion. Use p75 as
        # the "large bet" approximation.
        target = row.get('bet_size_pct_p75')
    else:
        return None
    return float(target) if target is not None else None
