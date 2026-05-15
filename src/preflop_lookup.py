"""Runtime preflop strategy: data-driven lookup with hierarchical key fallback.

`PreflopLookup` is a drop-in replacement for `PreflopRanges` in the
DecisionEngine pipeline. At each preflop decision it consults the cohort
data at progressively coarser granularities; only true errors fall through
to the static-range fallback.

Lookup hierarchy (each level uses cohort data — just less granular):
  L0: (hole, position, action_history, stack_bucket, num_to_act)  [exact]
  L1: (hole, position, action_history, stack_bucket)              [drop nta]
  L2: (hole, position, action_history)                            [drop stack]
  L3: empty → 100% fold default

If ANY exception is raised during the lookup path, fall through to the
static-range fallback (`PreflopRanges`). The fallback is therefore reserved
for genuine error recovery (DB load failure, malformed spot key, etc.) —
NOT for sparse-spot recovery.

Returns `(strategy: np.ndarray, custom_size_bb: Optional[float])`.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.preflop_db import connect, load_lookup_into_memory
from src.preflop_spot_key import (
    MIN_SAMPLES_PER_SPOT, build_spot_key,
)

logger = logging.getLogger('preflop_lookup')


# Action constants (mirrored from src.agent.game_state to avoid
# circular dependency; if these go out of sync, runtime will break loudly)
FOLD = 0
CALL = 1
RAISE_HALF = 2
RAISE_POT = 3
ALL_IN = 4
RAISE_MIN = 5
RAISE_CUSTOM = 8  # agent-only override; carries explicit BB amount


def _percentiles(sorted_sizes: List[float]) -> Tuple[Optional[float], ...]:
    """Return (mean, p25, p50, p75) from a sorted list of raise sizes."""
    if not sorted_sizes:
        return (None, None, None, None)
    n = len(sorted_sizes)
    mean = sum(sorted_sizes) / n
    p25 = sorted_sizes[max(0, int(0.25 * n) - 1)]
    p50 = sorted_sizes[n // 2]
    p75 = sorted_sizes[min(n - 1, int(0.75 * n))]
    return (mean, p25, p50, p75)


def _round_to_half_bb(size_bb: float, min_bb: float = 2.0) -> float:
    """Round a raise size to the nearest 0.5 BB and floor at min_bb.

    Cohort medians like 2.42 BB get clamped to 2.5; 11.4 → 11.5; 23.83 → 24.0.
    Keeps the typed value clean without losing meaningful granularity.
    """
    rounded = round(size_bb * 2) / 2
    return max(rounded, min_bb)


class PreflopLookup:
    """Data-driven preflop strategy with hierarchical cohort fallback.

    Parameters
    ----------
    db_path : str
        Path to the SQLite lookup DB built by `scripts/build_preflop_lookup.py`.
    fallback : object
        A `PreflopRanges`-shaped fallback used ONLY for error recovery
        (exceptions during the lookup path). For sparse data, the lookup
        coarsens within the cohort data. May be None — empty/error spots
        then fall to fold default.
    min_samples : int
        Minimum cohort sample count for the EXACT spot key. Spots below
        this defer to coarser keys (NOT to the fallback). Defaults to
        MIN_SAMPLES_PER_SPOT.
    min_samples_coarse : int
        Floor for coarser-key matches. Default 10 — slightly higher than
        exact since coarser data has less spot-specific signal.
    """

    def __init__(self, db_path: str, fallback=None,
                 min_samples_exact: int = 5,
                 min_samples_no_nta: int = 3,
                 min_samples_no_stack: int = 5,
                 min_play_rate: float = 0.25,
                 min_play_rate_vs_3bet: Optional[float] = None,  # DEPRECATED
                 min_play_rate_vs_4bet: Optional[float] = None,  # DEPRECATED
                 l2_large_call_bb: float = 30.0,
                 l2_large_call_play_rate: float = 0.40,
                 pos_min_play_rate: Optional[Dict[str, float]] = None):
        """
        Thresholds tuned for showdown-only HH bias. Rationale:
        - L0 exact: need a few confirmations to trust precise spot strategy.
        - L1 (drop nta): aggregating across nta is mild info loss; 3 ok.
        - L2 (drop stack): aggregating across stack mixes shove vs small-3bet
          sizings, AND the showdown-only DB systematically overrepresents
          'play' over 'fold' for marginal hands. Require ≥5 to filter the
          n=1 noise (e.g. one cohort player limping 94o once becoming
          'always limp 94o' in the lookup). Premium hands with >5 samples
          at L2 still win.

        With dense cohort data (large remote archive), the higher levels
        usually win and these thresholds barely matter. With sparse cohort
        data (testing build), L2's threshold of 5 prevents single-sample
        showdown-bias from leaking into runtime strategy.

        Fallback policy:
        - L0/L1 with play_rate below `min_play_rate` → fold (the cohort
          *did* see this spot enough times to fold; absence of plays is
          signal).
        - L2 with play_rate below `min_play_rate` → static range fallback
          (L2 pools across stacks, so its distribution is less reliable;
          static GTO ranges are safer than blind fold).
        - L2 with call_amount ≥ `l2_large_call_bb` AND play_rate below
          `l2_large_call_play_rate` → static range fallback (stack-pooled
          rows are dangerous when the decision commits a lot of chips).
        - L3 (no data anywhere) → static range fallback if available,
          else fold default.
        """
        self._fallback = fallback
        self._min_samples = min_samples_exact
        self._min_samples_no_nta = min_samples_no_nta
        self._min_samples_no_stack = min_samples_no_stack
        self._min_play_rate = min_play_rate
        # `min_play_rate_vs_3bet` and `min_play_rate_vs_4bet` are accepted for
        # backwards compatibility with existing configs but no longer have
        # any effect — vs_3bet / vs_4bet+ spots now bypass cohort lookup
        # entirely (see `_static_action_cohort_size`). Warn if a non-default
        # value was passed so users know it's now a no-op.
        if min_play_rate_vs_3bet is not None or min_play_rate_vs_4bet is not None:
            logger.warning(
                "min_play_rate_vs_3bet/vs_4bet are deprecated and ignored; "
                "vs_3bet/vs_4bet+ spots use static ranges for action choice "
                "and cohort data only for raise sizing.")
        self._l2_large_call_bb = l2_large_call_bb
        self._l2_large_call_play_rate = l2_large_call_play_rate
        # Per-position play_rate gate override. When a spot's `spot.position`
        # is in this dict, use the dict value as the play_rate threshold
        # instead of the global `min_play_rate`. Use to tighten one position
        # (higher gate → folds more of the cohort's wider marginal band) or
        # widen another (lower gate → opens more marginal hands). Hands
        # missing from the dict use the global `min_play_rate`.
        self._pos_min_play_rate: Dict[str, float] = pos_min_play_rate or {}
        if not Path(db_path).exists():
            raise FileNotFoundError(
                f'Preflop lookup DB not found at {db_path}. '
                f'Run scripts/build_preflop_lookup.py to create it.')
        conn = connect(db_path)
        try:
            self._exact: Dict[Tuple, Dict] = load_lookup_into_memory(conn)
            self._level_no_nta, self._level_no_stack = (
                self._build_coarser_levels(conn))
            # Build AA reference index (per-combo count at each
            # (position, action_history, stack_bucket)). AA is universally
            # played → its per-combo count ≈ deal frequency. Other hands'
            # per-combo count divided by AA's gives the empirical play rate
            # — recovers P(played | dealt) from showdown-only data.
            self._aa_per_combo: Dict[Tuple[str, str, str], float] = (
                self._build_aa_reference(conn))
        finally:
            conn.close()

        # Per-level decision counters. Logged periodically so the user can
        # confirm the lookup is firing in production. Reset by reset_counters().
        # `static_action_cohort_size` counts the 3bet+ early-branch path
        # where the static range drives the action and cohort drives the size.
        self._counters = {
            'L0_exact': 0, 'L1_no_nta': 0, 'L2_no_stack': 0,
            'L3_fold_default': 0, 'L3_static_fallback': 0,
            'L2_static_fallback': 0, 'error_fallback': 0,
            'static_action_cohort_size': 0,
        }
        self._log_every = 50    # emit usage summary every N decisions
        logger.info(
            f"PreflopLookup initialized: db={db_path}, "
            f"exact_spots={len(self._exact)}, "
            f"L1_keys={len(self._level_no_nta)}, "
            f"L2_keys={len(self._level_no_stack)}, "
            f"AA_refs={len(self._aa_per_combo)}, "
            f"thresholds=L0:{min_samples_exact} L1:{min_samples_no_nta} "
            f"L2:{min_samples_no_stack} "
            f"play_rate(open/vs_raise)>={min_play_rate:.2f} "
            f"vs_3bet/vs_4bet+: static-action + cohort-size")
        if self._pos_min_play_rate:
            logger.info(
                f"PreflopLookup pos_min_play_rate={self._pos_min_play_rate} "
                f"(overrides global {min_play_rate:.2f} per position)")

    def reset_counters(self) -> None:
        for k in self._counters:
            self._counters[k] = 0

    @property
    def counters(self) -> Dict[str, int]:
        return dict(self._counters)

    def _log_decision(self, level: str, spot, n_samples: Optional[int],
                      result: Optional[Tuple[float, float, float]],
                      custom_size_bb: Optional[float],
                      extra: str = '') -> None:
        """DEBUG: per-decision trace; INFO: periodic summary."""
        self._counters[level] += 1
        if logger.isEnabledFor(logging.DEBUG):
            spot_repr = (f"{spot.hole_class}/{spot.position}/"
                         f"{spot.action_history}/{spot.stack_bucket}/nta={spot.num_to_act}")
            if result is not None:
                f, c, r = result
                size_str = f"{custom_size_bb:.2f}bb" if custom_size_bb else "preset"
                logger.debug(
                    f"PF[{level}] {spot_repr} n={n_samples} "
                    f"F/C/R={f:.0f}/{c:.0f}/{r:.0f} size={size_str}"
                    f"{(' ' + extra) if extra else ''}")
            else:
                logger.debug(f"PF[{level}] {spot_repr} → fold default"
                             f"{(' ' + extra) if extra else ''}")

        # Periodic summary at INFO so users see hits without DEBUG noise
        total = sum(self._counters.values())
        if total > 0 and total % self._log_every == 0:
            c = self._counters
            logger.info(
                f"PreflopLookup usage (last {total} decisions): "
                f"L0={c['L0_exact']} L1={c['L1_no_nta']} L2={c['L2_no_stack']} "
                f"fold_default={c['L3_fold_default']} "
                f"errors={c['error_fallback']}")

    # ─── Public API matching the existing PreflopRanges contract ────────

    def get_strategy(self, hole_cards: List[str], rel_pos: int,
                     pot_bb: float, legal_actions: List[int],
                     action_size: int = 6,
                     num_limpers: int = 0,
                     effective_stack: float = 100.0,
                     num_callers: int = 0,
                     # New parameters used by the lookup but ignored by the
                     # legacy PreflopRanges fallback.
                     num_players: int = 6,
                     action_history: Optional[str] = None,
                     num_to_act: int = 0,
                     call_amount_bb: float = 0.0,
                     ) -> Tuple[np.ndarray, Optional[float]]:
        """Return (strategy[action_size], custom_size_bb_or_None).

        Hierarchical lookup: exact spot → drop num_to_act → drop stack →
        fold default. Static range fallback fires ONLY on exception.
        """
        try:
            prior = self._synth_prior(action_history, num_limpers, pot_bb)
            spot = build_spot_key(
                hole_cards=hole_cards,
                rel_pos=rel_pos, num_players=num_players,
                prior_actions=prior,
                effective_bb=effective_stack,
                num_to_act=num_to_act,
            )

            # ── 3bet+ early branch: static action, cohort size ───────────
            # For 3bet / 3bet-call / 4bet+ spots, the cohort DB is too
            # noisy/exploitable for action selection (NL5 cohort calls 3bets
            # too wide; gives back the cohort EV edge). Hand-tuned static
            # ranges drive the action; the cohort is consulted ONLY for the
            # raise size (median BB) when a raise is chosen.
            from src.preflop_spot_key import HIST_VS_3BET, HIST_VS_4BET
            if spot.action_history in (HIST_VS_3BET, HIST_VS_4BET):
                return self._static_action_cohort_size(
                    spot, hole_cards, rel_pos, pot_bb, legal_actions,
                    action_size, num_limpers, effective_stack, num_callers,
                )

            # All paths below handle HIST_OPEN, HIST_LIMPED_POT, HIST_VS_RAISE
            # only — vs_3bet / vs_4bet+ took the early branch above.

            # L0 — exact spot key
            row = self._exact.get(spot.to_tuple())
            if row and row['n_samples'] >= self._min_samples:
                play_rate = self._play_rate(
                    spot.hole_class, spot.position, spot.action_history,
                    spot.stack_bucket, row['n_samples'], 'L0_exact')
                min_pr = self._pos_min_play_rate.get(
                    spot.position, self._min_play_rate)
                if play_rate < min_pr:
                    self._log_decision(
                        'L3_fold_default', spot, row['n_samples'],
                        None, None,
                        extra=(f"play_rate={play_rate*100:.1f}% "
                               f"< {min_pr*100:.0f}% (pos={spot.position})"))
                    return self._fold_default(legal_actions, action_size), None
                strategy, custom_size = self._row_to_strategy(
                    row, legal_actions, action_size, pot_bb, call_amount_bb,
                    effective_stack)
                self._log_decision(
                    'L0_exact', spot, row['n_samples'],
                    (row['fold_pct'], row['call_pct'], row['raise_pct']),
                    custom_size)
                return strategy, custom_size

            # L1 — drop num_to_act (same stack, same hist → sizing still trusted)
            row = self._level_no_nta.get(
                (spot.hole_class, spot.position,
                 spot.action_history, spot.stack_bucket))
            if row and row['n_samples'] >= self._min_samples_no_nta:
                play_rate = self._play_rate(
                    spot.hole_class, spot.position, spot.action_history,
                    spot.stack_bucket, row['n_samples'], 'L1_no_nta')
                min_pr = self._pos_min_play_rate.get(
                    spot.position, self._min_play_rate)
                if play_rate < min_pr:
                    self._log_decision(
                        'L3_fold_default', spot, row['n_samples'],
                        None, None,
                        extra=(f"play_rate={play_rate*100:.1f}% "
                               f"< {min_pr*100:.0f}% (pos={spot.position})"))
                    return self._fold_default(legal_actions, action_size), None
                strategy, custom_size = self._row_to_strategy(
                    row, legal_actions, action_size, pot_bb, call_amount_bb,
                    effective_stack, trust_size=True)
                self._log_decision(
                    'L1_no_nta', spot, row['n_samples'],
                    (row['fold_pct'], row['call_pct'], row['raise_pct']),
                    custom_size)
                return strategy, custom_size

            # L2 — drop stack_bucket too. Sizing across stack depths is
            # unreliable (e.g. 3bet vs short-stacker is shove; vs deep-stacker
            # is small). Use the action distribution but skip custom sizing —
            # the raise mass routes to nearest preset instead.
            row = self._level_no_stack.get(
                (spot.hole_class, spot.position, spot.action_history))
            if row and row['n_samples'] >= self._min_samples_no_stack:
                play_rate = self._play_rate(
                    spot.hole_class, spot.position, spot.action_history,
                    spot.stack_bucket, row['n_samples'], 'L2_no_stack')
                # L2 is the noisiest level (pooled across stack depths) —
                # when the gate fails, prefer the static GTO range over a
                # blind fold default; the cohort's "play_rate" estimate may
                # be biased low at coarse keys but the static range still
                # encodes solid baseline strategy.
                min_pr = self._pos_min_play_rate.get(
                    spot.position, self._min_play_rate)
                if play_rate < min_pr:
                    return self._static_or_fold(
                        spot, row['n_samples'], hole_cards, rel_pos, pot_bb,
                        legal_actions, action_size, num_limpers,
                        effective_stack, num_callers,
                        reason=(f"L2 play_rate={play_rate*100:.1f}% "
                                f"< {min_pr*100:.0f}% (pos={spot.position})"))
                # Stack-pooled rows are also dangerous when the decision
                # commits a lot of chips (jam-call defense). Require a
                # higher confidence floor for large calls; otherwise defer
                # to static ranges. Note: vs_3bet/vs_4bet+ never reach this
                # gate (they take the static-action early branch); the gate
                # remains here for very-deep vs_raise spots where the open
                # was unusually large.
                if (call_amount_bb >= self._l2_large_call_bb and
                        play_rate < self._l2_large_call_play_rate):
                    return self._static_or_fold(
                        spot, row['n_samples'], hole_cards, rel_pos, pot_bb,
                        legal_actions, action_size, num_limpers,
                        effective_stack, num_callers,
                        reason=(f"L2 large call ({call_amount_bb:.1f}bb) "
                                f"play_rate={play_rate*100:.1f}% "
                                f"< {self._l2_large_call_play_rate*100:.0f}%"))
                # L2 normally suppresses custom sizing because raise sizes
                # aggregated across stack depths are unreliable (3bet vs
                # short-stacker = shove; vs deep = small raise). EXCEPTION:
                # when cohort consensus is overwhelming (≥80% raise) AND the
                # pooled median sits in a sane preflop range (2.0-15.0 BB),
                # the median is far better than letting the UI compute a
                # half/pot preset bet — short-stacked tables otherwise lose
                # custom sizing entirely (effective stack lands in <40bb
                # bucket where cohort coverage is thin → L0/L1 miss → L2
                # fires → preset size). Bounded range keeps us safe: 2bb
                # floor avoids min-raise corner cases, 15bb ceiling keeps
                # 3bet sizings from leaking through (vs_3bet/vs_4bet take
                # the static-action branch above and never reach here).
                raise_p_l2 = float(row['raise_pct']) / 100.0
                median_bb_l2 = row.get('raise_size_bb_p50')
                trust_l2_size = (
                    raise_p_l2 >= 0.80
                    and median_bb_l2 is not None
                    and 2.0 <= float(median_bb_l2) <= 15.0
                )
                strategy, custom_size = self._row_to_strategy(
                    row, legal_actions, action_size, pot_bb, call_amount_bb,
                    effective_stack, trust_size=trust_l2_size)
                self._log_decision(
                    'L2_no_stack', spot, row['n_samples'],
                    (row['fold_pct'], row['call_pct'], row['raise_pct']),
                    custom_size,
                    extra=(f"size_pooled={median_bb_l2}" if trust_l2_size
                           else "size_suppressed (low consensus or out-of-range)"))
                return strategy, custom_size

            # L3 — no data at any granularity. Prefer the static GTO range
            # over a blind fold default: a hand with no cohort coverage is
            # more often "rare spot the cohort hasn't reached at showdown"
            # than "spot where the cohort always folds." The static range
            # still encodes a coherent baseline.
            return self._static_or_fold(
                spot, 0, hole_cards, rel_pos, pot_bb,
                legal_actions, action_size, num_limpers,
                effective_stack, num_callers,
                reason="L3 no data at any granularity")

        except Exception as e:
            # True error path: lookup machinery failed. Fall back to static.
            self._counters['error_fallback'] += 1
            logger.warning(f"PreflopLookup error path: {e} — "
                           f"falling back to static ranges")
            return self._fallback_strategy(
                hole_cards, rel_pos, pot_bb, legal_actions, action_size,
                num_limpers, effective_stack, num_callers,
            ), None

    # ─── Internals ──────────────────────────────────────────────────────

    def _build_coarser_levels(self, conn):
        """Aggregate `decisions` into two coarser-key dicts at startup.

        L1 key: (hole, pos, hist, stack)        — drop num_to_act
        L2 key: (hole, pos, hist)                — drop stack too

        Single pass over `decisions`, single SQL query.
        """
        from collections import defaultdict

        # raw[key] = {'fold': N, 'call': N, 'raise': N, 'sizes': [...]}
        l1: Dict[Tuple, Dict] = defaultdict(
            lambda: {'fold': 0, 'call': 0, 'raise': 0, 'sizes': []})
        l2: Dict[Tuple, Dict] = defaultdict(
            lambda: {'fold': 0, 'call': 0, 'raise': 0, 'sizes': []})

        for hole, pos, hist, stack, nta, action, size in conn.execute(
            """SELECT hole_class, position, action_history, stack_bucket,
                      num_to_act, action, size_bb FROM decisions"""
        ):
            for key, bucket in (
                ((hole, pos, hist, stack), l1),
                ((hole, pos, hist),         l2),
            ):
                bucket[key][action] += 1
                if action == 'raise' and size is not None:
                    bucket[key]['sizes'].append(float(size))

        return self._finalize_level(l1), self._finalize_level(l2)

    @staticmethod
    def _combos_for_class(hole_class: str) -> int:
        """Combinatorial multiplicity of a 169-class hand:
        pair=6 combos, suited=4, offsuit=12. Used to normalize raw
        decision counts into per-combo deal frequency."""
        if len(hole_class) == 2:
            return 6  # pair (e.g., 'AA')
        if hole_class.endswith('S'):
            return 4
        return 12

    def _build_aa_reference(self, conn) -> Dict[Tuple[str, str, str], float]:
        """For each (position, action_history, stack_bucket), record AA's
        per-combo decision count. AA is universally played → its per-combo
        count ≈ deal frequency at that spot. Other hands' per-combo counts
        divided by AA's give the empirical play rate, recovering
        P(played | dealt) from showdown-only data.

        Builds three references:
          self._aa_per_combo[(pos, hist, stack)]   — for L0/L1 rows
          self._aa_per_combo_no_stack[(pos, hist)] — for L2 rows (sum across stacks)
          self._aa_per_combo_no_hist[(pos,)]       — last-resort fallback
        """
        # Aggregate across num_to_act: AA rows can be split per nta in the
        # lookup table, and we want the spot-total deal frequency. Without
        # GROUP BY the dict was overwritten per nta and silently kept whichever
        # row came last (often the smaller one), which broke the play-rate
        # gate for any (pos, hist, stack) with multiple AA rows.
        out: Dict[Tuple[str, str, str], float] = {}
        for pos, hist, stack, total_n in conn.execute(
            """SELECT position, action_history, stack_bucket, SUM(n_samples)
               FROM lookup WHERE hole_class='AA'
               GROUP BY position, action_history, stack_bucket"""):
            out[(pos, hist, stack)] = total_n / 6.0

        # L2-aligned: sum AA samples across all stack buckets
        ns_out: Dict[Tuple[str, str], float] = {}
        for pos, hist, total_n in conn.execute(
            """SELECT position, action_history, SUM(n_samples)
               FROM lookup WHERE hole_class='AA'
               GROUP BY position, action_history"""):
            ns_out[(pos, hist)] = total_n / 6.0

        self._aa_per_combo_no_stack = ns_out
        return out

    def _play_rate(self, hole_class: str, position: str,
                   action_history: str, stack_bucket: str,
                   n_samples: int, level: str) -> float:
        """Estimate P(played | dealt) for this hand at this spot.

        The level determines which AA reference to use, since aggregation
        across L0/L1/L2 changes both the row's n_samples scope AND the
        baseline:
          L0_exact, L1_no_nta → AA at same (pos, hist, stack)
          L2_no_stack          → AA summed across all stacks (matches row)

        Returns play_rate in [0, ~1.5]. Values >1 are possible when AA
        was played slightly less than the hand was — treat as full play.
        Returns 1.0 (trust the lookup) when no AA reference is available.
        """
        if level == 'L2_no_stack':
            aa_pc = self._aa_per_combo_no_stack.get((position, action_history))
        else:
            aa_pc = self._aa_per_combo.get(
                (position, action_history, stack_bucket))
            if not aa_pc:
                # Stack-specific AA missing — fall through to no-stack baseline
                aa_pc = self._aa_per_combo_no_stack.get(
                    (position, action_history))
        if not aa_pc:
            return 1.0
        combos = self._combos_for_class(hole_class)
        hand_pc = n_samples / combos
        return hand_pc / aa_pc if aa_pc > 0 else 1.0

    @staticmethod
    def _finalize_level(raw: Dict[Tuple, Dict]) -> Dict[Tuple, Dict]:
        """Convert raw counters to the same shape as the lookup table rows."""
        out: Dict[Tuple, Dict] = {}
        for key, b in raw.items():
            n = b['fold'] + b['call'] + b['raise']
            if n == 0:
                continue
            sizes = sorted(b['sizes'])
            mean, p25, p50, p75 = _percentiles(sizes)
            out[key] = {
                'n_samples': n,
                'fold_pct': 100.0 * b['fold'] / n,
                'call_pct': 100.0 * b['call'] / n,
                'raise_pct': 100.0 * b['raise'] / n,
                'raise_size_bb_mean': mean,
                'raise_size_bb_p25': p25,
                'raise_size_bb_p50': p50,
                'raise_size_bb_p75': p75,
            }
        return out

    def _row_to_strategy(self, row: Dict, legal_actions: List[int],
                        action_size: int, pot_bb: float,
                        call_amount_bb: float, effective_stack: float,
                        trust_size: bool = True
                        ) -> Tuple[np.ndarray, Optional[float]]:
        """Map a lookup row's frequencies onto a strategy vector.
        Returns (strategy, custom_size_bb).

        trust_size=False suppresses custom-size emission (raise mass routes
        to nearest preset). Used at L2 where size aggregation across stack
        buckets is unreliable.
        """
        strategy = np.zeros(action_size, dtype=np.float32)
        fold_p = float(row['fold_pct']) / 100.0
        call_p = float(row['call_pct']) / 100.0
        raise_p = float(row['raise_pct']) / 100.0
        target_size_bb = row.get('raise_size_bb_p50') if trust_size else None

        if FOLD in legal_actions:
            strategy[FOLD] = fold_p
        if CALL in legal_actions:
            strategy[CALL] = call_p

        custom_size: Optional[float] = None
        if raise_p > 0:
            # Strategy vector always points at a preset slot (production
            # action_size=6 has no RAISE_CUSTOM index). If we have a target
            # raise size from cohort data (L0/L1 with trust_size=True), ALWAYS
            # emit it via the side-channel — the action layer decides
            # type-vs-preset based on whether bet_input is calibrated.
            #
            # The expanded-action-size case (action_size > 8, used in some
            # tests) writes raise mass directly to the RAISE_CUSTOM index.
            has_direct_slot = (action_size > RAISE_CUSTOM
                               and RAISE_CUSTOM in legal_actions
                               and target_size_bb is not None)
            if has_direct_slot:
                strategy[RAISE_CUSTOM] = raise_p
                custom_size = _round_to_half_bb(float(target_size_bb))
            else:
                preset = self._nearest_preset(
                    target_size_bb, pot_bb, call_amount_bb,
                    effective_stack, legal_actions)
                if preset is not None:
                    strategy[preset] = strategy[preset] + raise_p
                    # Emit size whenever cohort gave us one — action layer
                    # decides how to execute (type or preset size).
                    if target_size_bb is not None:
                        custom_size = _round_to_half_bb(float(target_size_bb))
                else:
                    if CALL in legal_actions:
                        strategy[CALL] += raise_p
                    elif FOLD in legal_actions:
                        strategy[FOLD] += raise_p

        # Renormalize over legal actions (covers cases where some actions
        # weren't legal and their mass was reassigned).
        s = float(strategy.sum())
        if s <= 0.0:
            return self._fold_default([a for a in legal_actions if a < action_size],
                                      action_size), None
        if abs(s - 1.0) > 1e-3:
            strategy = strategy / s

        return strategy, custom_size

    @staticmethod
    def _fold_default(legal_actions: List[int], action_size: int) -> np.ndarray:
        """100% fold strategy (or call if fold isn't legal)."""
        s = np.zeros(action_size, dtype=np.float32)
        if FOLD in legal_actions and FOLD < action_size:
            s[FOLD] = 1.0
        elif CALL in legal_actions and CALL < action_size:
            s[CALL] = 1.0
        elif legal_actions:
            for a in legal_actions:
                if 0 <= a < action_size:
                    s[a] = 1.0 / len(legal_actions)
        return s

    @staticmethod
    def _synth_prior(action_history, num_limpers, pot_bb):
        """Encode an explicit action_history string as the minimal action list
        derive_action_history maps to. Falls back to inferring from
        (num_limpers, pot_bb) if action_history is None."""
        from src.preflop_spot_key import (
            HIST_OPEN, HIST_LIMPED_POT, HIST_VS_RAISE,
            HIST_VS_3BET, HIST_VS_4BET,
        )
        if action_history == HIST_OPEN:
            return []
        if action_history == HIST_LIMPED_POT:
            return [('call', 1.0)]
        if action_history == HIST_VS_RAISE:
            return [('raise', 2.5)]
        if action_history == HIST_VS_3BET:
            return [('raise', 2.5), ('raise', 9.0)]
        if action_history == HIST_VS_4BET:
            return [('raise', 2.5), ('raise', 9.0), ('raise', 22.0)]
        # Inference from pot_bb and limpers
        if pot_bb >= 50:
            return [('raise', 2.5), ('raise', 9.0), ('raise', 22.0)]
        if pot_bb >= 18:
            return [('raise', 2.5), ('raise', 9.0)]
        if pot_bb >= 4:
            return [('raise', 2.5)]
        if num_limpers > 0:
            return [('call', 1.0)] * num_limpers
        return []

    @staticmethod
    def _nearest_preset(target_size_bb: Optional[float], pot_bb: float,
                        call_amount_bb: float, effective_stack: float,
                        legal_actions: List[int]) -> Optional[int]:
        """Pick the legal preset whose resulting BB size is closest to target."""
        if target_size_bb is None:
            for a in (RAISE_HALF, RAISE_POT, RAISE_MIN, ALL_IN):
                if a in legal_actions:
                    return a
            return None
        candidates = {}
        if RAISE_MIN in legal_actions:
            candidates[RAISE_MIN] = max(2.0, 2.0 * call_amount_bb)
        if RAISE_HALF in legal_actions:
            new_pot = pot_bb + call_amount_bb
            candidates[RAISE_HALF] = call_amount_bb + 0.5 * new_pot
        if RAISE_POT in legal_actions:
            new_pot = pot_bb + call_amount_bb
            candidates[RAISE_POT] = call_amount_bb + new_pot
        if ALL_IN in legal_actions:
            candidates[ALL_IN] = effective_stack
        if not candidates:
            return None
        return min(candidates.items(),
                   key=lambda kv: abs(kv[1] - target_size_bb))[0]

    def _cohort_raise_size_bb(self, spot) -> Optional[float]:
        """Walk L0 → L1 → L2 for a raise-size hint (median BB), or None.

        Used by the 3bet+ early branch: action choice comes from the static
        range, but if a raise is chosen we still want a data-driven size
        rather than a fixed preset. Falls back through the same hierarchy
        as the main lookup, but skips the play-rate gate (we already
        decided to raise; the gate is only relevant for action selection).
        """
        # L0 exact
        row = self._exact.get(spot.to_tuple())
        if row and row['n_samples'] >= self._min_samples:
            sz = row.get('raise_size_bb_p50')
            if sz is not None:
                return _round_to_half_bb(float(sz))
        # L1 no_nta (same stack, sizing still trusted)
        row = self._level_no_nta.get(
            (spot.hole_class, spot.position,
             spot.action_history, spot.stack_bucket))
        if row and row['n_samples'] >= self._min_samples_no_nta:
            sz = row.get('raise_size_bb_p50')
            if sz is not None:
                return _round_to_half_bb(float(sz))
        # L2 no_stack — sizing across stack depths is noisy for 3bet+
        # (jam vs small-3bet pooled together). Don't use L2 sizing here.
        return None

    def _static_action_cohort_size(self, spot, hole_cards, rel_pos, pot_bb,
                                   legal_actions, action_size, num_limpers,
                                   effective_stack, num_callers
                                   ) -> Tuple[np.ndarray, Optional[float]]:
        """3bet+ branch: static range drives action, cohort drives size.

        If the static range picks a raise (raise_p > 0), look up the cohort's
        median raise size for this exact spot and emit it via the side-channel.
        If no cohort data exists for sizing, return None — the action layer
        will fall back to the nearest preset.
        """
        # Pass spot.action_history through so the static range routes by the
        # action-sequence (vs_3bet / vs_4bet+) rather than the pot_bb
        # heuristic — at short stacks a 4bet pot can be smaller than the
        # 50bb threshold and silently fall into the vs_3bet bucket.
        strategy = self._fallback_strategy(
            hole_cards, rel_pos, pot_bb, legal_actions, action_size,
            num_limpers, effective_stack, num_callers,
            action_history=spot.action_history,
        )
        # Check if any raise mass exists in the static strategy. If so,
        # consult cohort for size; otherwise the action is fold/call and
        # no size is needed.
        custom_size: Optional[float] = None
        raise_indices = [a for a in (RAISE_HALF, RAISE_POT, ALL_IN, RAISE_MIN)
                         if a < action_size]
        raise_mass = float(sum(strategy[a] for a in raise_indices
                                if a < len(strategy)))
        if raise_mass > 0:
            custom_size = self._cohort_raise_size_bb(spot)
        self._counters['static_action_cohort_size'] += 1
        if logger.isEnabledFor(logging.DEBUG):
            spot_repr = (f"{spot.hole_class}/{spot.position}/"
                         f"{spot.action_history}/{spot.stack_bucket}/"
                         f"nta={spot.num_to_act}")
            size_str = f"{custom_size:.2f}bb" if custom_size else "preset"
            logger.debug(f"PF[static_action_cohort_size] {spot_repr} "
                         f"raise_mass={raise_mass:.2f} size={size_str}")
        return strategy, custom_size

    def _static_or_fold(self, spot, n_samples, hole_cards, rel_pos, pot_bb,
                        legal_actions, action_size, num_limpers,
                        effective_stack, num_callers, reason: str
                        ) -> Tuple[np.ndarray, Optional[float]]:
        """Use the static-range fallback if available; else fold default.

        Called from L2 (when the gate fails or the call commits a lot of
        chips) and L3 (no data anywhere). The static GTO range is a safer
        default than blind fold for spots where the cohort data is too
        sparse or noisy to trust.
        """
        if self._fallback is not None:
            try:
                strategy = self._fallback_strategy(
                    hole_cards, rel_pos, pot_bb, legal_actions, action_size,
                    num_limpers, effective_stack, num_callers,
                    action_history=spot.action_history,
                )
                self._counters['L3_static_fallback' if n_samples == 0
                                else 'L2_static_fallback'] += 1
                if logger.isEnabledFor(logging.DEBUG):
                    spot_repr = (f"{spot.hole_class}/{spot.position}/"
                                 f"{spot.action_history}/{spot.stack_bucket}/"
                                 f"nta={spot.num_to_act}")
                    level = 'L3_static' if n_samples == 0 else 'L2_static'
                    logger.debug(f"PF[{level}] {spot_repr} → static range "
                                 f"({reason})")
                return strategy, None
            except Exception as e:
                logger.warning(f"Static fallback raised: {e} — "
                               f"using fold default")
        # No fallback or fallback raised — preserve old fold-default behavior
        self._log_decision('L3_fold_default', spot, n_samples, None, None,
                           extra=reason)
        return self._fold_default(legal_actions, action_size), None

    def _fallback_strategy(self, hole_cards, rel_pos, pot_bb, legal_actions,
                           action_size, num_limpers, effective_stack,
                           num_callers, action_history: Optional[str] = None):
        """ERROR-RECOVERY ONLY. Called when the lookup path raised."""
        if self._fallback is not None:
            try:
                return self._fallback.get_strategy(
                    hole_cards=hole_cards, rel_pos=rel_pos, pot_bb=pot_bb,
                    legal_actions=legal_actions, action_size=action_size,
                    num_limpers=num_limpers, effective_stack=effective_stack,
                    num_callers=num_callers, action_history=action_history,
                )
            except TypeError:
                # Older PreflopRanges without action_history kwarg
                return self._fallback.get_strategy(
                    hole_cards, rel_pos, pot_bb, legal_actions, action_size,
                    num_limpers, effective_stack, num_callers,
                )
        return self._fold_default(legal_actions, action_size)

    # ─── Diagnostics ────────────────────────────────────────────────────

    @property
    def num_spots(self) -> int:
        return len(self._exact)

    @property
    def coverage_levels(self) -> Dict[str, int]:
        """Spot counts at each granularity (for build-time inspection)."""
        return {
            'exact':       len(self._exact),
            'no_num_to_act': len(self._level_no_nta),
            'no_stack':    len(self._level_no_stack),
        }

    def has_spot(self, hole_cards, rel_pos, num_players, action_history,
                 effective_bb, num_to_act) -> bool:
        spot = build_spot_key(
            hole_cards=hole_cards, rel_pos=rel_pos, num_players=num_players,
            prior_actions=self._synth_prior(action_history, 0, 0.0),
            effective_bb=effective_bb, num_to_act=num_to_act,
        )
        row = self._exact.get(spot.to_tuple())
        return row is not None and row['n_samples'] >= self._min_samples
