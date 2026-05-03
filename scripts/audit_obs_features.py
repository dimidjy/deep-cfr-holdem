"""Audit obs_vector features in logs/action_debug for breakage.

Scans every action_debug JSON, then for each dim of the runtime obs_vector:

  1. Verifies the value at obs_vector[i] matches the named feature in the
     `features` dict at the same logical position (catches index drift).
  2. Flags features that are constant across all spots.
  3. Flags features that are constant within a particular street while
     varying elsewhere (catches the bug we just fixed: hero_has_initiative
     was 0/1 preflop but always 0 on flop/turn/river).
  4. Flags out-of-range values for the bounded features.

Run:
    python3 scripts/audit_obs_features.py
    python3 scripts/audit_obs_features.py --since 20260501
    python3 scripts/audit_obs_features.py --max-files 500

Exit code is 0; this is a diagnostic, not a gate.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agent.obs_builder import feature_names

LOG_DIR = REPO_ROOT / 'logs' / 'action_debug'
STREETS = ['preflop', 'flop', 'turn', 'river']

# Bounds we expect features to live within. None = no check.
# (lo, hi) — values outside trigger a warning.
EXPECTED_BOUNDS = {
    # Position one-hots
    'pos_btn': (0.0, 1.0), 'pos_sb': (0.0, 1.0), 'pos_bb': (0.0, 1.0),
    'pos_utg': (0.0, 1.0), 'pos_mp': (0.0, 1.0), 'pos_co': (0.0, 1.0),
    'is_button': (0.0, 1.0), 'pos_norm': (0.0, 1.0),
    # Booleans / 0-1 normalized
    'is_pair': (0.0, 1.0), 'is_suited': (0.0, 1.0),
    'is_preflop': (0.0, 1.0), 'is_flop': (0.0, 1.0),
    'is_turn': (0.0, 1.0), 'is_river': (0.0, 1.0),
    'facing_bet': (0.0, 1.0), 'can_raise': (0.0, 1.0),
    'top_pair': (0.0, 1.0),
    'pot_was_raised': (0.0, 1.0), 'facing_3bet_plus': (0.0, 1.0),
    'hero_has_initiative': (0.0, 1.0),
    # Normalized [0, 1]
    'pot_normalized': (0.0, 1.0), 'stack_to_pot': (0.0, 1.0),
    'pot_odds': (0.0, 1.0), 'high_card_norm': (0.0, 1.0),
    'low_card_norm': (0.0, 1.0), 'rank_gap_norm': (0.0, 1.0),
    'm_ratio': (0.0, 1.0), 'relative_stack': (0.0, 1.0),
    'hand_strength': (0.0, 1.0), 'made_hand_cat': (0.0, 1.0),
    'hero_flush_participation': (0.0, 1.0),
    'hero_straight_participation': (0.0, 1.0),
    'board_pairing_level': (0.0, 1.0),
    'board_flush_texture': (0.0, 1.0),
    'board_straight_texture': (0.0, 1.0),
    'hand_rank_within_cat': (0.0, 1.0),
    'players_alive_norm': (0.0, 1.0),
    'call_amount_norm': (0.0, 1.0),
    'aggression_level': (0.0, 1.0),
    'bet_to_pot_norm': (0.0, 1.0),
    'my_pot_share': (0.0, 1.0),
    'not_raise_fraction': (0.0, 1.0),
    'min_opp_stack_norm': (0.0, 1.0),
    'opp_aggression_ratio': (0.0, 1.0),
    'opp_fold_ratio': (0.0, 1.0),
    'avg_bet_sizing': (0.0, 1.0),
    'max_bet_sizing': (0.0, 1.0),
    'num_raises_norm': (0.0, 1.0),
    'street_aggression': (0.0, 1.0),
    'opp_check_ratio': (0.0, 1.0),
    'pot_escalation': (0.0, 1.0),
    'pot_at_street_start_norm': (0.0, 1.0),
    'agg_count_this_street_norm': (0.0, 1.0),
    # Signed
    'equity_vs_odds': (-1.0, 1.0),
}


def detect_obs_layout(any_obs_len: int) -> Tuple[bool, bool]:
    """Detect (behavioral_stats, equity_vs_odds) flags from obs vector length.

    Layout sizes (with pad_to_6p=True):
      base = 8+3+5+2+9+4+6+5 = 42
      +behavioral_stats (8) + betting_line (3) = +11 → 53
      +equity_vs_odds = +1 → 54  (or 50/51 in the no-bh era)
    """
    if any_obs_len == 50:
        return False, True   # legacy 42 + 8 (no bh) + ... actually rare
    if any_obs_len == 53:
        return True, False   # base 42 + bh 8 + betting_line 3
    if any_obs_len == 54:
        return True, True
    if any_obs_len == 42:
        return False, False
    # Unknown — best guess
    return True, any_obs_len > 53


def load_logs(max_files: Optional[int], since: Optional[str]) -> List[dict]:
    paths = sorted(LOG_DIR.glob('*.json'))
    if since:
        paths = [p for p in paths if p.name >= since]
    if max_files is not None:
        paths = paths[:max_files]
    out = []
    for p in paths:
        try:
            with open(p) as fh:
                d = json.load(fh)
        except Exception:
            continue
        # Skip non-action logs (parser dumps from screen_*)
        if 'obs_vector' not in d or 'features' not in d:
            continue
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-files', type=int, default=None,
                    help='Limit how many logs to scan')
    ap.add_argument('--since', type=str, default=None,
                    help='Only scan logs whose filename is >= this prefix '
                         '(e.g. 20260501)')
    ap.add_argument('--top', type=int, default=10,
                    help='Examples to show per anomaly (default: 10)')
    args = ap.parse_args()

    print(f"Scanning {LOG_DIR}...")
    logs = load_logs(args.max_files, args.since)
    if not logs:
        print("No action_debug logs found.")
        return
    print(f"Loaded {len(logs)} logs.\n")

    obs_len = len(logs[0]['obs_vector'])
    use_bh, use_evo = detect_obs_layout(obs_len)
    expected_names = feature_names(
        num_players=6, pad_to_6p=True,
        behavioral_stats=use_bh, equity_vs_odds=use_evo)
    print(f"Detected layout: obs_len={obs_len}, behavioral_stats={use_bh}, "
          f"equity_vs_odds={use_evo}")
    print(f"Expected feature count: {len(expected_names)}")
    if len(expected_names) != obs_len:
        print(f"  ⚠️  MISMATCH: obs_vector has {obs_len} dims but "
              f"feature_names() produced {len(expected_names)} names — "
              f"index alignment audits below may be off.\n")
    else:
        print(f"  ✓ feature_names() length matches obs_vector\n")

    # ── Per-dim accumulators ───────────────────────────────────────────────
    distinct_global: List[set] = [set() for _ in range(obs_len)]
    distinct_per_street: List[Dict[str, set]] = [
        defaultdict(set) for _ in range(obs_len)]
    obs_vs_features_mismatches: List[List[Tuple[float, float]]] = [
        [] for _ in range(obs_len)]
    out_of_range: List[List[float]] = [[] for _ in range(obs_len)]

    for d in logs:
        obs = d['obs_vector']
        feats = d['features']
        street = d.get('game_state', {}).get('street', 'unknown')
        for i, v in enumerate(obs):
            r = round(v, 4)
            distinct_global[i].add(r)
            distinct_per_street[i][street].add(r)
            # Index alignment check
            if i < len(expected_names):
                name = expected_names[i]
                if name in feats:
                    fv = feats[name]
                    if abs(float(fv) - float(v)) > 1e-4:
                        obs_vs_features_mismatches[i].append((v, fv))
                # Range check
                if name in EXPECTED_BOUNDS:
                    lo, hi = EXPECTED_BOUNDS[name]
                    if not (lo - 1e-4 <= v <= hi + 1e-4):
                        out_of_range[i].append(v)

    # ── Reports ────────────────────────────────────────────────────────────
    issues = []   # list of (severity, message)

    # Global constants
    print("=" * 78)
    print("CONSTANT FEATURES (only one distinct value across all spots)")
    print("=" * 78)
    found_const = False
    for i, vals in enumerate(distinct_global):
        if len(vals) == 1:
            name = expected_names[i] if i < len(expected_names) else f'?'
            v = next(iter(vals))
            # 0/1 booleans being constant on a small sample is suspicious
            # but constants like is_pair=0 across the run can be normal —
            # only flag those that should clearly vary.
            print(f"  obs[{i:2d}] {name:<32} = {v}")
            found_const = True
            issues.append(('LOW', f'obs[{i}] {name} is constant ({v})'))
    if not found_const:
        print("  (none)")
    print()

    # Pre-compute "explainable" positions: position one-hots whose
    # constant-postflop pattern is fully explained by the lobby-size mix.
    n_active_per_street = defaultdict(Counter)
    for d in logs:
        s = d.get('game_state', {}).get('street', '?')
        n = d.get('game_state', {}).get('num_active_players', 0)
        n_active_per_street[s][n] += 1
    pos_to_min_n = {
        'pos_btn': 2, 'pos_sb': 2, 'pos_bb': 2,
        'pos_co': 4, 'pos_mp': 5, 'pos_utg': 6,
    }

    def position_explained(name: str, const_streets: List[str]) -> bool:
        if name not in pos_to_min_n:
            return False
        min_n = pos_to_min_n[name]
        # If for every constant street the agent had < 30 eligible decisions,
        # call it explained. Otherwise it's suspicious.
        for st in const_streets:
            n_elig = sum(c for n_, c in n_active_per_street[st].items()
                         if n_ >= min_n)
            if n_elig >= 30:
                return False
        return True

    # Features the obs builder DELIBERATELY zeros out on preflop. The 9
    # equity-block features depend on community cards (returns zeros via
    # `if not community_cards: return np.zeros(9)`). hero_has_initiative
    # is structurally 0 preflop because hero hasn't acted on first
    # decision and `call_amount > 0` resets it on subsequent decisions
    # (re-raise spots). All by design — being constant 0 on preflop is
    # NOT a bug for these.
    EXPECTED_ZERO_PREFLOP = {
        'hand_strength', 'made_hand_cat',
        'hero_flush_participation', 'hero_straight_participation',
        'top_pair', 'board_pairing_level',
        'board_flush_texture', 'board_straight_texture',
        'hand_rank_within_cat',
        'hero_has_initiative',
    }

    def is_expected_zero_preflop(name: str, const_vals: dict) -> bool:
        """True if the only constant-street is preflop with value 0 and
        this feature is in the known by-design-zero list."""
        if name not in EXPECTED_ZERO_PREFLOP:
            return False
        return (list(const_vals.keys()) == ['preflop']
                and const_vals.get('preflop') == 0.0)

    # Street-degenerate features
    print("=" * 78)
    print("STREET-DEGENERATE FEATURES (constant on some streets, varies on others)")
    print("=" * 78)
    found_st = False
    for i, by_street in enumerate(distinct_per_street):
        if len(distinct_global[i]) <= 1:
            continue   # already reported as global const
        const_streets = []
        var_streets = []
        for s in STREETS:
            n = len(by_street.get(s, set()))
            if n == 0:
                continue
            if n == 1:
                const_streets.append(s)
            else:
                var_streets.append(s)
        if const_streets and var_streets:
            name = expected_names[i] if i < len(expected_names) else f'?'
            const_vals = {
                s: next(iter(by_street[s])) for s in const_streets}
            var_counts = {
                s: len(by_street[s]) for s in var_streets}
            # Suppress two known-by-design patterns:
            #  - position one-hots constant postflop because lobby mix
            #    rarely played 6-handed (UTG/MP/CO never reached flop)
            #  - equity-block + initiative features that the obs builder
            #    intentionally zeros on preflop
            explained = (position_explained(name, const_streets)
                         or is_expected_zero_preflop(name, const_vals))
            if position_explained(name, const_streets):
                tag = '   (explained by lobby mix)'
            elif is_expected_zero_preflop(name, const_vals):
                tag = '   (zero preflop by design)'
            else:
                tag = ''
            print(f"  obs[{i:2d}] {name:<32}{tag}")
            print(f"           constant on: {const_vals}")
            print(f"           varies on:   {var_counts}")
            found_st = True
            if not explained:
                sev = ('HIGH' if any(v == 0 for v in const_vals.values())
                       else 'MED')
                issues.append((sev,
                    f'obs[{i}] {name} constant {const_vals} on '
                    f'{",".join(const_streets)}; varies on '
                    f'{",".join(var_streets)}'))
    if not found_st:
        print("  (none)")
    print()

    # Index alignment with named features dict
    print("=" * 78)
    print("OBS_VECTOR vs FEATURES DICT MISMATCHES (suggests wrong index → name)")
    print("=" * 78)
    found_mm = False
    for i, mismatches in enumerate(obs_vs_features_mismatches):
        if not mismatches:
            continue
        name = expected_names[i] if i < len(expected_names) else f'?'
        n = len(mismatches)
        rate = 100.0 * n / len(logs)
        # A few examples
        sample = mismatches[:3]
        sample_str = ', '.join(
            f'obs={a:.4f} feat={b:.4f}' for a, b in sample)
        print(f"  obs[{i:2d}] {name:<32} {n}/{len(logs)} ({rate:.1f}%) "
              f"mismatch — e.g. {sample_str}")
        found_mm = True
        if rate > 1.0:
            issues.append(('HIGH',
                f'obs[{i}] {name} disagrees with features[\"{name}\"] '
                f'in {rate:.1f}% of spots'))
    if not found_mm:
        print("  (none — obs_vector and features dict align on every dim)")
    print()

    # Out-of-range values
    print("=" * 78)
    print("OUT-OF-RANGE VALUES (feature outside its declared bounds)")
    print("=" * 78)
    found_oor = False
    for i, vals in enumerate(out_of_range):
        if not vals:
            continue
        name = expected_names[i] if i < len(expected_names) else f'?'
        bounds = EXPECTED_BOUNDS.get(name, ('?', '?'))
        lo_v = min(vals)
        hi_v = max(vals)
        print(f"  obs[{i:2d}] {name:<32} bounds={bounds}  observed: "
              f"[{lo_v:.3f}, {hi_v:.3f}] (n={len(vals)})")
        found_oor = True
        issues.append(('MED',
            f'obs[{i}] {name} outside bounds {bounds}: '
            f'observed [{lo_v:.3f}, {hi_v:.3f}]'))
    if not found_oor:
        print("  (none)")
    print()

    # Cross-checks — specific known-shape expectations
    print("=" * 78)
    print("DOMAIN CROSS-CHECKS")
    print("=" * 78)
    cc_failed = []

    # Off-grid values in features that should be discrete
    discrete_features = {
        'top_pair': {0.0, 1.0},
        'is_pair': {0.0, 1.0},
        'is_suited': {0.0, 1.0},
        'is_button': {0.0, 1.0},
        'facing_bet': {0.0, 1.0},
        'can_raise': {0.0, 1.0},
        'pot_was_raised': {0.0, 1.0},
        'facing_3bet_plus': {0.0, 1.0},
        'is_preflop': {0.0, 1.0}, 'is_flop': {0.0, 1.0},
        'is_turn': {0.0, 1.0}, 'is_river': {0.0, 1.0},
        'pos_btn': {0.0, 1.0}, 'pos_sb': {0.0, 1.0}, 'pos_bb': {0.0, 1.0},
        'pos_utg': {0.0, 1.0}, 'pos_mp': {0.0, 1.0}, 'pos_co': {0.0, 1.0},
        'hero_has_initiative': {0.0, 1.0},
        'hero_flush_participation': {0.0, 0.5, 1.0},
        'hero_straight_participation': {0.0, 0.5, 1.0},
        'board_pairing_level': {0.0, 0.33, 0.34, 0.67, 1.0},
    }
    for fname, allowed in discrete_features.items():
        if fname not in expected_names:
            continue
        idx = expected_names.index(fname)
        offgrid = []
        for d in logs:
            v = float(d['obs_vector'][idx])
            if not any(abs(v - a) < 0.02 for a in allowed):
                offgrid.append(v)
        if offgrid:
            rate = 100.0 * len(offgrid) / len(logs)
            if rate > 1.0:
                cc_failed.append(
                    f'{fname} off-grid in {rate:.1f}% of spots '
                    f'(should be {sorted(allowed)}; e.g. {offgrid[0]:.4f})')

    # 1. Street one-hot exclusivity: exactly one of is_preflop/flop/turn/river=1
    street_idx = {n: expected_names.index(n) for n in
                  ('is_preflop', 'is_flop', 'is_turn', 'is_river')
                  if n in expected_names}
    if len(street_idx) == 4:
        bad = 0
        for d in logs:
            obs = d['obs_vector']
            s = sum(round(obs[idx]) for idx in street_idx.values())
            if s != 1:
                bad += 1
        if bad:
            cc_failed.append(
                f'street one-hot is not exclusive in {bad}/{len(logs)} '
                f'spots (expected exactly one of is_preflop/flop/turn/river=1)')

    # 2. is_preflop=1 ↔ community_cards == 0
    if 'is_preflop' in expected_names:
        idx = expected_names.index('is_preflop')
        bad = 0
        for d in logs:
            is_pf = round(d['obs_vector'][idx])
            n_cc = len(d.get('game_state', {}).get('community_cards', []))
            if (is_pf == 1) != (n_cc == 0):
                bad += 1
        if bad:
            cc_failed.append(
                f'is_preflop disagrees with len(community_cards) in '
                f'{bad}/{len(logs)} spots')

    # 3. facing_bet=1 ↔ call_amount > 0
    if 'facing_bet' in expected_names:
        idx = expected_names.index('facing_bet')
        bad = 0
        for d in logs:
            fb = round(d['obs_vector'][idx])
            ca = d.get('game_state', {}).get('call_amount', 0)
            if (fb == 1) != (ca > 0):
                bad += 1
        if bad:
            cc_failed.append(
                f'facing_bet disagrees with call_amount sign in '
                f'{bad}/{len(logs)} spots')

    # 4. Equity in [0, 1] (could use -1..1 in eval7 misuse)
    if 'hand_strength' in expected_names:
        idx = expected_names.index('hand_strength')
        n_zero = sum(1 for d in logs if d['obs_vector'][idx] == 0)
        n_postflop_zero = sum(
            1 for d in logs
            if d['obs_vector'][idx] == 0
            and d.get('game_state', {}).get('street') != 'preflop')
        if n_postflop_zero > len(logs) * 0.05:
            cc_failed.append(
                f'hand_strength == 0 in {n_postflop_zero} postflop spots '
                f'({100.0*n_postflop_zero/len(logs):.1f}% of all decisions) — '
                f'check equity model is loaded')

    if cc_failed:
        for msg in cc_failed:
            print(f"  ⚠️  {msg}")
            issues.append(('MED', msg))
    else:
        print("  ✓ all cross-checks passed")
    print()

    # ── Summary ────────────────────────────────────────────────────────────
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if not issues:
        print("  ✓ No anomalies detected.")
        return
    by_sev = defaultdict(list)
    for sev, msg in issues:
        by_sev[sev].append(msg)
    for sev in ('HIGH', 'MED', 'LOW'):
        if by_sev[sev]:
            print(f"\n  [{sev}] {len(by_sev[sev])} issue(s):")
            for m in by_sev[sev]:
                print(f"    - {m}")


if __name__ == '__main__':
    main()
