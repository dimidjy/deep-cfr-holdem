"""Audit the postflop sizing DB for ALL-IN promotion patterns.

Read-only investigation of ``data/postflop_sizing.sqlite``. Produces five
sections (printed and dumped to JSON) for understanding which spots trigger
the runtime ALL-IN promotion in ``src/postflop_sizing.py`` and which spots
are *near-miss* candidates that a relaxed threshold would unlock.

Usage:
    python3 scripts/analysis/postflop_allin_audit.py
    python3 scripts/analysis/postflop_allin_audit.py --min-samples 20 --threshold 0.30
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.postflop_db import connect, load_lookup_into_memory  # noqa: E402

DEFAULT_DB = str(ROOT / 'data' / 'postflop_sizing.sqlite')
RESULTS_DIR = ROOT / 'scripts' / 'analysis' / 'results'

SAMPLE_THRESHOLDS = [5, 10, 20, 50]
PROMO_THRESHOLDS = [0.25, 0.30, 0.35, 0.45, 0.55, 0.65, 0.75]


def coverage_matrix(spots):
    """Section 1 — for each (min_samples, threshold) pair, count qualifying spots."""
    out = []
    for min_n in SAMPLE_THRESHOLDS:
        for thr in PROMO_THRESHOLDS:
            qualifying = [
                r for r in spots
                if r['n_samples'] >= min_n
                and r['all_in_pct'] >= thr * 100
            ]
            out.append({
                'min_samples': min_n,
                'threshold_pct': thr * 100,
                'spots': len(qualifying),
                'samples_covered': sum(r['n_samples'] for r in qualifying),
                'all_in_samples_covered': sum(r['n_all_in'] for r in qualifying),
            })
    return out


def per_dimension_breakdown(spots, min_n=20):
    """Section 2 — for each dimension, group counts and avg all_in_pct."""
    qualifying = [r for r in spots if r['n_samples'] >= min_n]
    dimensions = ['street', 'spr_bucket', 'position', 'action_context',
                  'hand_bucket', 'board_texture', 'draw_type']
    out = {}
    for dim in dimensions:
        groups = defaultdict(lambda: {'spots': 0, 'samples': 0, 'sum_pct': 0.0,
                                       'promo_count': 0})
        for r in qualifying:
            g = groups[r[dim]]
            g['spots'] += 1
            g['samples'] += r['n_samples']
            g['sum_pct'] += r['all_in_pct']
            if r['all_in_pct'] >= 35.0:
                g['promo_count'] += 1
        rows = []
        for k, g in groups.items():
            avg = g['sum_pct'] / g['spots'] if g['spots'] else 0.0
            rows.append({
                'value': k, 'spots': g['spots'], 'samples': g['samples'],
                'avg_all_in_pct': round(avg, 1), 'promo_count': g['promo_count'],
            })
        rows.sort(key=lambda x: x['promo_count'], reverse=True)
        out[dim] = rows
    return out


def top_promoting_spots(spots, min_n=20, threshold=0.35, limit=25):
    """Section 3 — top N currently-promoting spots by sample count."""
    qualifying = [
        r for r in spots
        if r['n_samples'] >= min_n and r['all_in_pct'] >= threshold * 100
    ]
    qualifying.sort(key=lambda r: r['n_samples'], reverse=True)
    return qualifying[:limit]


def near_miss_spots(spots, min_n=50, lo=0.20, hi=0.35, limit=25):
    """Section 4 — high-sample spots in the [lo, hi) all_in_pct band."""
    qualifying = [
        r for r in spots
        if r['n_samples'] >= min_n
        and lo * 100 <= r['all_in_pct'] < hi * 100
    ]
    qualifying.sort(key=lambda r: r['n_samples'], reverse=True)
    return qualifying[:limit]


def noise_risk_audit(spots, threshold=0.35):
    """Section 5 — currently-promoting spots at risk of being noise."""
    promo = [r for r in spots if r['all_in_pct'] >= threshold * 100]
    buckets = {
        'n_in_5_to_10': [r for r in promo if 5 <= r['n_samples'] < 10],
        'n_in_10_to_20': [r for r in promo if 10 <= r['n_samples'] < 20],
        'n_lt_20_and_pct_100': [r for r in promo
                                 if r['n_samples'] < 20 and r['all_in_pct'] >= 99.999],
    }
    out = {}
    for name, rows in buckets.items():
        rows.sort(key=lambda r: (-r['all_in_pct'], r['n_samples']))
        out[name] = {'count': len(rows), 'examples': rows[:5]}
    return out


def fmt_spot(r):
    """Compact one-line spot description."""
    return (f"{r['street']:5s} {r['hand_bucket']:14s} {r['draw_type']:11s} "
            f"{r['board_texture']:11s} {r['position']:3s} SPR{r['spr_bucket']:5s} "
            f"{r['action_context']:16s}")


def print_coverage(rows):
    print("\n=== SECTION 1: COVERAGE MATRIX (min_samples × threshold) ===")
    print(f"{'min_n':>6} {'thr%':>6} {'spots':>7} {'samples':>9} {'all_in_samples':>15}")
    for r in rows:
        print(f"{r['min_samples']:6d} {r['threshold_pct']:6.1f} "
              f"{r['spots']:7d} {r['samples_covered']:9d} "
              f"{r['all_in_samples_covered']:15d}")


def print_breakdown(by_dim):
    print("\n=== SECTION 2: PER-DIMENSION BREAKDOWN (n>=20) ===")
    for dim, rows in by_dim.items():
        print(f"\n  -- {dim} --")
        print(f"  {'value':<18} {'spots':>6} {'samples':>9} "
              f"{'avg_pct':>8} {'promo_n':>8}")
        for r in rows:
            print(f"  {str(r['value']):<18} {r['spots']:6d} {r['samples']:9d} "
                  f"{r['avg_all_in_pct']:8.1f} {r['promo_count']:8d}")


def print_top_spots(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'spot':<75} {'n':>5} {'n_ai':>5} {'%':>5} "
          f"{'p25%':>5} {'p50%':>5} {'p75%':>5}")
    for r in rows:
        p25 = r.get('bet_size_pct_p25') or 0
        p50 = r.get('bet_size_pct_p50') or 0
        p75 = r.get('bet_size_pct_p75') or 0
        print(f"{fmt_spot(r):<75} {r['n_samples']:5d} {r['n_all_in']:5d} "
              f"{r['all_in_pct']:5.1f} {p25*100:5.0f} {p50*100:5.0f} "
              f"{p75*100:5.0f}")


def print_noise(audit):
    print("\n=== SECTION 5: NOISE-RISK AUDIT (currently-promoting spots) ===")
    for name, info in audit.items():
        print(f"\n  -- {name}: {info['count']} spots --")
        for r in info['examples']:
            print(f"    {fmt_spot(r)} n={r['n_samples']:>3d} "
                  f"n_ai={r['n_all_in']:>3d} pct={r['all_in_pct']:5.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--min-samples', type=int, default=20,
                    help='Sample floor for sections 2-3 (default: 20)')
    ap.add_argument('--threshold', type=float, default=0.35,
                    help='All-in promotion threshold for sections 3, 5 (default: 0.35)')
    ap.add_argument('--out-dir', default=str(RESULTS_DIR))
    args = ap.parse_args()

    if not Path(args.db).exists():
        sys.exit(f"DB not found: {args.db}")

    conn = connect(args.db)
    try:
        cache = load_lookup_into_memory(conn)
    finally:
        conn.close()

    spots = []
    for key, row in cache.items():
        spots.append({
            'street': key[0], 'hand_bucket': key[1], 'draw_type': key[2],
            'board_texture': key[3], 'position': key[4],
            'spr_bucket': key[5], 'action_context': key[6],
            **row,
        })

    print(f"Loaded {len(spots):,} spots from {args.db}")
    print(f"Total raw samples: {sum(s['n_samples'] for s in spots):,}")
    print(f"Total all-in samples: {sum(s['n_all_in'] for s in spots):,}")

    coverage = coverage_matrix(spots)
    by_dim = per_dimension_breakdown(spots, min_n=args.min_samples)
    top_promo = top_promoting_spots(spots, min_n=args.min_samples,
                                     threshold=args.threshold, limit=25)
    near_miss = near_miss_spots(spots, min_n=50, lo=0.20, hi=0.35, limit=25)
    noise = noise_risk_audit(spots, threshold=args.threshold)

    print_coverage(coverage)
    print_breakdown(by_dim)
    print_top_spots(top_promo,
                    f"SECTION 3: TOP-25 PROMOTING SPOTS (n>={args.min_samples}, "
                    f"all_in_pct>={args.threshold*100:.0f}%)")
    print_top_spots(near_miss,
                    "SECTION 4: NEAR-MISS SPOTS (n>=50, all_in_pct in [20%, 35%))")
    print_noise(noise)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'postflop_allin_audit_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'generated_at': ts,
            'db': args.db,
            'min_samples': args.min_samples,
            'threshold': args.threshold,
            'totals': {
                'spots': len(spots),
                'samples': sum(s['n_samples'] for s in spots),
                'all_in_samples': sum(s['n_all_in'] for s in spots),
            },
            'section1_coverage': coverage,
            'section2_per_dimension': by_dim,
            'section3_top_promoting': top_promo,
            'section4_near_miss': near_miss,
            'section5_noise_risk': noise,
        }, f, indent=2, default=str)
    print(f"\nJSON dump → {out_path}")


if __name__ == '__main__':
    main()
