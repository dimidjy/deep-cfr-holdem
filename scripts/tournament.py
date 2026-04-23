"""Full tournament across all checkpoints in a model directory.

For each model in the iter range, runs:
  1. Spot-score validation (fast)
  2. HH validation (default 40k hands)
  3. H2H self-play vs every OTHER model (default 30k hands per match, both directions)

Saves per-test JSON files so runs are RESUMABLE — skip any test whose output
file already exists. Aggregates everything into a final comparison table at the end.

Usage:
    # Run the full tournament (slow — hours)
    python3 scripts/tournament.py --models-dir models/v12 --min-iter 2600 --max-iter 5000

    # Just aggregate existing results (fast)
    python3 scripts/tournament.py --aggregate-only

    # Customize hand counts
    python3 scripts/tournament.py --h2h-hands 30000 --hh-hands 40000

    # Only H2H phase (skip spots + HH if already done)
    python3 scripts/tournament.py --skip-spots --skip-hh

    # Dry run — print commands without running
    python3 scripts/tournament.py --dry-run

Output layout:
    tournament_results/
        spots/<model>_spots.json
        hh/<model>_hh.json
        h2h/<a>_vs_<b>.json              # a in seats 0,2,4; b in 1,3,5
        summary.json                     # final aggregated table
        summary.csv                      # human-readable
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / 'tournament_results'  # overridden per-run via --output-dir

# Regex patterns to extract key metrics from stdout
_RE_SPOT_SCORE = re.compile(r'\[spots (\d+)\] score=(\S+)/\S+ \((\d+\.\d+)%\)')
_RE_SPOT_EXACT = re.compile(r'exact=(\d+)\s+partial=(\d+)')
_RE_HH_EDGE    = re.compile(r'Model edge:\s+([+-]?\d+\.\d+)\s+BB/hand')
_RE_HH_AGREE   = re.compile(r'agreement=(\d+\.\d+)%')
_RE_H2H_BB100  = re.compile(r'BB/100\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)')
_RE_H2H_AF     = re.compile(r'AF\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)')
_RE_H2H_CBET   = re.compile(r'CBet\s+([+-]?\d+\.\d+)%\s+([+-]?\d+\.\d+)%')
_RE_H2H_WTSD   = re.compile(r'WTSD\s+([+-]?\d+\.\d+)%\s+([+-]?\d+\.\d+)%')
_RE_H2H_VPIP   = re.compile(r'VPIP\s+([+-]?\d+\.\d+)%\s+([+-]?\d+\.\d+)%')
_RE_H2H_PFR    = re.compile(r'PFR\s+([+-]?\d+\.\d+)%\s+([+-]?\d+\.\d+)%')


def discover_models(models_dir: Path, min_iter: int, max_iter: int) -> List[Tuple[str, Path]]:
    """Return sorted list of (label, path) for checkpoints in iter range.

    Matches BOTH plain and suffixed filenames:
      checkpoint_iter_3600.pt       -> label 'iter_3600'
      checkpoint_iter_3600_best.pt  -> label 'iter_3600_best'
      checkpoint_iter_3600_v2.pt    -> label 'iter_3600_v2'

    When a directory contains both plain and _best for the same iter (e.g. the
    user copied 'best' variants into a v-dir that already has training output),
    BOTH are included as distinct entries — they're treated as different models
    since they have different weights.
    """
    pat = re.compile(r'checkpoint_iter_(\d+)(_[A-Za-z0-9]+)?\.pt$')
    seen_labels = {}  # label -> path, to detect collisions
    out = []
    for p in models_dir.glob('checkpoint_iter_*.pt'):
        m = pat.search(p.name)
        if not m:
            continue
        n = int(m.group(1))
        suffix = m.group(2) or ''
        label = f'iter_{n}{suffix}'
        if min_iter <= n <= max_iter:
            if label in seen_labels:
                continue  # duplicate (same filename twice, shouldn't happen)
            seen_labels[label] = p
            out.append((label, p))
    # Sort by (iter number, suffix) so plain comes before suffixed at same iter
    out.sort(key=lambda x: (
        int(re.match(r'iter_(\d+)', x[0]).group(1)),
        x[0],  # secondary sort: lexicographic (empty suffix sorts first)
    ))
    return out


def _run(cmd: List[str], dry_run: bool = False) -> str:
    """Run a subprocess, return captured stdout. Returns '' on dry-run."""
    if dry_run:
        print(f'  [dry-run] {" ".join(cmd)}')
        return ''
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        print(f'  FAILED: {proc.stderr[-500:]}', file=sys.stderr)
        return ''
    return proc.stdout


def run_spot_score(label: str, model_path: Path, output: Path,
                    dry_run: bool = False) -> Optional[Dict]:
    if output.exists():
        return json.loads(output.read_text())
    cmd = ['python3', 'scripts/validate_spot_logs.py', '--model', str(model_path)]
    out = _run(cmd, dry_run=dry_run)
    if dry_run or not out:
        return None
    result: Dict = {'model': label, 'path': str(model_path)}
    m = _RE_SPOT_SCORE.search(out)
    if m:
        result['n_spots'] = int(m.group(1))
        result['score'] = float(m.group(2))
        result['score_pct'] = float(m.group(3))
    m = _RE_SPOT_EXACT.search(out)
    if m:
        result['exact'] = int(m.group(1))
        result['partial'] = int(m.group(2))
    # Per-action breakdown
    per_action = {}
    for line in out.splitlines():
        m = re.match(r'\s+(Fold|Check/Call|Raise.|RaisePot|AllIn)\s+n=(\d+)\s+score=(\S+)/\d+\s+\((\d+)%\)', line)
        if m:
            per_action[m.group(1)] = {
                'n': int(m.group(2)), 'score': float(m.group(3)), 'pct': int(m.group(4))
            }
    result['per_action'] = per_action
    output.write_text(json.dumps(result, indent=2))
    return result


def run_hh(label: str, model_path: Path, hands: int, output: Path,
            dry_run: bool = False) -> Optional[Dict]:
    if output.exists():
        return json.loads(output.read_text())
    cmd = [
        'python3', 'scripts/validate_vs_hh.py',
        '--model', str(model_path),
        '--max-hands', str(hands),
        '--behavioral-stats',
    ]
    out = _run(cmd, dry_run=dry_run)
    if dry_run or not out:
        return None
    result: Dict = {'model': label, 'hands': hands}
    m = _RE_HH_EDGE.search(out)
    if m:
        result['bb_per_hand'] = float(m.group(1))
        result['bb_per_100'] = float(m.group(1)) * 100
    m = _RE_HH_AGREE.search(out)
    if m:
        result['agreement_pct'] = float(m.group(1))
    output.write_text(json.dumps(result, indent=2))
    return result


def run_h2h(label_a: str, path_a: Path, label_b: str, path_b: Path,
             hands: int, output: Path, dry_run: bool = False) -> Optional[Dict]:
    """Run A vs B self-play — A in seats 0,2,4; B in 1,3,5. Extract A's edge."""
    if output.exists():
        return json.loads(output.read_text())
    cmd = [
        'python3', 'scripts/analysis/self_play.py',
        '--model', str(path_a),
        '--vs-facing-bet', str(path_b),
        '--vs-checked-to', str(path_b),
        '--hands', str(hands),
    ]
    out = _run(cmd, dry_run=dry_run)
    if dry_run or not out:
        return None

    # Parse A vs B report
    result: Dict = {
        'model_a': label_a, 'model_b': label_b, 'hands': hands,
        'stats_a': {}, 'stats_b': {},
    }
    m = _RE_H2H_BB100.search(out)
    if m:
        result['stats_a']['bb_per_100'] = float(m.group(1))
        result['stats_b']['bb_per_100'] = float(m.group(2))
        result['edge_a'] = float(m.group(3))  # A - B
    for pat, key in [(_RE_H2H_VPIP, 'vpip'), (_RE_H2H_PFR, 'pfr'),
                     (_RE_H2H_AF, 'af'), (_RE_H2H_CBET, 'cbet'),
                     (_RE_H2H_WTSD, 'wtsd')]:
        m = pat.search(out)
        if m:
            result['stats_a'][key] = float(m.group(1))
            result['stats_b'][key] = float(m.group(2))
    output.write_text(json.dumps(result, indent=2))
    return result


def aggregate(models: List[Tuple[str, Path]]) -> Dict:
    """Build final summary from all saved JSONs. Skips models with missing data."""
    spots_dir = RESULTS_DIR / 'spots'
    hh_dir = RESULTS_DIR / 'hh'
    h2h_dir = RESULTS_DIR / 'h2h'
    summary: Dict[str, Dict] = {}

    for label, _ in models:
        row: Dict = {
            'model': label,
            'spot_pct': None,
            'hh_bb_per_hand': None,
            'hh_agreement_pct': None,
            'avg_h2h_edge': None,
            'avg_vpip': None, 'avg_pfr': None, 'avg_cbet': None,
            'avg_af': None, 'avg_wtsd': None,
            'h2h_wins': 0, 'h2h_losses': 0, 'h2h_ties': 0,
        }
        # Spot score
        p = spots_dir / f'{label}_spots.json'
        if p.exists():
            d = json.loads(p.read_text())
            row['spot_pct'] = d.get('score_pct')
        # HH
        p = hh_dir / f'{label}_hh.json'
        if p.exists():
            d = json.loads(p.read_text())
            row['hh_bb_per_hand'] = d.get('bb_per_hand')
            row['hh_agreement_pct'] = d.get('agreement_pct')
        # H2H: average across matches where this model is EITHER A or B
        edges = []
        vpips, pfrs, cbets, afs, wtsds = [], [], [], [], []
        for match_path in h2h_dir.glob(f'{label}_vs_*.json'):
            d = json.loads(match_path.read_text())
            edges.append(d.get('edge_a', 0.0))  # this model's edge (it's A)
            s = d.get('stats_a', {})
            if 'vpip' in s: vpips.append(s['vpip'])
            if 'pfr' in s: pfrs.append(s['pfr'])
            if 'cbet' in s: cbets.append(s['cbet'])
            if 'af' in s: afs.append(s['af'])
            if 'wtsd' in s: wtsds.append(s['wtsd'])
        for match_path in h2h_dir.glob(f'*_vs_{label}.json'):
            # Skip self-match (shouldn't exist, but be safe)
            if match_path.name.startswith(f'{label}_vs_'):
                continue
            d = json.loads(match_path.read_text())
            edges.append(-d.get('edge_a', 0.0))  # flip sign: this model is B
            s = d.get('stats_b', {})
            if 'vpip' in s: vpips.append(s['vpip'])
            if 'pfr' in s: pfrs.append(s['pfr'])
            if 'cbet' in s: cbets.append(s['cbet'])
            if 'af' in s: afs.append(s['af'])
            if 'wtsd' in s: wtsds.append(s['wtsd'])
        if edges:
            row['avg_h2h_edge'] = sum(edges) / len(edges)
            row['h2h_matches'] = len(edges)
            row['h2h_wins'] = sum(1 for e in edges if e > 3)
            row['h2h_losses'] = sum(1 for e in edges if e < -3)
            row['h2h_ties'] = sum(1 for e in edges if -3 <= e <= 3)
        if vpips: row['avg_vpip'] = sum(vpips) / len(vpips)
        if pfrs: row['avg_pfr'] = sum(pfrs) / len(pfrs)
        if cbets: row['avg_cbet'] = sum(cbets) / len(cbets)
        if afs: row['avg_af'] = sum(afs) / len(afs)
        if wtsds: row['avg_wtsd'] = sum(wtsds) / len(wtsds)
        summary[label] = row

    # Save summary JSON
    out_json = RESULTS_DIR / 'summary.json'
    out_json.write_text(json.dumps(summary, indent=2))

    # Save human-readable CSV
    out_csv = RESULTS_DIR / 'summary.csv'
    cols = ['model', 'spot_pct', 'hh_bb_per_hand', 'hh_agreement_pct',
            'avg_h2h_edge', 'h2h_wins', 'h2h_ties', 'h2h_losses',
            'avg_vpip', 'avg_pfr', 'avg_cbet', 'avg_af', 'avg_wtsd']
    lines = [','.join(cols)]
    for label in sorted(summary.keys(),
                         key=lambda k: (int(k.split('_')[1]), k)):
        row = summary[label]
        vals = []
        for c in cols:
            v = row.get(c)
            if v is None:
                vals.append('')
            elif isinstance(v, float):
                vals.append(f'{v:.3f}')
            else:
                vals.append(str(v))
        lines.append(','.join(vals))
    out_csv.write_text('\n'.join(lines))

    return summary


def print_summary_table(summary: Dict) -> None:
    """Pretty-print the summary table, sorted by HH BB/hand desc."""
    print('\n' + '=' * 110)
    print(' TOURNAMENT SUMMARY — sorted by HH BB/hand')
    print('=' * 110)
    header = (f'{"Model":<10} {"Spot%":>7} {"HH BB/h":>8} {"Agree%":>7} '
              f'{"H2H_edge":>9} {"W-T-L":>9} {"VPIP":>6} {"PFR":>6} '
              f'{"CBet":>6} {"AF":>6} {"WTSD":>6}')
    print(header)
    print('-' * 110)
    rows = sorted(summary.values(),
                   key=lambda r: r.get('hh_bb_per_hand') or -99, reverse=True)
    for r in rows:
        def fmt(v, w, fmt_str='.2f'):
            if v is None:
                return ' ' * (w - 1) + '-'
            return f'{v:{w}{fmt_str}}'
        wtl = f"{r.get('h2h_wins', 0)}-{r.get('h2h_ties', 0)}-{r.get('h2h_losses', 0)}"
        line = (f'{r["model"]:<10} '
                f'{fmt(r.get("spot_pct"), 7, ".1f")} '
                f'{fmt(r.get("hh_bb_per_hand"), 8, ".3f")} '
                f'{fmt(r.get("hh_agreement_pct"), 7, ".1f")} '
                f'{fmt(r.get("avg_h2h_edge"), 9, ".2f")} '
                f'{wtl:>9} '
                f'{fmt(r.get("avg_vpip"), 6, ".1f")} '
                f'{fmt(r.get("avg_pfr"), 6, ".1f")} '
                f'{fmt(r.get("avg_cbet"), 6, ".1f")} '
                f'{fmt(r.get("avg_af"), 6, ".2f")} '
                f'{fmt(r.get("avg_wtsd"), 6, ".1f")}')
        print(line)
    print('-' * 110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', default='models/v12')
    ap.add_argument('--output-dir', default='tournament_results',
                    help='Where to write per-test JSONs + summary. Use a '
                         'different dir per run to avoid mixing results '
                         'across model versions (e.g. tournament_v13_results).')
    ap.add_argument('--min-iter', type=int, default=2600)
    ap.add_argument('--max-iter', type=int, default=5000)
    ap.add_argument('--h2h-hands', type=int, default=30000)
    ap.add_argument('--hh-hands', type=int, default=40000)
    ap.add_argument('--skip-spots', action='store_true')
    ap.add_argument('--skip-hh', action='store_true')
    ap.add_argument('--skip-h2h', action='store_true')
    ap.add_argument('--h2h-one-direction', action='store_true',
                    help='Only run A vs B (skip B vs A) to halve runtime — '
                         'asymmetric but still comparable if averaged across many opponents')
    ap.add_argument('--aggregate-only', action='store_true',
                    help='Skip all tests, just rebuild summary from existing JSONs')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    global RESULTS_DIR
    RESULTS_DIR = Path(args.output_dir) if Path(args.output_dir).is_absolute() \
        else ROOT / args.output_dir
    for sub in ('spots', 'hh', 'h2h'):
        (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)
    print(f'Results dir: {RESULTS_DIR}')

    models = discover_models(Path(args.models_dir), args.min_iter, args.max_iter)
    print(f'Found {len(models)} models in iter range [{args.min_iter}, {args.max_iter}]')
    for label, _ in models:
        print(f'  {label}')

    if args.aggregate_only:
        summary = aggregate(models)
        print_summary_table(summary)
        print(f'\nSaved: {RESULTS_DIR}/summary.json, {RESULTS_DIR}/summary.csv')
        return

    t0 = time.time()
    n_tests_total = 0
    n_tests_skipped = 0

    # Phase 1: Spot scores
    if not args.skip_spots:
        print(f'\n[Phase 1/3] Spot scores ({len(models)} tests)')
        for i, (label, path) in enumerate(models, 1):
            output = RESULTS_DIR / 'spots' / f'{label}_spots.json'
            prefix = f'  [{i}/{len(models)}] {label}'
            if output.exists():
                print(f'{prefix}: SKIP (exists)')
                n_tests_skipped += 1
                continue
            n_tests_total += 1
            print(f'{prefix}: running...')
            t = time.time()
            run_spot_score(label, path, output, dry_run=args.dry_run)
            print(f'{prefix}: done in {time.time()-t:.0f}s')

    # Phase 2: HH validation
    if not args.skip_hh:
        print(f'\n[Phase 2/3] HH validation ({len(models)} tests, {args.hh_hands} hands each)')
        for i, (label, path) in enumerate(models, 1):
            output = RESULTS_DIR / 'hh' / f'{label}_hh.json'
            prefix = f'  [{i}/{len(models)}] {label}'
            if output.exists():
                print(f'{prefix}: SKIP (exists)')
                n_tests_skipped += 1
                continue
            n_tests_total += 1
            print(f'{prefix}: running ({args.hh_hands} hands)...')
            t = time.time()
            run_hh(label, path, args.hh_hands, output, dry_run=args.dry_run)
            print(f'{prefix}: done in {time.time()-t:.0f}s')

    # Phase 3: H2H tournament
    if not args.skip_h2h:
        pairs = []
        for i, a in enumerate(models):
            for j, b in enumerate(models):
                if i == j:
                    continue
                if args.h2h_one_direction and i > j:
                    continue
                pairs.append((a, b))
        print(f'\n[Phase 3/3] H2H tournament ({len(pairs)} matches, {args.h2h_hands} hands each)')
        for k, ((la, pa), (lb, pb)) in enumerate(pairs, 1):
            output = RESULTS_DIR / 'h2h' / f'{la}_vs_{lb}.json'
            prefix = f'  [{k}/{len(pairs)}] {la} vs {lb}'
            if output.exists():
                print(f'{prefix}: SKIP (exists)')
                n_tests_skipped += 1
                continue
            n_tests_total += 1
            print(f'{prefix}: running ({args.h2h_hands} hands)...')
            t = time.time()
            run_h2h(la, pa, lb, pb, args.h2h_hands, output, dry_run=args.dry_run)
            print(f'{prefix}: done in {time.time()-t:.0f}s')

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed/60:.1f} min. Ran {n_tests_total} tests, skipped {n_tests_skipped}.')

    # Phase 4: Aggregate
    summary = aggregate(models)
    print_summary_table(summary)
    print(f'\nResults saved to:')
    print(f'  {RESULTS_DIR}/summary.json')
    print(f'  {RESULTS_DIR}/summary.csv')
    print(f'  Per-test JSONs in {RESULTS_DIR}/spots/, hh/, h2h/')


if __name__ == '__main__':
    main()
