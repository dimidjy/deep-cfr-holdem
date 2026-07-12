"""Solve logged action_debug spots with TexasSolver and write gto_* fields.

For each in-scope spot (heads-up, postflop, streets in --streets) we recompute
the solve inputs (src/gto/spot_inputs.py), build hero+villain ranges from the
empirical opp-range DB with a tier fallback (src/gto/range_builder.py), solve
with TexasSolver (src/gto/solver.py — ONE binary invocation amortizes the ~24s
dictionary load across the whole batch), navigate the solved tree to hero's
node (src/gto/extract.py), and merge gto_* fields into the JSON preserving all
annotations (src/gto/writer.py).

Identical solves are de-duplicated by a solve signature
(board, pot, eff_stack, hero_range, villain_range, hero_is_oop) so repeated
spots share one solve. Already-solved spots are skipped unless --overwrite.

Examples:
    bash scripts/setup_texassolver.sh && export TEXASSOLVER_BIN=tools/texassolver/console_solver
    python3 scripts/solve_spots_gto.py --streets river --limit 5 --dry-run
    python3 scripts/solve_spots_gto.py --streets river --limit 20
    python3 scripts/solve_spots_gto.py --streets river,turn          # full pass
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gto.spot_inputs import build_gto_spot_inputs
from src.gto.range_builder import build_hero_and_villain_ranges
from src.gto.solver import GTOSolver, SolveSpec
from src.gto.extract import solve_result_to_gto_fields
from src.gto.writer import write_gto_into_spot
from src.opp_range_lookup import get_default_lookup


def _solve_signature(inputs, hero_range, villain_range) -> str:
    raw = '|'.join([
        ','.join(inputs.board), str(inputs.pot_chips),
        str(inputs.eff_stack_chips), str(inputs.hero_is_oop),
        # injected villain bet size changes the tree → part of the signature
        str(inputs.villain_bet_side), str(inputs.villain_bet_pct),
        hero_range, villain_range,
    ])
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spots-dir', default='logs/action_debug')
    ap.add_argument('--file', default=None, help='solve a single JSON file')
    ap.add_argument('--streets', default='river,turn',
                    help='comma list of streets to solve (river,turn[,flop])')
    ap.add_argument('--db', default='data/opp_range.sqlite')
    ap.add_argument('--binary', default=None, help='console_solver path '
                    '(default $TEXASSOLVER_BIN or tools/texassolver/console_solver)')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--accuracy', type=float, default=0.5)
    ap.add_argument('--max-iter', type=int, default=80)
    ap.add_argument('--bluff-factor', type=float, default=1.0,
                    help='exploit knob (1.0=pure GTO). <1 trims hero bluff '
                         'frequency on no-showdown-value hands and shifts it to '
                         'check/fold (e.g. 0.3 for a calling-station pool, 0=no bluffs)')
    ap.add_argument('--limit', type=int, default=None,
                    help='cap number of spots considered')
    ap.add_argument('--overwrite', action='store_true',
                    help='re-solve spots that already have gto_strategy')
    ap.add_argument('--dry-run', action='store_true',
                    help='print solve specs / ranges, do not run the solver')
    ap.add_argument('--keep-tmp', default=None,
                    help='keep solver dumps in this dir (debug)')
    args = ap.parse_args()

    streets = tuple(s.strip() for s in args.streets.split(',') if s.strip())

    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(os.path.join(args.spots_dir, '*.json')))
    if args.limit:
        files = files[:args.limit]

    lookup = get_default_lookup(args.db)
    if lookup is None:
        print(f"WARNING: opp-range DB not found at {args.db}; "
              f"all ranges will use the tier fallback.")

    # ── Phase 1: build inputs + ranges, dedup by solve signature ──
    # sig -> SolveSpec ; sig -> list of (path, inputs) sharing that solve
    specs_by_sig = {}
    members = defaultdict(list)
    n_scanned = n_inscope = n_skipped_done = n_no_inputs = 0

    for path in files:
        n_scanned += 1
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        gs = data.get('game_state') or {}
        if gs.get('street') not in streets:
            continue
        if not args.overwrite and 'gto_strategy' in data:
            n_skipped_done += 1
            continue
        inputs = build_gto_spot_inputs(data, scope_streets=streets)
        if inputs is None:
            n_no_inputs += 1
            continue
        n_inscope += 1
        hero_range, villain_range, source = build_hero_and_villain_ranges(
            inputs.hero_spot_key, inputs.villain_spot_key, lookup,
            inputs.hero_hand_class, inputs.aggression_tier,
            hero_hand_combo=inputs.hero_hand_combo)
        sig = _solve_signature(inputs, hero_range, villain_range)
        members[sig].append((path, inputs, source))
        if sig not in specs_by_sig:
            specs_by_sig[sig] = SolveSpec(
                solve_id=sig, inputs=inputs,
                hero_range=hero_range, villain_range=villain_range)

    print(f"Scanned {n_scanned} | in-scope {n_inscope} | "
          f"already-solved {n_skipped_done} | no-inputs {n_no_inputs}")
    print(f"Unique solves: {len(specs_by_sig)} "
          f"(dedup factor {n_inscope / max(len(specs_by_sig),1):.1f}x)")

    if not specs_by_sig:
        print("Nothing to solve.")
        return

    if args.dry_run:
        for i, (sig, spec) in enumerate(list(specs_by_sig.items())[:8]):
            inp = spec.inputs
            print(f"\n[{i}] sig={sig} board={inp.board} street={inp.street} "
                  f"nav={inp.nav_type} hero_oop={inp.hero_is_oop} "
                  f"pot={inp.pot_chips} eff={inp.eff_stack_chips} "
                  f"({len(members[sig])} spots)")
            print(f"    hero_range[:120]={spec.hero_range[:120]}")
            print(f"    vill_range[:120]={spec.villain_range[:120]}")
        print(f"\n(dry-run: {len(specs_by_sig)} solves would run)")
        return

    # ── Phase 2: solve the unique set in one (amortized) batch ──
    solver = GTOSolver(binary=args.binary, threads=args.threads,
                       accuracy=args.accuracy, max_iteration=args.max_iter)
    if not solver.available:
        print(f"ERROR: console_solver not found ({solver._resolved_bin}). "
              f"Run scripts/setup_texassolver.sh or set $TEXASSOLVER_BIN.")
        sys.exit(1)
    print(f"Solving {len(specs_by_sig)} unique spots via {solver.version()} ...")
    t0 = time.time()
    results = solver.run_batch(list(specs_by_sig.values()),
                               workdir=args.keep_tmp, verbose=True)
    print(f"Solver finished in {time.time()-t0:.1f}s "
          f"({len(results)}/{len(specs_by_sig)} solved)")

    # ── Phase 3: extract + write back per spot ──
    n_written = n_err = 0
    for sig, spec in specs_by_sig.items():
        result = results.get(sig)
        for (path, inputs, source) in members[sig]:
            if result is None:
                fields = {'gto_error': 'solver did not produce a result',
                          'gto_range_source': source,
                          'gto_spot_key': list(inputs.hero_spot_key.as_tuple())}
            else:
                fields = solve_result_to_gto_fields(
                    result, inputs, bluff_factor=args.bluff_factor)
                fields['gto_range_source'] = source
            fields['gto_solver_version'] = solver.version()
            try:
                write_gto_into_spot(path, fields)
            except Exception as exc:
                print(f"  write failed {path}: {exc}")
                continue
            if 'gto_error' in fields:
                n_err += 1
            else:
                n_written += 1

    print(f"Wrote gto_* into {n_written + n_err} spots "
          f"({n_written} solved, {n_err} errored).")


if __name__ == '__main__':
    main()
