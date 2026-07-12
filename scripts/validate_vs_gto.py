"""GTO leak-finder: compare your logged play to the TexasSolver strategy.

Reads spots that `scripts/solve_spots_gto.py` annotated with `gto_strategy`
(+ `gto_action`, `gto_freq_played`) and compares them to what you actually
played (`action_idx`). TexasSolver's console emits strategy FREQUENCIES, not
per-hand EV, so the leak metric is strategy-based:

    agreement       = (your action == GTO's most-frequent action)
    gto_freq_played = how often GTO takes the action you took (0..1)
    leak_score      = gto_strategy[gto_action] - gto_freq_played

`leak_score` is ~0 when you played GTO's top action OR the spot is a near-mix
(no clearly-best action — low severity), and large when there was a clear best
action you skipped. Leaks are grouped by street / action-context / your→GTO
action pair and ranked by total leak_score, surfacing systematic mistakes
("you fold rivers GTO calls") rather than one-off coin-flips.

Usage:
    python3 scripts/validate_vs_gto.py
    python3 scripts/validate_vs_gto.py --min-leak 0.3
    python3 scripts/validate_vs_gto.py --list "Fold->Call" --limit 20
    python3 scripts/validate_vs_gto.py --by river            # only river spots

Caveats: ranges come from a showdown-biased empirical DB (or tier fallback);
this measures "you vs GTO-against-an-empirical-range," tagged per spot by
`gto_range_source` (empirical / mixed / tier_fallback). See docs/ / the plan.
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

ACTION_NAMES = {
    0: 'Fold', 1: 'Check/Call', 2: 'Raise½', 3: 'RaisePot',
    4: 'AllIn', 5: 'RaiseMin', 7: 'Raise⅓',
}
# Logged actions that aren't model outputs map onto a GTO bucket for comparison.
_PLAYED_REMAP = {5: 2, 7: 2}


def _played_idx(action_idx):
    if action_idx is None or action_idx < 0:
        return None
    return _PLAYED_REMAP.get(action_idx, action_idx)


def _short(action_idx):
    return ACTION_NAMES.get(action_idx, f'a{action_idx}')


def load_spots(spots_dir, street=None, source=None):
    """Yield (path, data) for solved, non-deleted spots in scope."""
    for path in sorted(glob.glob(os.path.join(spots_dir, '*.json'))):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get('deleted'):
            continue
        if 'gto_strategy' not in data or 'gto_action' not in data:
            continue
        if data.get('action_idx') is None:
            continue
        key = data.get('gto_spot_key') or []
        spot_street = key[0] if key else (data.get('game_state') or {}).get('street')
        if street and spot_street != street:
            continue
        if source and data.get('gto_range_source') != source:
            continue
        yield path, data


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spots-dir', default='logs/action_debug')
    ap.add_argument('--by', default=None,
                    help='restrict to a single street (river/turn/flop)')
    ap.add_argument('--source', default=None,
                    help='restrict to a gto_range_source '
                         '(empirical/mixed/tier_fallback)')
    ap.add_argument('--min-leak', type=float, default=0.25,
                    help='leak_score threshold for the leak table (default .25)')
    ap.add_argument('--list', default=None, metavar='PAIR',
                    help='list spots for a "Played->GTO" pair, e.g. "Fold->Call"')
    ap.add_argument('--limit', type=int, default=15)
    args = ap.parse_args()

    spots = list(load_spots(args.spots_dir, args.by, args.source))
    if not spots:
        print("No GTO-solved spots found. Run scripts/solve_spots_gto.py first.")
        return

    n = 0
    n_agree = 0
    sum_freq_played = 0.0
    by_street = defaultdict(lambda: [0, 0, 0.0])        # [n, agree, sum_freq]
    by_context = defaultdict(lambda: [0, 0, 0.0])
    by_source = defaultdict(lambda: [0, 0, 0.0])
    # leak buckets: (street, context, played_name->gto_name) -> [count, sum_leak]
    leaks = defaultdict(lambda: [0, 0.0])
    listed = []

    for path, d in spots:
        gto_strat = d['gto_strategy']
        gto_action = int(d['gto_action'])
        played_raw = int(d['action_idx'])
        played = _played_idx(played_raw)
        if played is None:
            continue
        key = d.get('gto_spot_key') or ['?', '?', '?', '?', '?']
        street = key[0]
        context = key[4] if len(key) > 4 else '?'
        source = d.get('gto_range_source', '?')
        freq_played = d.get('gto_freq_played')
        if freq_played is None:
            freq_played = (gto_strat[played] if 0 <= played < len(gto_strat) else 0.0)

        agree = (played == gto_action)
        gto_top_freq = gto_strat[gto_action] if 0 <= gto_action < len(gto_strat) else 0.0
        leak = max(0.0, gto_top_freq - freq_played)

        n += 1
        n_agree += int(agree)
        sum_freq_played += freq_played
        for agg, k in ((by_street, street), (by_context, context),
                       (by_source, source)):
            agg[k][0] += 1
            agg[k][1] += int(agree)
            agg[k][2] += freq_played

        if not agree and leak >= args.min_leak:
            pair = f'{_short(played_raw)}->{_short(gto_action)}'
            bk = (street, context, pair)
            leaks[bk][0] += 1
            leaks[bk][1] += leak
            if args.list and pair == args.list:
                listed.append((leak, path, d, freq_played, gto_top_freq))

    # ── List mode ──
    if args.list:
        listed.sort(reverse=True)
        print(f"\nSpots where you played {args.list} "
              f"(leak ≥ {args.min_leak}), worst first:\n")
        for leak, path, d, fp, topf in listed[:args.limit]:
            key = d.get('gto_spot_key') or []
            gs = d.get('game_state') or {}
            board = ''.join(c['rank'] + c['suit'] for c in gs.get('community_cards', []))
            print(f"  leak={leak:.2f}  {os.path.basename(path)}")
            print(f"     {key}  board={board}  src={d.get('gto_range_source')}")
            print(f"     GTO: " + ', '.join(
                f'{_short(i)}={v:.2f}' for i, v in enumerate(d['gto_strategy']) if v > 0.005))
            print(f"     your action freq in GTO={fp:.2f}, GTO-top freq={topf:.2f}")
        if not listed:
            print("  (none)")
        return

    # ── Summary ──
    print(f"\n=== GTO comparison: {n} solved spots "
          f"({args.spots_dir}{', '+args.by if args.by else ''}) ===")
    print(f"Action agreement with GTO mode : {n_agree}/{n} = {100*n_agree/max(n,1):.1f}%")
    print(f"Mean GTO freq of your action   : {sum_freq_played/max(n,1):.3f} "
          f"(1.0 = always played GTO's choice)")

    def _table(title, agg):
        print(f"\n{title}:")
        print(f"  {'bucket':<22} {'n':>5} {'agree%':>7} {'meanFreq':>9}")
        for k, (cnt, ag, sf) in sorted(agg.items(), key=lambda x: -x[1][0]):
            print(f"  {str(k):<22} {cnt:>5} {100*ag/max(cnt,1):>6.1f}% {sf/max(cnt,1):>9.3f}")

    _table("By street", by_street)
    _table("By action context", by_context)
    _table("By range source", by_source)

    print(f"\nBiggest systematic leaks (mismatch, leak ≥ {args.min_leak}, "
          f"ranked by total leak_score):")
    print(f"  {'street':<7} {'context':<18} {'your->GTO':<18} {'n':>4} {'Σleak':>7} {'avg':>6}")
    ranked = sorted(leaks.items(), key=lambda x: -x[1][1])
    for (street, ctx, pair), (cnt, sleak) in ranked[:args.limit]:
        print(f"  {street:<7} {ctx:<18} {pair:<18} {cnt:>4} {sleak:>7.2f} "
              f"{sleak/max(cnt,1):>6.2f}")
    if not ranked:
        print("  (no leaks above threshold — your play matched GTO closely)")
    print(f"\nTip: drill into one with  --list \"Fold->Call\" --limit 20")


if __name__ == '__main__':
    main()
