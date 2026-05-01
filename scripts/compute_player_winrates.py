"""Compute per-player winrates from a PokerStars NL5 hand history archive.

Walks `data/poker_archive/`, aggregates per-player money in (bets) and money
out (collected from pot), subtracts uncalled-bet returns, and computes net
BB and BB/100 over hands played. Writes a sorted CSV plus a good-players
text file consumable by the existing `--good-players` loader in
`scripts/validate_vs_hh.py:1598` and our preflop lookup builder.

Filtering:
  - Only NL5 hands (big_blind == $0.05)
  - Only 6-max table sizes (num_active_players == 6)
  - Only players who actually saw cards in the hand (had any action,
    including a fold or blind post)

Money accounting:
  Net = collected - committed + uncalled-returned
  - Committed: per street per player, raises overwrite (cumulative),
    blinds/calls/bets accumulate (incremental).
  - Uncalled: parser ignores "Uncalled bet ... returned to" lines, so we
    re-scan the raw file for them.

Usage:
    python3 scripts/compute_player_winrates.py
    python3 scripts/compute_player_winrates.py --min-hands 20000 --min-bb100 10
"""
import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hand_history.parser import parse_file
from src.preflop_spot_key import BB_USD, MIN_HANDS, MIN_BB100


# Same regex the parser declares but discards.
_RE_UNCALLED = re.compile(r"Uncalled bet \(\$([0-9.]+)\) returned to (.+?)\s*$")


def player_commit_per_hand(hand) -> dict:
    """Per-player gross dollar commitment for one ParsedHand.

    Walks each (player, street) action group:
      - raise:     cumulative (action.amount_dollars is the "to" value)
      - blind/call/bet: incremental (action.amount_dollars is what was added)
      - fold/check: no money in
    """
    by_player_street = defaultdict(lambda: defaultdict(list))
    for a in hand.actions:
        by_player_street[a.player_name][a.street].append(a)

    commit = defaultdict(float)
    for name, streets in by_player_street.items():
        for street, actions in streets.items():
            street_commit = 0.0
            for a in actions:
                if a.action_type == 'raise':
                    street_commit = a.amount_dollars
                elif a.action_type in ('call', 'bet', 'blind_sb', 'blind_bb'):
                    street_commit += a.amount_dollars
            commit[name] += street_commit
    return commit


def scan_uncalled(path: Path) -> dict:
    """Sum uncalled-bet returns per player across one HH file."""
    uncalled = defaultdict(float)
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = _RE_UNCALLED.search(line)
                if m:
                    uncalled[m.group(2).strip()] += float(m.group(1))
    except Exception:
        pass
    return uncalled


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--hh-dir', default='data/poker_archive')
    ap.add_argument('--out-csv', default='data/player_winrates.csv')
    ap.add_argument('--out-good-players',
                    default='data/good_players_winrate10.txt')
    ap.add_argument('--min-hands', type=int, default=MIN_HANDS,
                    help=f'Minimum hands to qualify (default {MIN_HANDS})')
    ap.add_argument('--min-bb100', type=float, default=MIN_BB100,
                    help=f'Minimum BB/100 to qualify (default {MIN_BB100})')
    ap.add_argument('--max-files', type=int, default=0,
                    help='Limit files for testing (0 = all)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    hh_dir = Path(args.hh_dir)
    if not hh_dir.exists():
        sys.exit(f'HH dir not found: {hh_dir}')

    files = sorted(hh_dir.rglob('*.txt'))
    if args.max_files > 0:
        files = files[:args.max_files]
    print(f'Found {len(files)} HH files in {hh_dir}')

    bets: dict = defaultdict(float)
    winnings: dict = defaultdict(float)
    uncalled_tot: dict = defaultdict(float)
    hands_played: dict = defaultdict(int)
    skipped_nonNL5 = 0
    skipped_not6max = 0

    t0 = time.time()
    for fi, path in enumerate(files):
        # Pass 1: uncalled returns from raw lines
        for name, amt in scan_uncalled(path).items():
            uncalled_tot[name] += amt

        # Pass 2: parse + aggregate
        try:
            hands = parse_file(str(path))
        except Exception:
            continue

        for h in hands:
            if abs(h.big_blind_dollars - BB_USD) > 0.001:
                skipped_nonNL5 += 1
                continue
            # Use max_seats (table type) not num_active_players (live count)
            # so we count winrate across all hands at a 6-max table, including
            # hands with a player sitting out.
            if h.max_seats != 6:
                skipped_not6max += 1
                continue

            seen = {a.player_name for a in h.actions}
            for name in seen:
                hands_played[name] += 1

            for name, amt in player_commit_per_hand(h).items():
                bets[name] += amt
            for w in h.result.winners:
                winnings[w['name']] += w['amount_dollars']

        if not args.quiet and (fi + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            eta = (len(files) - fi - 1) / rate if rate > 0 else 0
            print(f'  [{fi+1}/{len(files)}] {rate:.0f} files/s '
                  f'({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)')

    print(f'Done in {time.time()-t0:.0f}s. '
          f'NL5-6max hands per player: median '
          f'{sorted(hands_played.values())[len(hands_played)//2] if hands_played else 0}')
    print(f'  Skipped non-NL5: {skipped_nonNL5}, non-6max: {skipped_not6max}')

    rows = []
    for name, n in hands_played.items():
        gross_in = bets[name]
        gross_out = winnings[name]
        unc = uncalled_tot[name]
        net_dollars = gross_out - gross_in + unc
        net_bb = net_dollars / BB_USD
        bb_per_100 = net_bb * 100.0 / n if n > 0 else 0.0
        rows.append({
            'name': name,
            'hands': n,
            'gross_in_dollars': round(gross_in, 4),
            'gross_out_dollars': round(gross_out, 4),
            'uncalled_dollars': round(unc, 4),
            'net_dollars': round(net_dollars, 4),
            'net_bb': round(net_bb, 2),
            'bb_per_100': round(bb_per_100, 2),
        })
    rows.sort(key=lambda r: r['bb_per_100'], reverse=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f'Wrote {len(rows)} player winrates to {out_csv}')

    cohort = [r['name'] for r in rows
              if r['hands'] >= args.min_hands and r['bb_per_100'] >= args.min_bb100]
    out_good = Path(args.out_good_players)
    out_good.parent.mkdir(parents=True, exist_ok=True)
    with open(out_good, 'w') as f:
        f.write('\n'.join(cohort) + ('\n' if cohort else ''))
    print(f'Cohort: {len(cohort)} players '
          f'(hands >= {args.min_hands}, bb/100 >= {args.min_bb100}) → {out_good}')

    # Quick sanity print
    if rows:
        print('\nTop 10 players by BB/100 (with min sample filter applied):')
        eligible = [r for r in rows if r['hands'] >= args.min_hands]
        for r in eligible[:10]:
            print(f"  {r['name']:<25} hands={r['hands']:>6}  "
                  f"net_bb={r['net_bb']:>+9.0f}  bb/100={r['bb_per_100']:>+6.2f}")


if __name__ == '__main__':
    main()
