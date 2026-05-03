"""Inspect Claude annotations side-by-side with the model's actual actions.

Use after running `annotate_spots_with_claude.py` to see:
  - agreement rate (how often Claude picked the same action as the model)
  - per-spot reasoning so you can sanity-check Claude's strategy
  - confidence distribution (how decisive Claude was)
  - flagged spots where Claude disagrees AND has high confidence — these
    are the highest-signal disagreements for imitation training

Usage:
    python3 scripts/verify_claude_annotations.py
    python3 scripts/verify_claude_annotations.py --show-all
    python3 scripts/verify_claude_annotations.py --show-disagreements
    python3 scripts/verify_claude_annotations.py --min-confidence 0.8
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ACTION_NAMES = {0: 'fold', 1: 'call', 2: 'bet_half', 3: 'bet_pot',
                4: 'all_in', 5: 'bet_min', 7: 'bet_third'}


def kind(action_idx, action_name=''):
    """Bucket actions for agreement comparison: fold | call | raise."""
    if action_idx == 0:
        return 'fold'
    if action_idx == 1:
        return 'call'
    if action_idx in (2, 3, 4, 5, 7) or 'raise' in action_name:
        return 'raise'
    return f'a{action_idx}'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spots-dir', default='logs/action_debug')
    ap.add_argument('--show-all', action='store_true',
                    help='Print every spot, not just summary')
    ap.add_argument('--show-disagreements', action='store_true',
                    help='Print only spots where Claude disagrees with model')
    ap.add_argument('--min-confidence', type=float, default=0.0,
                    help='Filter to Claude annotations with confidence >= this')
    args = ap.parse_args()

    spots_dir = Path(args.spots_dir)
    annotated = []
    for p in sorted(spots_dir.glob('*.json')):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if 'claude_reasoning' not in d:
            continue
        if d.get('claude_confidence', 0) < args.min_confidence:
            continue
        annotated.append((p.name, d))

    if not annotated:
        print('No Claude-annotated spots found.')
        print(f'Run: python3 scripts/annotate_spots_with_claude.py --mode single --max-spots 5')
        return 1

    # ── Aggregate stats ───────────────────────────────────────────────
    n = len(annotated)
    agreements = 0
    confidences = []
    claude_action_dist = Counter()
    model_action_dist = Counter()
    by_street = Counter()
    disagreements = []

    for name, d in annotated:
        gs = d['game_state']
        model_action = d.get('action_idx', -1)
        model_kind = kind(model_action, d.get('action_name', ''))
        claude_action = d['right_action']
        claude_kind = kind(claude_action)
        confidences.append(d['claude_confidence'])
        claude_action_dist[ACTION_NAMES.get(claude_action, '?')] += 1
        model_action_dist[ACTION_NAMES.get(model_action, '?')] += 1
        by_street[gs.get('street', '?')] += 1
        if model_kind == claude_kind:
            agreements += 1
        else:
            disagreements.append((name, d, model_kind, claude_kind))

    print(f'\n{"="*72}\nCLAUDE ANNOTATION VERIFICATION\n{"="*72}')
    print(f'Annotated spots:       {n}')
    print(f'Agreement (kind):      {agreements}/{n} ({100*agreements/n:.1f}%)')
    print(f'Disagreements:         {len(disagreements)}')
    print(f'  Streets covered:     {dict(by_street)}')
    print(f'  Confidence avg:      {sum(confidences)/len(confidences):.2f}')
    print(f'  Confidence ≥ 0.85:   {sum(1 for c in confidences if c >= 0.85)}/{n}')
    print(f'  Confidence ≤ 0.65:   {sum(1 for c in confidences if c <= 0.65)}/{n}')
    print(f'\nClaude action distribution: {dict(claude_action_dist)}')
    print(f'Model action distribution:  {dict(model_action_dist)}')

    # ── Per-spot detail ───────────────────────────────────────────────
    spots_to_show = (
        annotated if args.show_all else
        [(n, d) for n, d, _, _ in disagreements] if args.show_disagreements else
        []  # default: only summary
    )

    if spots_to_show:
        print(f'\n{"-"*72}\nPer-spot detail ({len(spots_to_show)} spots):\n{"-"*72}')
        for name, d in spots_to_show:
            gs = d['game_state']
            f = d.get('features', {})
            me = next((p for p in gs['players']
                       if p['seat'] == gs['my_seat']), {})
            hole = ''.join(c['rank']+c['suit'] for c in me.get('hole_cards', []))
            board = ' '.join(c['rank']+c['suit']
                             for c in gs.get('community_cards', [])) or '-'
            model_a = ACTION_NAMES.get(d.get('action_idx', -1), '?')
            claude_a = ACTION_NAMES.get(d['right_action'], '?')
            agree = '✓' if (kind(d.get('action_idx', -1), d.get('action_name', ''))
                            == kind(d['right_action'])) else '✗'
            print(f'\n  {agree}  {name}')
            print(f'     {gs["street"]:<7} {hole} on [{board}]  pot={gs["pot_total"]:.1f}BB call={gs["call_amount"]:.1f}BB')
            print(f'     model:  {model_a:<10}    claude: {claude_a:<10} (conf {d["claude_confidence"]:.2f})')
            print(f'     why:    {d["claude_reasoning"]}')

    # ── Highest-signal disagreements ──────────────────────────────────
    if disagreements:
        high_conf_disagree = [(n, d, mk, ck) for n, d, mk, ck in disagreements
                               if d['claude_confidence'] >= 0.8]
        if high_conf_disagree and not args.show_disagreements:
            print(f'\n{"-"*72}')
            print(f'HIGH-CONFIDENCE DISAGREEMENTS ({len(high_conf_disagree)}) — '
                  f'best candidates for imitation training:\n{"-"*72}')
            for name, d, mk, ck in high_conf_disagree[:10]:
                gs = d['game_state']
                me = next((p for p in gs['players']
                           if p['seat'] == gs['my_seat']), {})
                hole = ''.join(c['rank']+c['suit'] for c in me.get('hole_cards', []))
                board = ' '.join(c['rank']+c['suit']
                                  for c in gs.get('community_cards', [])) or '-'
                print(f'  {gs["street"]:<7} {hole} on [{board}]  '
                      f'model={mk}  claude={ck} (conf {d["claude_confidence"]:.2f})')
                print(f'     "{d["claude_reasoning"]}"')

    return 0


if __name__ == '__main__':
    sys.exit(main())
