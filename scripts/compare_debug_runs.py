"""Side-by-side comparison of two debug log directories.

Reads JSONL events from each dir (one file per PID), aggregates events by
type, and emits a markdown report listing:
  - Total event counts per run
  - Worker init configs (sanity check)
  - obs[53] distribution: should be present in WITH-feature, absent in WITHOUT
  - Shaper firing rates (should be ~identical across runs if shapers use obs[18])
  - Per-iteration losses
  - Opponent action source distribution

Usage:
    python3 scripts/compare_debug_runs.py \\
        --with-feature logs/debug_with \\
        --without-feature logs/debug_without
"""
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


def load_events(dir_path: Path) -> list:
    """Read all JSONL files in a dir, sort by timestamp."""
    events = []
    for f in sorted(dir_path.glob('*.jsonl')):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    events.sort(key=lambda e: e.get('t', 0.0))
    return events


def by_event(events: list) -> dict:
    """Group events by 'event' field."""
    grouped = defaultdict(list)
    for e in events:
        grouped[e.get('event', '?')].append(e)
    return dict(grouped)


def summarize_field(events, field, n_decimals=3):
    """Mean/std/percentiles of a numeric field across events."""
    vals = []
    for e in events:
        v = e.get(field)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(float(v))
    if not vals:
        return None
    arr = np.array(vals)
    return {
        'n': len(arr),
        'mean': round(float(arr.mean()), n_decimals),
        'std': round(float(arr.std()), n_decimals),
        'min': round(float(arr.min()), n_decimals),
        'max': round(float(arr.max()), n_decimals),
        'p10': round(float(np.percentile(arr, 10)), n_decimals),
        'p50': round(float(np.percentile(arr, 50)), n_decimals),
        'p90': round(float(np.percentile(arr, 90)), n_decimals),
    }


def render_section(title: str, *parts: str):
    """Just a md heading + body."""
    print(f'\n## {title}\n')
    for p in parts:
        print(p)


def fmt_summary(s):
    if s is None:
        return '(no data)'
    return (f'n={s["n"]}  mean={s["mean"]}  std={s["std"]}  '
            f'p10/p50/p90={s["p10"]}/{s["p50"]}/{s["p90"]}  '
            f'min={s["min"]} max={s["max"]}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--with-feature', required=True, type=Path)
    p.add_argument('--without-feature', required=True, type=Path)
    args = p.parse_args()

    print('# Debug run comparison report\n')
    print(f'- **WITH feature**:    `{args.with_feature}`')
    print(f'- **WITHOUT feature**: `{args.without_feature}`')

    events_with = load_events(args.with_feature)
    events_without = load_events(args.without_feature)
    by_with = by_event(events_with)
    by_without = by_event(events_without)

    # ── Event count summary ─────────────────────────────────────
    all_event_types = set(by_with) | set(by_without)
    rows = []
    rows.append(f'| Event | WITH count | WITHOUT count | Δ |')
    rows.append(f'|---|---:|---:|---:|')
    for et in sorted(all_event_types):
        a = len(by_with.get(et, []))
        b = len(by_without.get(et, []))
        rows.append(f'| `{et}` | {a} | {b} | {a - b:+d} |')
    render_section('Event counts', '\n'.join(rows))

    # ── Worker init: verify config differs only on threat_range_equity ─
    def worker_init_summary(events_dict):
        wi = events_dict.get('worker_init', [])
        if not wi:
            return '(none)'
        first = wi[0]
        keys = ['threat_range_equity', 'kbest_pool_size', 'num_players',
                'archetype_mode', 'deck_reshuffle',
                'bluff_penalty', 'call_bonus', 'value_bonus', 'fold_penalty',
                'draw_call_boost', 'draw_bet_boost',
                'barrel_fold_bonus', 'mid_pair_fold_bonus']
        return '  ' + '  '.join(f'{k}={first.get(k)}' for k in keys)

    render_section('Worker init configs (first worker each run)',
                   f'**WITH**:    {worker_init_summary(by_with)}',
                   f'',
                   f'**WITHOUT**: {worker_init_summary(by_without)}')

    # ── Obs feature presence ────────────────────────────────────
    def obs_53_stats(events_dict):
        starts = events_dict.get('branch_start', [])
        return summarize_field(starts, 'obs_53')

    def obs_18_stats(events_dict):
        starts = events_dict.get('branch_start', [])
        return summarize_field(starts, 'obs_18')

    render_section('Obs feature distributions (branch_start samples)',
        f'**obs[18] (hand_strength vs random)**:',
        f'  WITH    : {fmt_summary(obs_18_stats(by_with))}',
        f'  WITHOUT : {fmt_summary(obs_18_stats(by_without))}',
        '',
        f'**obs[53] (threat_range_equity)**:',
        f'  WITH    : {fmt_summary(obs_53_stats(by_with))}',
        f'  WITHOUT : {fmt_summary(obs_53_stats(by_without))}',
        '',
        '  ⇒ WITHOUT should have all obs_53 = null (feature disabled).')

    # ── Shaper firings ───────────────────────────────────────
    shapers = ['bluff_penalty_chips', 'call_bonus_chips',
                'value_bonus_chips', 'fold_penalty_chips',
                'draw_call_boost_chips', 'draw_bet_boost_chips',
                'barrel_fold_bonus_chips', 'mid_pair_fold_bonus_chips']

    def shaper_fire_rates(events_dict):
        shapers_events = events_dict.get('branch_shapers', [])
        total = len(shapers_events)
        if total == 0:
            return {s: (0, 0) for s in shapers}, total
        result = {}
        for s in shapers:
            fired = sum(1 for e in shapers_events if e.get(s, 0.0) > 0.0)
            mean_mag = np.mean([float(e.get(s, 0.0))
                                 for e in shapers_events if e.get(s, 0.0) > 0.0]) if fired else 0.0
            result[s] = (fired, mean_mag)
        return result, total

    rates_with, tot_with = shaper_fire_rates(by_with)
    rates_without, tot_without = shaper_fire_rates(by_without)

    lines = ['| Shaper | WITH fire% | WITH mean mag | WITHOUT fire% | WITHOUT mean mag | Δ fire% |']
    lines.append('|---|---:|---:|---:|---:|---:|')
    for s in shapers:
        fw, mw = rates_with[s]
        fo, mo = rates_without[s]
        pct_w = 100 * fw / tot_with if tot_with else 0
        pct_o = 100 * fo / tot_without if tot_without else 0
        lines.append(f'| {s} | {pct_w:.1f}% | {mw:.2f} | {pct_o:.1f}% | {mo:.2f} | '
                     f'{pct_w - pct_o:+.1f} |')
    render_section(f'Shaper firing rates (branch_shapers events: '
                   f'WITH={tot_with}, WITHOUT={tot_without})',
                   '\n'.join(lines),
                   '\n  ⇒ Fire rates should be ~identical if shapers use obs[18] only.')

    # ── Opponent action source distribution ─────────────────────
    def opp_source_counts(events_dict):
        opp = events_dict.get('opp_action', [])
        c = Counter()
        for e in opp:
            c[e.get('source', '?')] += 1
        return c, len(opp)

    src_w, tot_opp_w = opp_source_counts(by_with)
    src_o, tot_opp_o = opp_source_counts(by_without)
    all_srcs = sorted(set(src_w) | set(src_o))
    lines = ['| Opp source | WITH % | WITHOUT % |']
    lines.append('|---|---:|---:|')
    for s in all_srcs:
        pw = 100 * src_w.get(s, 0) / max(tot_opp_w, 1)
        po = 100 * src_o.get(s, 0) / max(tot_opp_o, 1)
        lines.append(f'| {s} | {pw:.1f}% | {po:.1f}% |')
    render_section('Opponent action source distribution',
                   '\n'.join(lines),
                   f'\nTotal sampled opp actions: WITH={tot_opp_w}, WITHOUT={tot_opp_o}')

    # ── Phase / pool usage ─────────────────────────────────────
    def phase_pool_counts(events_dict):
        ph = events_dict.get('phase_opp', [])
        c = Counter()
        for e in ph:
            key = (e.get('phase', '?'), e.get('source', '?'))
            c[key] += 1
        return c, len(ph)

    pc_w, tot_ph_w = phase_pool_counts(by_with)
    pc_o, tot_ph_o = phase_pool_counts(by_without)
    all_keys = sorted(set(pc_w) | set(pc_o))
    lines = ['| Phase | Source | WITH % | WITHOUT % |']
    lines.append('|---|---|---:|---:|')
    for k in all_keys:
        pw = 100 * pc_w.get(k, 0) / max(tot_ph_w, 1)
        po = 100 * pc_o.get(k, 0) / max(tot_ph_o, 1)
        lines.append(f'| {k[0]} | {k[1]} | {pw:.1f}% | {po:.1f}% |')
    render_section('Phase + opp source distribution',
                   '\n'.join(lines))

    # ── Loss curves ─────────────────────────────────────────────
    def loss_progression(events_dict):
        losses_per_iter = events_dict.get('iter_losses', [])
        adv_means = []
        strat_means = []
        for e in losses_per_iter:
            for L in e.get('adv_losses', []):
                if isinstance(L, list):
                    adv_means.append(float(np.mean(L)))
                else:
                    adv_means.append(float(L))
            for L in e.get('strat_losses', []):
                if isinstance(L, list):
                    strat_means.append(float(np.mean(L)))
                else:
                    strat_means.append(float(L))
        return adv_means, strat_means

    adv_w, strat_w = loss_progression(by_with)
    adv_o, strat_o = loss_progression(by_without)

    def stats_line(label, lst):
        if not lst:
            return f'{label}: (no data)'
        a = np.array(lst)
        return f'{label}: n={len(a)}  mean={a.mean():.4f}  last={a[-1]:.4f}'

    render_section('Loss curves',
                   stats_line('WITH adv',     adv_w),
                   stats_line('WITHOUT adv',  adv_o),
                   stats_line('WITH strat',   strat_w),
                   stats_line('WITHOUT strat', strat_o))

    # ── Reward distribution ───────────────────────────────────
    def reward_dist(events_dict):
        shaped = events_dict.get('branch_shapers', [])
        rewards = []
        for e in shaped:
            for a, r in (e.get('rewards') or {}).items():
                if isinstance(r, (int, float)) and not isinstance(r, bool):
                    rewards.append(float(r))
        if not rewards:
            return None
        arr = np.array(rewards)
        return {
            'n': len(arr),
            'mean': round(float(arr.mean()), 3),
            'std': round(float(arr.std()), 3),
            'min': round(float(arr.min()), 3),
            'max': round(float(arr.max()), 3),
        }
    render_section('Branch reward distribution',
                   f'WITH:    {reward_dist(by_with)}',
                   f'WITHOUT: {reward_dist(by_without)}')

    # ── End summary ────────────────────────────────────
    render_section('Anomaly check',
        '  - obs[53]: should be populated WITH, all-null WITHOUT (sanity)',
        '  - Shaper fire rates: should match between runs (proves shapers use obs[18] only)',
        '  - Opp source mix: should match (phase logic is feature-independent)',
        '  - Pool usage: should match if both have same kbest_pool_size',
        '  - Loss curves: feature-ON has +1 input dim, slight loss difference expected',
        '  - Anything else that diverges significantly is a hidden coupling between',
        '    the feature flag and another code path.')


if __name__ == '__main__':
    main()
