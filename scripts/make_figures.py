#!/usr/bin/env python3
"""Render the README figures (light + dark variants) from docs/results/.

  docs/figures/drift-{light,dark}.png      CBet by training iteration
  docs/figures/h2h-{light,dark}.png        head-to-head matrix (tournament v34)
  docs/figures/baselines-{light,dark}.png  BB/100 vs fixed opponents, 95% CI

Usage:
  python3 scripts/make_figures.py
"""
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'docs' / 'results'
OUT = ROOT / 'docs' / 'figures'

THEMES = {
    'light': dict(surface='#fcfcfb', text='#0b0b0b', muted='#52514e',
                  grid='#e4e3df', series='#2a78d6', neutral='#a3a29c',
                  band='#eef4fc', neg='#e34948', mid='#f0efec'),
    'dark': dict(surface='#1a1a19', text='#f0efec', muted='#c3c2b7',
                 grid='#33332f', series='#3987e5', neutral='#7a7973',
                 band='#1d2a3a', neg='#e66767', mid='#383835'),
}

CBET_BAND = (55, 70)


def _style(ax, t):
    ax.set_facecolor(t['surface'])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(t['grid'])
    ax.tick_params(colors=t['muted'], length=0, labelsize=10)
    ax.grid(axis='y', color=t['grid'], linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(t, size):
    fig, ax = plt.subplots(figsize=size, dpi=160)
    fig.patch.set_facecolor(t['surface'])
    _style(ax, t)
    return fig, ax


def _title(fig, t, title, subtitle):
    fig.text(0.012, 0.965, title, color=t['text'], fontsize=13,
             fontweight='bold', va='top')
    fig.text(0.012, 0.905, subtitle, color=t['muted'], fontsize=10, va='top')


def drift(mode: str) -> None:
    t = THEMES[mode]
    rows = list(csv.DictReader(open(RESULTS / 'training_drift.csv')))
    main = sorted((int(r['iteration']), float(r['cbet']))
                  for r in rows if r['run'] == 'main')
    sib = [(int(r['iteration']), float(r['cbet']))
           for r in rows if r['run'] == 'sibling']

    fig, ax = _figure(t, (8, 4.2))
    ax.axhspan(*CBET_BAND, color=t['band'], zorder=0)
    ax.text(2560, CBET_BAND[1] - 1.2, 'target band 55–70%', color=t['muted'],
            fontsize=9, va='top')
    ax.scatter(*zip(*sib), s=42, facecolors='none', edgecolors=t['neutral'],
               linewidths=1.5, label='sibling runs', zorder=2)
    x, y = zip(*main)
    ax.plot(x, y, color=t['series'], linewidth=2, zorder=3)
    ax.scatter(x, y, s=42, color=t['series'], edgecolors=t['surface'],
               linewidths=2, label='main run', zorder=4)

    peak = max(sib + main, key=lambda p: p[1])
    ax.annotate(f'{peak[1]:.0f}%', peak, xytext=(8, 0), textcoords='offset points',
                color=t['text'], fontsize=9, va='center')
    ax.annotate(f'{main[0][1]:.0f}%', main[0], xytext=(0, -14),
                textcoords='offset points', color=t['text'], fontsize=9,
                ha='center')

    ax.set_ylim(35, 82)
    ax.set_xlim(2520, 4280)
    ax.set_yticks(range(40, 81, 10))
    ax.set_yticklabels([f'{v}%' for v in range(40, 81, 10)])
    ax.set_xlabel('training iteration', color=t['muted'], fontsize=10)
    leg = ax.legend(loc='lower right', frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t['text'])
    _title(fig, t, 'Continuation-bet frequency drifts up with training',
           'Self-play CBet % per checkpoint; selection keeps the last checkpoint inside the band')
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.14)
    fig.savefig(OUT / f'drift-{mode}.png', facecolor=t['surface'])
    plt.close(fig)


def _h2h_matrix():
    text = (ROOT / 'tournament' / 'tornament_v34' / 'summary.md').read_text()
    block = text.split('## H2H Matrix', 1)[1]
    lines = [l for l in block.splitlines() if l.startswith('|')]
    header = [c.strip() for c in lines[0].strip('|').split('|')][1:]
    rows = []
    for l in lines[2:]:
        cells = [c.strip() for c in l.strip('|').split('|')]
        rows.append([np.nan if c in ('—', '') else float(c) for c in cells[1:]])
    labels = [re.sub(r'checkpoint_iter_(\d+)(\((\d)\))?',
                     lambda m: m.group(1) + (f' ({m.group(3)})' if m.group(3) else ''), h)
              for h in header]
    return labels, np.array(rows)


def h2h(mode: str) -> None:
    t = THEMES[mode]
    labels, m = _h2h_matrix()
    lim = np.nanmax(np.abs(m))
    cmap = LinearSegmentedColormap.from_list('div', [t['neg'], t['mid'], t['series']])
    cmap.set_bad(t['surface'])

    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=160)
    fig.patch.set_facecolor(t['surface'])
    ax.set_facecolor(t['surface'])
    ax.imshow(np.ma.masked_invalid(m), cmap=cmap, vmin=-lim, vmax=lim)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f'{m[i, j]:+.1f}', ha='center', va='center',
                        fontsize=8.5, color=t['text'])
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.tick_params(colors=t['muted'], length=0, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(labels)), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels)), minor=True)
    ax.grid(which='minor', color=t['surface'], linewidth=2)
    ax.tick_params(which='minor', length=0)
    ax.set_xlabel('opponent (iteration)', color=t['muted'], fontsize=10)
    ax.set_ylabel('model (iteration)', color=t['muted'], fontsize=10)
    _title(fig, t, 'Head-to-head edge, BB/100 (row vs column)',
           'edge = row BB/100 minus column BB/100; 6-max self-play, 50k hands per pairing')
    fig.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.17)
    fig.savefig(OUT / f'h2h-{mode}.png', facecolor=t['surface'])
    plt.close(fig)


def baselines(mode: str) -> None:
    t = THEMES[mode]
    data = json.loads((RESULTS / 'baselines.json').read_text())
    rows = data['results'][::-1]
    labels = [r['label'] for r in rows]
    vals = np.array([r['bb100'] for r in rows])
    ci = np.array([r['ci95'] for r in rows])
    y = np.arange(len(rows))

    fig, ax = _figure(t, (8, 0.42 * len(rows) + 1.6))
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=t['grid'], linewidth=0.8)
    ax.barh(y, vals, height=0.55, color=t['series'], zorder=2)
    ax.errorbar(vals, y, xerr=ci, fmt='none', ecolor=t['text'],
                elinewidth=1, capsize=3, zorder=3)
    ax.axvline(0, color=t['muted'], linewidth=1, zorder=1)
    span = (vals + np.sign(vals) * ci)
    pad = (span.max() - min(span.min(), 0)) * 0.02
    for yi, v, c in zip(y, vals, ci):
        x = v + c + pad if v >= 0 else v - c - pad
        ax.text(x, yi, f'{v:+.1f} ± {c:.1f}', va='center',
                ha='left' if v >= 0 else 'right', fontsize=9, color=t['text'])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=t['text'])
    lo = min(0, (vals - ci).min())
    hi = (vals + ci).max()
    ax.set_xlim(lo - (hi - lo) * 0.2 if lo < 0 else 0, hi + (hi - lo) * 0.22)
    ax.set_xlabel('model win rate, BB/100 per seat (95% CI)', color=t['muted'],
                  fontsize=10)
    _title(fig, t, f"{data['model'].replace('.pt', '')} against fixed opponents",
           f"6-max, 100BB, {data['hands_per_match']:,} hands per opponent; "
           'model in seats 1/3/5, opponent in 2/4/6')
    fig.subplots_adjust(left=0.30, right=0.98, top=1 - 1.0 / fig.get_figheight(),
                        bottom=0.55 / fig.get_figheight())
    fig.savefig(OUT / f'baselines-{mode}.png', facecolor=t['surface'])
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in THEMES:
        drift(mode)
        h2h(mode)
        if (RESULTS / 'baselines.json').exists():
            baselines(mode)
    print('wrote', sorted(p.name for p in OUT.glob('*.png')))


if __name__ == '__main__':
    main()
