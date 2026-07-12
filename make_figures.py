"""
Generate paper-quality figures from the held-out test evaluation.

Reads experiment_results/test_eval.json (produced by eval_paper.py) and writes
both PNG (300 dpi, for slides) and PDF (vector, for LaTeX) into figures/.

Figures
-------
  fig1_approx_ratio_by_k   : mean approximation ratio vs k, per method (log y)
  fig2_span_by_k           : mean span vs k, per method (log y)
  fig3_ablation_ladder     : random-p -> greedy -> model -> optimal (bar)
  fig4_overall_ratio       : overall mean approximation ratio, per method (bar)
  fig5_ratio_boxplot       : distribution of per-instance ratios (box)
  fig6_scatter_vs_optimal  : per-instance span vs ILP optimum (scatter, y=x)

Usage
-----
  python make_figures.py
  python make_figures.py --json experiment_results/test_eval.json --out figures
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe palette + consistent ordering/markers
STYLE = {
    'ILP':           ('#000000', 'o', 'ILP (optimal)'),
    'ML+DP(model)':  ('#0072B2', 'o', 'ML+DP (model)'),
    'ML+DP(hybrid)': ('#56B4E9', 's', 'ML+DP (hybrid)'),
    'ML+DP(greedy)': ('#E69F00', '^', 'ML+DP (greedy)'),
    'k-Inner':       ('#009E73', 'D', 'k-Inner'),
    'Baseline':      ('#D55E00', 'v', 'Baseline'),
    'k-Budget':      ('#CC79A7', 'P', 'k-Budget'),
}

# Random-p control (model-mode assignment, random probabilities), measured
# deterministically with torch.manual_seed(0) on the test set + best checkpoint.
# Reproduce: see EXPERIMENTS.md section 5.3.
RANDOM_P_SPAN = 3711.3


def load(json_path: str):
    d = json.load(open(json_path))
    return d['methods'], d['records']


def _save(fig, out_dir: str, name: str) -> None:
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {name}.png / {name}.pdf')


def per_instance_ratios(records: List[Dict], methods: List[str]) -> Dict[str, List[float]]:
    """ratio[m] = span_m / ILP_optimum, over instances where ILP is proven optimal."""
    out = {m: [] for m in methods if m != 'ILP'}
    for r in records:
        ilp = r['methods'].get('ILP', {})
        if ilp.get('optimal') is not True or not ilp.get('span'):
            continue
        opt = ilp['span']
        for m in out:
            s = r['methods'][m]['span']
            if s is not None:
                out[m].append(s / opt)
    return out


def by_k(records: List[Dict], methods: List[str]):
    """Return sorted k list, mean span[k][m], mean ratio[k][m]."""
    span = defaultdict(lambda: defaultdict(list))
    ratio = defaultdict(lambda: defaultdict(list))
    for r in records:
        k = r['k']
        ilp = r['methods'].get('ILP', {})
        opt = ilp['span'] if ilp.get('optimal') is True else None
        for m in methods:
            s = r['methods'][m]['span']
            if s is not None:
                span[k][m].append(s)
                if opt:
                    ratio[k][m].append(s / opt)
    ks = sorted(span)
    mean_span = {k: {m: np.mean(span[k][m]) if span[k][m] else np.nan for m in methods} for k in ks}
    mean_ratio = {k: {m: np.mean(ratio[k][m]) if ratio[k][m] else np.nan for m in methods} for k in ks}
    return ks, mean_span, mean_ratio


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_ratio_by_k(ks, mean_ratio, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ['ML+DP(model)', 'ML+DP(hybrid)', 'ML+DP(greedy)', 'k-Inner', 'Baseline', 'k-Budget']
    for m in order:
        c, mk, lbl = STYLE[m]
        ys = [mean_ratio[k][m] for k in ks]
        ax.plot(ks, ys, marker=mk, color=c, label=lbl, linewidth=2, markersize=6)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Optimal')
    ax.set_yscale('log')
    ax.set_xlabel('Interval budget $k$')
    ax.set_ylabel('Approximation ratio vs optimal  (log scale)')
    ax.set_title('Approximation ratio by $k$ (lower is better)')
    ax.set_xticks(ks)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out_dir, 'fig1_approx_ratio_by_k')


def fig_span_by_k(ks, mean_span, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ['ILP', 'ML+DP(model)', 'ML+DP(hybrid)', 'ML+DP(greedy)', 'k-Inner', 'Baseline', 'k-Budget']
    for m in order:
        c, mk, lbl = STYLE[m]
        ys = [mean_span[k][m] for k in ks]
        ls = '--' if m == 'ILP' else '-'
        ax.plot(ks, ys, marker=mk, color=c, label=lbl, linewidth=2, markersize=6, linestyle=ls)
    ax.set_yscale('log')
    ax.set_xlabel('Interval budget $k$')
    ax.set_ylabel('Average total span  (log scale)')
    ax.set_title('Average span by $k$ (lower is better)')
    ax.set_xticks(ks)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out_dir, 'fig2_span_by_k')


def fig_ablation_ladder(records, out_dir, random_p_span):
    # overall means from the JSON + the random-p control constant
    def overall(m):
        xs = [r['methods'][m]['span'] for r in records if r['methods'][m]['span'] is not None]
        return np.mean(xs)
    opt = overall('ILP')
    labels = ['Random $p$\n(model-mode)', 'Greedy\nspan-cost', 'Learned $p$\n(model-mode)', 'ILP\n(optimal)']
    vals   = [random_p_span, overall('ML+DP(greedy)'), overall('ML+DP(model)'), opt]
    colors = ['#999999', STYLE['ML+DP(greedy)'][0], STYLE['ML+DP(model)'][0], '#000000']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 40, f'{v:.0f}\n({v/opt:.2f}×)',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Average total span')
    ax.set_title('Ablation: only the learned signal helps\n(same network, same DP — only the assignment differs)')
    ax.set_ylim(0, RANDOM_P_SPAN * 1.18)
    ax.grid(True, axis='y', alpha=0.3)
    _save(fig, out_dir, 'fig3_ablation_ladder')


def fig_overall_ratio(ratios, out_dir):
    order = ['ML+DP(model)', 'ML+DP(hybrid)', 'k-Inner', 'ML+DP(greedy)', 'Baseline', 'k-Budget']
    means = [np.mean(ratios[m]) for m in order]
    colors = [STYLE[m][0] for m in order]
    labels = [STYLE[m][2] for m in order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(labels, means, color=colors, edgecolor='black', linewidth=0.6)
    ax.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    for b, v in zip(bars, means):
        ax.text(v + 0.1, b.get_y() + b.get_height() / 2, f'{v:.2f}×', va='center', fontsize=9)
    ax.set_xlabel('Mean approximation ratio vs optimal')
    ax.set_title('Overall approximation ratio (held-out test set, n=200)')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    _save(fig, out_dir, 'fig4_overall_ratio')


def fig_ratio_boxplot(ratios, out_dir):
    order = ['ML+DP(model)', 'ML+DP(hybrid)', 'ML+DP(greedy)', 'k-Inner']
    data = [ratios[m] for m in order]
    labels = [STYLE[m][2] for m in order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True,
                    medianprops=dict(color='black'))
    for patch, m in zip(bp['boxes'], order):
        patch.set_facecolor(STYLE[m][0]); patch.set_alpha(0.6)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Optimal')
    ax.set_ylabel('Per-instance approximation ratio')
    ax.set_title('Distribution of approximation ratios')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out_dir, 'fig5_ratio_boxplot')


def fig_scatter_vs_optimal(records, out_dir):
    opt, model, kinner = [], [], []
    for r in records:
        ilp = r['methods'].get('ILP', {})
        if ilp.get('optimal') is not True:
            continue
        opt.append(ilp['span'])
        model.append(r['methods']['ML+DP(model)']['span'])
        kinner.append(r['methods']['k-Inner']['span'])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(opt, kinner, color=STYLE['k-Inner'][0], marker='D', s=22, alpha=0.6, label='k-Inner')
    ax.scatter(opt, model, color=STYLE['ML+DP(model)'][0], marker='o', s=22, alpha=0.7, label='ML+DP (model)')
    lim = max(max(opt), max(kinner)) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.7, label='y = x (optimal)')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('ILP optimal span')
    ax.set_ylabel('Method span')
    ax.set_title('Per-instance span vs optimum\n(closer to the diagonal is better)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    _save(fig, out_dir, 'fig6_scatter_vs_optimal')


def main(args):
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'figure.dpi': 110})
    methods, records = load(args.json)
    print(f'Loaded {len(records)} records, methods: {methods}')

    ratios = per_instance_ratios(records, methods)
    ks, mean_span, mean_ratio = by_k(records, methods)

    print(f'Writing figures to {args.out}/ ...')
    if len(ks) >= 2:
        fig_ratio_by_k(ks, mean_ratio, args.out)
        fig_span_by_k(ks, mean_span, args.out)
    else:
        print(f'  (single k={ks[0]}: skipping by-k line plots)')
    fig_ablation_ladder(records, args.out, args.random_p_span)
    fig_overall_ratio(ratios, args.out)
    fig_ratio_boxplot(ratios, args.out)
    fig_scatter_vs_optimal(records, args.out)
    print('Done.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Generate paper figures from test_eval.json')
    ap.add_argument('--json', default='experiment_results/test_eval.json')
    ap.add_argument('--out', default='figures')
    ap.add_argument('--random-p-span', type=float, default=RANDOM_P_SPAN,
                    help='Mean span of the random-p control for the ablation-ladder figure '
                         '(measure per dataset; default is the k in [2,10] value)')
    main(ap.parse_args())
