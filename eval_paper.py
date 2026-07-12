"""
Comprehensive paper-grade evaluation on a held-out test set.

For every test instance this runs, in a single pass:
  - ILP (Gurobi) with a wall-clock budget, recording whether the returned
    solution was proven optimal or only feasible
  - ML+DP in all three edge-assignment modes (greedy / model / hybrid)
    from ONE model forward pass
  - k-Inner, k-Budget, Baseline heuristics

It records per-instance span and coverage for every method, aggregates by k and
overall, computes approximation ratios against the proven-optimal ILP subset,
and writes both a human-readable report and a machine-readable JSON for the
paper.

The expensive parts (ILP, heuristics) are computed exactly once per instance;
the three ML modes share the same forward pass, so adding modes is nearly free.

CLI
---
  python eval_paper.py --test data/test --checkpoint checkpoints/best_model.pt \
      --ilp-budget 30 --out experiment_results/test_eval.json
"""

import argparse
import json
import os
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

import baseline as baseline_mod
import ilp_version as ilp_mod
import k_budget as k_budget_mod
import k_inner as k_inner_mod
from data_pipeline import TemporalCoverDataset
from model import DLMinTCk
from postprocess import coverage_fraction, postprocess, sum_span

warnings.filterwarnings('ignore')

# ML+DP variants share one forward pass; (label -> (mode, beta))
ML_MODES = {
    'ML+DP(greedy)': ('greedy', 1.0),
    'ML+DP(model)':  ('model', 1.0),
    'ML+DP(hybrid)': ('hybrid', 1.0),
}
HEURISTICS = {
    'k-Inner':  lambda ts, k: k_inner_mod.runKInner(ts, k),
    'k-Budget': lambda ts, k: k_budget_mod.runKBudget(ts, k),
    'Baseline': lambda ts, k: baseline_mod.kbaseline(ts, k),
}


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def heuristic_intervals(Xstart: Dict, Xend: Dict) -> Dict[int, List[Tuple[int, int]]]:
    out: Dict[int, List[Tuple[int, int]]] = {}
    for node in Xstart:
        out[node] = []
        for idx in Xstart[node]:
            s, e = Xstart[node][idx], Xend[node][idx]
            if not (np.isinf(s) or np.isinf(e)):
                out[node].append((int(s), int(e)))
    return out


def run_ilp(
    timestamps: List[Tuple[int, int, int]],
    n_nodes: int,
    k: int,
    budget: float,
) -> Tuple[Optional[Dict[int, List[Tuple[int, int]]]], bool]:
    """
    Returns (intervals, is_optimal).  intervals is None if no feasible solution
    was found within the budget; is_optimal is True only if Gurobi proved
    optimality (status GRB.OPTIMAL).
    """
    from gurobipy import GRB
    try:
        G = nx.path_graph(n_nodes)  # topology irrelevant; V = range(n_nodes)
        m, x = ilp_mod.ilp(timestamps, G, k)
        m.setParam('TimeLimit', budget)
        m.setParam('OutputFlag', 0)
        m.setParam('LogFile', '')
        m.optimize()
        if m.SolCount == 0:
            return None, False
        return ilp_mod.active_intervals(m, x), (m.status == GRB.OPTIMAL)
    except Exception as exc:
        warnings.warn(f'ILP failed: {exc}')
        return None, False


# ---------------------------------------------------------------------------
# Per-instance evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_instance(
    inst: Dict,
    model: DLMinTCk,
    device: torch.device,
    ilp_budget: float,
) -> Dict[str, Dict]:
    """{method: {'span': int|None, 'coverage': float|None, 'optimal': bool|None}}."""
    nf           = inst['node_features'].to(device)
    edges        = inst['edges'].to(device)
    k            = int(inst['k'])
    n_nodes      = int(inst['n_nodes'])
    unique_times = inst['unique_times']
    timestamps   = [(unique_times[ti], u, v) for u, v, ti in edges.cpu().tolist()]

    res: Dict[str, Dict] = {}

    # --- ML+DP (one forward pass, three assignment modes) ---
    p = model(nf, edges, k, n_nodes, inst['T']).cpu()
    edges_cpu = edges.cpu()
    for label, (mode, beta) in ML_MODES.items():
        iv = postprocess(p, edges_cpu, k, unique_times, mode=mode, beta=beta)
        res[label] = {
            'span':     sum_span(iv),
            'coverage': coverage_fraction(iv, edges_cpu, unique_times),
            'optimal':  None,
        }

    # --- Heuristics ---
    for name, fn in HEURISTICS.items():
        try:
            Xs, Xe = fn(timestamps, k)
            iv = heuristic_intervals(Xs, Xe)
            res[name] = {
                'span':     sum_span(iv),
                'coverage': coverage_fraction(iv, edges_cpu, unique_times),
                'optimal':  None,
            }
        except Exception as exc:
            warnings.warn(f'{name} failed: {exc}')
            res[name] = {'span': None, 'coverage': None, 'optimal': None}

    # --- ILP ---
    if ilp_budget > 0:
        iv, is_opt = run_ilp(timestamps, n_nodes, k, ilp_budget)
        if iv is None:
            res['ILP'] = {'span': None, 'coverage': None, 'optimal': False}
        else:
            res['ILP'] = {
                'span':     sum_span(iv),
                'coverage': coverage_fraction(iv, edges_cpu, unique_times),
                'optimal':  is_opt,
            }

    return res


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------

def aggregate(records: List[Dict], methods: List[str]) -> Dict:
    """Group per-instance records by k and overall; compute means."""
    by_k: Dict[int, Dict[str, Dict[str, List]]] = defaultdict(
        lambda: {m: {'span': [], 'coverage': []} for m in methods}
    )
    for rec in records:
        k = rec['k']
        for m in methods:
            by_k[k][m]['span'].append(rec['methods'][m]['span'])
            by_k[k][m]['coverage'].append(rec['methods'][m]['coverage'])
    return by_k


def _mean(xs: List) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float('nan')


def report(records: List[Dict], methods: List[str], out_path: Optional[str]) -> None:
    by_k = aggregate(records, methods)

    # ---- per-k table ----
    print('\n=== Average span by k (lower is better) ===')
    col = 16
    header = f"{'k':>3} {'n':>4}  " + "".join(f"{m:>{col}}" for m in methods)
    print(header); print('-' * len(header))
    for k in sorted(by_k):
        d = by_k[k]
        n = len(d[methods[0]]['span'])
        row = f"{k:>3} {n:>4}  " + "".join(f"{_mean(d[m]['span']):>{col}.1f}" for m in methods)
        print(row)

    # ---- coverage table ----
    print('\n=== Average coverage by k (1.0 = valid cover) ===')
    print(header); print('-' * len(header))
    for k in sorted(by_k):
        d = by_k[k]
        n = len(d[methods[0]]['span'])
        row = f"{k:>3} {n:>4}  " + "".join(f"{_mean(d[m]['coverage']):>{col}.4f}" for m in methods)
        print(row)

    # ---- overall ----
    print('\n=== Overall (all k) ===')
    overall = {m: {'span': [], 'coverage': []} for m in methods}
    for k in by_k:
        for m in methods:
            overall[m]['span'].extend(by_k[k][m]['span'])
            overall[m]['coverage'].extend(by_k[k][m]['coverage'])
    for m in methods:
        print(f'  {m:<16} avg_span={_mean(overall[m]["span"]):>10.1f}  '
              f'avg_coverage={_mean(overall[m]["coverage"]):.4f}')

    # ---- approximation ratios on the proven-optimal ILP subset ----
    opt_idx = [
        i for i, rec in enumerate(records)
        if rec['methods'].get('ILP', {}).get('optimal') is True
    ]
    if opt_idx:
        print(f'\n=== Approximation ratio vs proven-optimal ILP '
              f'({len(opt_idx)} instances) ===')
        for m in methods:
            if m == 'ILP':
                continue
            ratios = []
            for i in opt_idx:
                opt = records[i]['methods']['ILP']['span']
                val = records[i]['methods'][m]['span']
                if opt is not None and val is not None and opt > 0:
                    ratios.append(val / opt)
            if ratios:
                print(f'  {m:<16} mean_ratio={np.mean(ratios):.3f}×  '
                      f'median={np.median(ratios):.3f}×')
    else:
        print('\n(no proven-optimal ILP instances — skipping ratio table)')

    # ---- machine-readable dump ----
    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({'methods': methods, 'records': records}, f, indent=2)
        print(f'\nRaw results written to {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    if args.device == 'auto':
        device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
    else:
        device = torch.device(args.device)
    print(f'Device: {device}')

    ds = TemporalCoverDataset(args.test)
    print(f'Test instances: {len(ds)}')

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DLMinTCk(
        k_emb_dim=args.k_emb_dim, sage_hidden=args.sage_hidden,
        sage_layers=args.sage_layers, tf_hidden=args.tf_hidden,
        tf_heads=args.tf_heads, tf_layers=args.tf_layers,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Loaded checkpoint from epoch {ckpt.get("epoch", "?")} '
          f'(val_span={ckpt.get("val_metrics", {}).get("pp_span", "?")})')

    methods = ['ILP'] + list(ML_MODES) + list(HEURISTICS)
    if args.ilp_budget == 0:
        methods = [m for m in methods if m != 'ILP']
        print('ILP skipped (--ilp-budget 0)')
    else:
        print(f'ILP budget: {args.ilp_budget}s/instance')

    records: List[Dict] = []
    t0 = time.time()
    for i, inst in enumerate(ds):
        if 'unique_times' not in inst:
            continue
        res = eval_instance(inst, model, device, args.ilp_budget)
        records.append({
            'k': int(inst['k']),
            'n_nodes': int(inst['n_nodes']),
            'T': int(inst['T']),
            'methods': res,
        })
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(ds)} done  ({time.time() - t0:.0f}s)')

    report(records, methods, args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paper-grade test-set evaluation')
    parser.add_argument('--test',       type=str, default='data/test')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--ilp-budget', type=float, default=30.0,
                        help='Per-instance ILP time budget in seconds (0 = skip ILP)')
    parser.add_argument('--out',        type=str, default='experiment_results/test_eval.json')
    parser.add_argument('--device',     type=str, default='auto')
    # Arch must match the checkpoint
    parser.add_argument('--k-emb-dim',   type=int, default=16)
    parser.add_argument('--sage-hidden', type=int, default=64)
    parser.add_argument('--sage-layers', type=int, default=2)
    parser.add_argument('--tf-hidden',   type=int, default=64)
    parser.add_argument('--tf-heads',    type=int, default=4)
    parser.add_argument('--tf-layers',   type=int, default=2)
    args = parser.parse_args()
    main(args)
