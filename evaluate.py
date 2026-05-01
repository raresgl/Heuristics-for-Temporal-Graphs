"""
Layer 5: Evaluation — compare ML+postprocess vs. baselines on held-out data.

For each instance in the validation set, four methods are evaluated:
  1. ML+DP   : DLMinTCk forward pass → postprocess DP
  2. k-Inner : runKInner  (k_inner.py)
  3. k-Budget: runKBudget (k_budget.py)
  4. Baseline: kbaseline  (baseline.py)

Metrics (per instance):
  sum_span   : total active time (lower is better)
  coverage   : fraction of edges covered (1.0 = valid cover)

Results are printed as a table grouped by k value.

CLI usage
---------
  python evaluate.py --val data/val --checkpoint checkpoints/best_model.pt
"""

import argparse
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import baseline as baseline_mod
import k_budget as k_budget_mod
import k_inner as k_inner_mod
from data_pipeline import TemporalCoverDataset
from model import DLMinTCk
from postprocess import coverage_fraction, postprocess, sum_span


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def heuristic_active_intervals(
    Xstart: Dict, Xend: Dict, unique_times: List[int]
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Convert Xstart/Xend (index-based) to {node: [(actual_start, actual_end), ...]}.
    Xstart/Xend use the integer timestamp values directly (they come from the
    heuristic solvers which work in the reindexed timestamp domain where the
    timestamps happen to be the same integer values).
    """
    result: Dict[int, List[Tuple[int, int]]] = {}
    for node in Xstart:
        result[node] = []
        for idx in Xstart[node]:
            s = Xstart[node][idx]
            e = Xend[node][idx]
            if s != -np.inf and e != np.inf:
                result[node].append((int(s), int(e)))
    return result


def edges_as_timestamps(
    edges: torch.Tensor, unique_times: List[int]
) -> List[Tuple[int, int, int]]:
    """Reconstruct [(t, u, v)] from edge tensor + unique_times index."""
    return [(unique_times[ti], u, v) for u, v, ti in edges.tolist()]


def run_heuristic(
    fn,
    timestamps_reindexed: List[Tuple],
    k: int,
    unique_times: List[int],
) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    try:
        Xs, Xe = fn(timestamps_reindexed, k)
        return heuristic_active_intervals(Xs, Xe, unique_times)
    except Exception as exc:
        warnings.warn(f"Heuristic failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Per-instance evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_instance(
    inst: Dict,
    model: DLMinTCk,
    device: torch.device,
) -> Optional[Dict[str, Dict]]:
    """
    Returns {method: {'span': int, 'coverage': float}} for one instance,
    or None if the instance is missing 'unique_times' (old format).
    """
    if 'unique_times' not in inst:
        return None

    nf          = inst['node_features'].to(device)
    edges       = inst['edges'].to(device)
    k           = int(inst['k'])
    unique_times: List[int] = inst['unique_times']
    timestamps  = edges_as_timestamps(edges.cpu(), unique_times)

    results: Dict[str, Dict] = {}

    # 1. ML + DP
    p = model(nf, edges, k, inst['n_nodes'], inst['T'])
    ml_intervals = postprocess(p.cpu(), edges.cpu(), k, unique_times)
    results['ML+DP'] = {
        'span':     sum_span(ml_intervals),
        'coverage': coverage_fraction(ml_intervals, edges.cpu(), unique_times),
    }

    # 2–4. Heuristics (they work on reindexed timestamps — same as what we stored)
    heuristics = [
        ('k-Inner',  lambda ts, k: k_inner_mod.runKInner(ts, k)),
        ('k-Budget', lambda ts, k: k_budget_mod.runKBudget(ts, k)),
        ('Baseline', lambda ts, k: baseline_mod.kbaseline(ts, k)),
    ]
    for name, fn in heuristics:
        intervals = run_heuristic(fn, timestamps, k, unique_times)
        if intervals is None:
            results[name] = {'span': None, 'coverage': None}
        else:
            results[name] = {
                'span':     sum_span(intervals),
                'coverage': coverage_fraction(intervals, edges.cpu(), unique_times),
            }

    return results


# ---------------------------------------------------------------------------
# Aggregation & printing
# ---------------------------------------------------------------------------

def print_table(
    by_k: Dict[int, Dict[str, List]],
    methods: List[str],
) -> None:
    col_w = 14
    header = f"{'k':>4}  {'n':>5}  " + "".join(
        f"{'span_' + m:>{col_w}}  {'cov_' + m:>{col_w}}" for m in methods
    )
    print(header)
    print("-" * len(header))

    for k in sorted(by_k.keys()):
        data = by_k[k]
        n = len(data[methods[0]]['span'])
        row = f"{k:>4}  {n:>5}  "
        for m in methods:
            spans = [s for s in data[m]['span'] if s is not None]
            covs  = [c for c in data[m]['coverage'] if c is not None]
            avg_span = sum(spans) / len(spans) if spans else float('nan')
            avg_cov  = sum(covs)  / len(covs)  if covs  else float('nan')
            row += f"{avg_span:>{col_w}.1f}  {avg_cov:>{col_w}.4f}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f'Device: {device}')

    # Dataset
    ds = TemporalCoverDataset(args.val)
    print(f'Validation instances: {len(ds)}')

    # Model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DLMinTCk(
        k_emb_dim=args.k_emb_dim,
        sage_hidden=args.sage_hidden,
        sage_layers=args.sage_layers,
        tf_hidden=args.tf_hidden,
        tf_heads=args.tf_heads,
        tf_layers=args.tf_layers,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Loaded checkpoint from epoch {ckpt.get("epoch", "?")}')

    methods = ['ML+DP', 'k-Inner', 'k-Budget', 'Baseline']
    by_k: Dict[int, Dict[str, Dict[str, List]]] = defaultdict(
        lambda: {m: {'span': [], 'coverage': []} for m in methods}
    )

    skipped = 0
    for i, inst in enumerate(ds):
        res = evaluate_instance(inst, model, device)
        if res is None:
            skipped += 1
            continue
        k = int(inst['k'])
        for m in methods:
            by_k[k][m]['span'].append(res[m]['span'])
            by_k[k][m]['coverage'].append(res[m]['coverage'])

        if (i + 1) % 20 == 0:
            print(f'  evaluated {i + 1}/{len(ds)} instances  (skipped={skipped})')

    if skipped:
        print(
            f'\nNote: {skipped} instances skipped (missing unique_times — '
            f'regenerate data with the current data_pipeline.py to fix this).'
        )

    print('\n=== Results by k ===')
    print_table(by_k, methods)

    # Overall summary
    print('\n=== Overall (all k) ===')
    overall: Dict[str, Dict[str, List]] = {m: {'span': [], 'coverage': []} for m in methods}
    for k_data in by_k.values():
        for m in methods:
            overall[m]['span'].extend(k_data[m]['span'])
            overall[m]['coverage'].extend(k_data[m]['coverage'])

    for m in methods:
        spans = [s for s in overall[m]['span'] if s is not None]
        covs  = [c for c in overall[m]['coverage'] if c is not None]
        avg_span = sum(spans) / len(spans) if spans else float('nan')
        avg_cov  = sum(covs) / len(covs)   if covs  else float('nan')
        print(f'  {m:<12}  avg_span={avg_span:.1f}  avg_coverage={avg_cov:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DLMinTCk vs baselines')
    parser.add_argument('--val',        type=str, default='data/val')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--device',     type=str, default='auto')
    # Model arch must match the checkpoint
    parser.add_argument('--k-emb-dim',   type=int, default=16)
    parser.add_argument('--sage-hidden', type=int, default=64)
    parser.add_argument('--sage-layers', type=int, default=2)
    parser.add_argument('--tf-hidden',   type=int, default=64)
    parser.add_argument('--tf-heads',    type=int, default=4)
    parser.add_argument('--tf-layers',   type=int, default=2)
    args = parser.parse_args()
    main(args)
