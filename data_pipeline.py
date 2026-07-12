"""
Layer 1: Data pipeline for the DLMinTC+ k>1 extension.

Generates (graph, k, binary-mask) instances as .pt files.
Each saved instance contains:
  node_features : FloatTensor (n_nodes, T)  — per-(v,t) degree
  edges         : LongTensor  (E, 3)         — (u_idx, v_idx, t_idx)
  mask          : FloatTensor (n_nodes, T)   — binary ground-truth activity mask
  k             : int
  n_nodes       : int
  T             : int  (number of unique timestamps)
  gt_source     : str  ('ilp' or 'heuristic')
  seed          : int
"""

import os
import json
import random
import argparse
from math import floor
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import torch

import graph as utils
import baseline as baseline_mod
import k_inner as k_inner_mod
import k_budget as k_budget_mod
import ilp_version as ilp_mod


# ---------------------------------------------------------------------------
# Timestamp utilities
# ---------------------------------------------------------------------------

def floor_timestamps(timestamps: List[Tuple]) -> List[Tuple[int, int, int]]:
    return [(floor(t), int(u), int(v)) for (t, u, v) in timestamps]


def deduplicate(timestamps: List[Tuple]) -> List[Tuple]:
    return sorted(set(timestamps))


# ---------------------------------------------------------------------------
# Node / time index helpers
# ---------------------------------------------------------------------------

def build_indices(timestamps_int: List[Tuple]) -> Tuple[List[int], Dict, List[int], Dict]:
    """
    Returns (nodes, node_to_idx, unique_times, t_to_idx).
    Nodes are the sorted active node IDs; unique_times are sorted integer timestamps.
    """
    node_set = set()
    time_set = set()
    for t, u, v in timestamps_int:
        node_set.update([u, v])
        time_set.add(t)
    nodes = sorted(node_set)
    unique_times = sorted(time_set)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    t_to_idx = {t: i for i, t in enumerate(unique_times)}
    return nodes, node_to_idx, unique_times, t_to_idx


def reindex(timestamps_int: List[Tuple], G: nx.Graph, node_to_idx: Dict):
    """
    Relabel node IDs in timestamps and G to consecutive 0-based integers.
    Required so the ILP's V = range(G.number_of_nodes()) matches timestamp nodes.
    """
    ts_reindexed = [(t, node_to_idx[u], node_to_idx[v]) for t, u, v in timestamps_int]
    active_nodes = sorted(node_to_idx.keys())
    G_sub = G.subgraph(active_nodes).copy()
    G_reindexed = nx.relabel_nodes(G_sub, node_to_idx)
    return ts_reindexed, G_reindexed


# ---------------------------------------------------------------------------
# Feature / mask construction
# ---------------------------------------------------------------------------

def compute_node_features(
    timestamps_int: List[Tuple],
    node_to_idx: Dict,
    t_to_idx: Dict,
) -> np.ndarray:
    """Per-(v,t) degree. Shape: (n_nodes, T)."""
    features = np.zeros((len(node_to_idx), len(t_to_idx)), dtype=np.float32)
    for t, u, v in timestamps_int:
        ti = t_to_idx[t]
        features[node_to_idx[u], ti] += 1.0
        features[node_to_idx[v], ti] += 1.0
    return features


def build_edge_tensor(
    timestamps_int: List[Tuple],
    node_to_idx: Dict,
    t_to_idx: Dict,
) -> torch.Tensor:
    """Edge list (u_idx, v_idx, t_idx). Shape: (E, 3)."""
    rows = [
        [node_to_idx[u], node_to_idx[v], t_to_idx[t]]
        for t, u, v in timestamps_int
    ]
    if not rows:
        return torch.zeros((0, 3), dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)


def active_intervals_to_mask(
    active_ints: Dict,
    node_to_idx: Dict,
    unique_times: List[int],
) -> np.ndarray:
    """
    Convert {node: [(s, e), ...]} to binary mask of shape (n_nodes, T).
    Uses numpy broadcasting for speed.
    """
    times_arr = np.array(unique_times, dtype=np.int64)
    mask = np.zeros((len(node_to_idx), len(unique_times)), dtype=np.float32)
    for node, intervals in active_ints.items():
        if node not in node_to_idx:
            continue
        vi = node_to_idx[node]
        for s, e in intervals:
            mask[vi] = np.maximum(
                mask[vi],
                ((times_arr >= int(s)) & (times_arr <= int(e))).astype(np.float32),
            )
    return mask


def xstart_xend_to_active_intervals(Xstart: Dict, Xend: Dict) -> Dict:
    """Convert Xstart/Xend dict-of-dicts to {node: [(s, e), ...]}."""
    result = {}
    for node in Xstart:
        result[node] = []
        for idx in Xstart[node]:
            s, e = Xstart[node][idx], Xend[node][idx]
            if s != -np.inf and e != np.inf:
                result[node].append((float(s), float(e)))
    return result


# ---------------------------------------------------------------------------
# Ground-truth computation
# ---------------------------------------------------------------------------

def compute_ground_truth(
    timestamps_reindexed: List[Tuple],
    G_reindexed: nx.Graph,
    k: int,
    ilp_time_budget: float = 300.0,
    ilp_node_limit: int = 100,
    ilp_edge_limit: int = 1000,
) -> Tuple[Dict, str]:
    """
    For small instances (nodes ≤ ilp_node_limit AND edges ≤ ilp_edge_limit):
      try ILP first, fall back to best-of-heuristics.
    For large instances:
      skip ILP entirely, use k-Inner directly (fastest good heuristic),
      falling back to k-Budget then Baseline if k-Inner fails.
    Returns (active_intervals_dict, source) where source ∈ {'ilp', 'heuristic', 'none'}.
    """
    n_nodes = G_reindexed.number_of_nodes()
    n_edges = len(timestamps_reindexed)
    instance_is_small = (n_nodes <= ilp_node_limit) and (n_edges <= ilp_edge_limit)

    # --- ILP attempt (small instances only) ---
    if instance_is_small:
        try:
            m, x = ilp_mod.ilp(timestamps_reindexed, G_reindexed, k)
            m.setParam('TimeLimit', ilp_time_budget)
            m.setParam('OutputFlag', 0)
            m.setParam('LogFile', '')
            m.optimize()
            if m.SolCount > 0:
                return ilp_mod.active_intervals(m, x), 'ilp'
        except Exception:
            pass

    # --- Heuristic path ---
    # For large instances k-Inner is tried first (user preference).
    # For small instances that failed ILP, we try all three and pick cheapest.
    if instance_is_small:
        solvers = [
            lambda ts, k: k_inner_mod.runKInner(ts, k),
            lambda ts, k: k_budget_mod.runKBudget(ts, k),
            lambda ts, k: baseline_mod.kbaseline(ts, k),
        ]
    else:
        solvers = [
            lambda ts, k: k_inner_mod.runKInner(ts, k),
            lambda ts, k: k_budget_mod.runKBudget(ts, k),
            lambda ts, k: baseline_mod.kbaseline(ts, k),
        ]

    best_cost = float('inf')
    best_active_ints = None
    for fn in solvers:
        try:
            Xs, Xe = fn(timestamps_reindexed, k)
            cost = utils.getCost(Xs, Xe)
            if cost < best_cost:
                best_cost = cost
                best_active_ints = xstart_xend_to_active_intervals(Xs, Xe)
            # For large instances, stop after the first successful solver
            if not instance_is_small and best_active_ints is not None:
                break
        except Exception:
            pass

    if best_active_ints is None:
        return {}, 'none'
    return best_active_ints, 'heuristic'


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def generate_instance(
    n_nodes: int,
    k: int,
    event_length: int = 10,
    overlap: float = 0.5,
    seed: Optional[int] = None,
    ilp_time_budget: float = 300.0,
    ilp_node_limit: int = 50,
    ilp_edge_limit: int = 500,
) -> Optional[Dict]:
    """
    Generate one labeled instance. Returns None if generation fails.
    """
    if seed is None:
        seed = random.randint(0, 10 ** 6)

    try:
        G = utils.generateGraph(n=n_nodes)
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return None

        raw_ts, _ = utils.generateIntervals(
            G,
            event_length=event_length,
            overlap=overlap,
            seed=float(seed),
            number_intervals=k,
        )
        if not raw_ts:
            return None

        timestamps_int = deduplicate(floor_timestamps(raw_ts))
        if len(timestamps_int) < 2:
            return None

        nodes, node_to_idx, unique_times, t_to_idx = build_indices(timestamps_int)
        if not nodes or not unique_times:
            return None

        # Reindex to 0-based so ILP's V = range(n) matches timestamp node labels
        ts_reindexed, G_reindexed = reindex(timestamps_int, G, node_to_idx)

        node_features = compute_node_features(ts_reindexed, node_to_idx, t_to_idx)
        edge_tensor = build_edge_tensor(ts_reindexed, node_to_idx, t_to_idx)

        # After reindexing, node_to_idx maps old IDs → 0-based, so the reindexed
        # node IDs ARE 0-based. Rebuild a trivial identity map for the reindexed domain.
        n = len(nodes)
        T = len(unique_times)
        reindexed_node_to_idx = {i: i for i in range(n)}

        active_ints, gt_source = compute_ground_truth(
            ts_reindexed, G_reindexed, k, ilp_time_budget,
            ilp_node_limit, ilp_edge_limit,
        )
        if not active_ints:
            return None

        mask = active_intervals_to_mask(active_ints, reindexed_node_to_idx, unique_times)

        return {
            'node_features': torch.tensor(node_features),  # (n_nodes, T)
            'edges': edge_tensor,                           # (E, 3)
            'mask': torch.tensor(mask),                    # (n_nodes, T)
            'k': k,
            'n_nodes': n,
            'T': T,
            'unique_times': unique_times,                  # List[int], length T
            'gt_source': gt_source,
            'seed': seed,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n_instances: int,
    output_dir: str,
    k_range: Tuple[int, int] = (2, 10),
    n_nodes_choices: Tuple[int, ...] = (20, 30, 50),
    event_length_choices: Tuple[int, ...] = (5, 10),
    overlap_choices: Tuple[float, ...] = (0.0, 0.3, 0.5),
    ilp_time_budget: float = 300.0,
    ilp_node_limit: int = 50,
    ilp_edge_limit: int = 500,
    max_T: int = 500,
    seed: int = 42,
) -> None:
    """
    Generate n_instances and save each as a .pt file in output_dir.
    Randomises graph size, k, event_length, overlap per instance.

    max_T caps the number of unique timestamps per instance.  The Transformer's
    self-attention is O(n_nodes × T²), so large T exhausts GPU/MPS memory quickly.
    Instances exceeding max_T are silently rejected and regenerated.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    gt_sources: Dict[str, int] = {'ilp': 0, 'heuristic': 0, 'none': 0}
    saved = 0
    attempted = 0

    while saved < n_instances:
        attempted += 1
        k = rng.randint(*k_range)
        n_nodes = rng.choice(n_nodes_choices)
        event_length = rng.choice(event_length_choices)
        overlap = rng.choice(overlap_choices)
        inst_seed = rng.randint(0, 10 ** 6)

        instance = generate_instance(
            n_nodes=n_nodes,
            k=k,
            event_length=event_length,
            overlap=overlap,
            seed=inst_seed,
            ilp_time_budget=ilp_time_budget,
            ilp_node_limit=ilp_node_limit,
            ilp_edge_limit=ilp_edge_limit,
        )

        if instance is None:
            continue

        if instance['T'] > max_T:
            continue  # reject: too many timestamps → OOM in Transformer

        gt_sources[instance['gt_source']] = gt_sources.get(instance['gt_source'], 0) + 1
        torch.save(instance, os.path.join(output_dir, f'instance_{saved:05d}.pt'))
        saved += 1

        if saved % 100 == 0 or saved == n_instances:
            print(
                f'[{saved}/{n_instances}] attempts={attempted}  '
                f'ilp={gt_sources["ilp"]}  heuristic={gt_sources["heuristic"]}'
            )

    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump({
            'n_instances': n_instances,
            'k_range': list(k_range),
            'n_nodes_choices': list(n_nodes_choices),
            'event_length_choices': list(event_length_choices),
            'overlap_choices': list(overlap_choices),
            'max_T': max_T,
            'ilp_node_limit': ilp_node_limit,
            'ilp_edge_limit': ilp_edge_limit,
            'gt_sources': gt_sources,
        }, f, indent=2)

    print(f'\nDataset complete: {saved} instances saved to {output_dir}')
    print(f'Ground-truth sources: {gt_sources}')


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

class TemporalCoverDataset(torch.utils.data.Dataset):
    """Load pre-generated .pt instances from a directory."""

    def __init__(self, data_dir: str):
        self.files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.pt')
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        return torch.load(self.files[idx], weights_only=False)


def verify_mask_coverage(instance: Dict) -> bool:
    """
    Sanity check: every temporal edge (u, v, t) must have mask[u, t]=1 or mask[v, t]=1.
    """
    mask = instance['mask']      # (n_nodes, T)
    edges = instance['edges']    # (E, 3): u_idx, v_idx, t_idx
    for u, v, t in edges.tolist():
        if mask[u, t] < 0.5 and mask[v, t] < 0.5:
            return False
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training data for DLMinTC+ k>1')
    parser.add_argument('--n', type=int, default=200,
                        help='Number of training instances')
    parser.add_argument('--out', type=str, default='data/train',
                        help='Training output directory')
    parser.add_argument('--val-n', type=int, default=0,
                        help='Number of validation instances (0 = skip)')
    parser.add_argument('--val-out', type=str, default='data/val',
                        help='Validation output directory')
    parser.add_argument('--ilp-budget', type=float, default=60.0,
                        help='Per-instance ILP time budget in seconds')
    parser.add_argument('--ilp-node-limit', type=int, default=50,
                        help='Skip ILP for instances with more nodes than this')
    parser.add_argument('--ilp-edge-limit', type=int, default=500,
                        help='Skip ILP for instances with more temporal edges than this')
    parser.add_argument('--max-t', type=int, default=500,
                        help='Reject instances with more than this many unique timestamps (OOM guard)')
    parser.add_argument('--n-nodes', type=str, default='20,30,50',
                        help='Comma-separated list of graph sizes to sample from, e.g. 100,200,500')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verify', action='store_true',
                        help='Verify coverage on first 10 instances of each split')
    args = parser.parse_args()

    n_nodes_choices = tuple(int(x) for x in args.n_nodes.split(','))

    print(f'=== Training split ({args.n} instances, max_T={args.max_t}) ===')
    print(f'Node sizes: {n_nodes_choices}')
    print(f'ILP limits: nodes≤{args.ilp_node_limit}, edges≤{args.ilp_edge_limit}; larger instances use k-Inner')
    generate_dataset(
        n_instances=args.n,
        output_dir=args.out,
        n_nodes_choices=n_nodes_choices,
        ilp_time_budget=args.ilp_budget,
        ilp_node_limit=args.ilp_node_limit,
        ilp_edge_limit=args.ilp_edge_limit,
        max_T=args.max_t,
        seed=args.seed,
    )

    if args.val_n > 0:
        print(f'\n=== Validation split ({args.val_n} instances, max_T={args.max_t}) ===')
        generate_dataset(
            n_instances=args.val_n,
            output_dir=args.val_out,
            n_nodes_choices=n_nodes_choices,
            ilp_time_budget=args.ilp_budget,
            ilp_node_limit=args.ilp_node_limit,
            ilp_edge_limit=args.ilp_edge_limit,
            max_T=args.max_t,
            seed=args.seed + 1,
        )

    if args.verify:
        for label, directory in [('train', args.out), ('val', args.val_out)]:
            if not os.path.isdir(directory):
                continue
            print(f'\nVerifying coverage on first 10 {label} instances...')
            ds = TemporalCoverDataset(directory)
            ok = 0
            for i in range(min(10, len(ds))):
                inst = ds[i]
                covered = verify_mask_coverage(inst)
                status = 'OK' if covered else 'FAIL'
                print(f'  instance_{i:05d}.pt  n={inst["n_nodes"]}  T={inst["T"]}  '
                      f'k={inst["k"]}  gt={inst["gt_source"]}  coverage={status}')
                if covered:
                    ok += 1
            print(f'{ok}/10 {label} instances passed.')
