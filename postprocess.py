"""
Layer 4: Post-processing DP for DLMinTCk outputs.

Given the model's soft mask p (n_nodes, T) and the temporal edges, this module:
1. Assigns each edge to the endpoint with higher probability (or either if both
   are predicted active).
2. Runs an O(n^2 * k) interval-covering DP per node to find the k cheapest
   intervals that cover all timestamps the node is responsible for.
3. Returns {node_idx: [(start_time, end_time), ...]} using actual timestamp values.

API
---
  postprocess(p, edges, k, unique_times) -> Dict[int, List[Tuple[int,int]]]
  sum_span(active_intervals)             -> int
  coverage_fraction(active_intervals, edges, unique_times) -> float
"""

from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# DP core
# ---------------------------------------------------------------------------

def find_minimal_intervals(
    times: List[int],
    k: int,
) -> List[Tuple[int, int]]:
    """
    Given a sorted list of integer timestamps that must be covered and a budget
    of k intervals, return the k (or fewer) intervals of minimum total span.

    Cost of an interval [times[i], times[j]] = times[j] - times[i].
    A single timestamp costs 0 (degenerate interval).

    Returns [] if `times` is empty.
    Time complexity: O(n^2 * k) where n = len(times).
    """
    if not times:
        return []

    times = sorted(set(times))
    n = len(times)
    k = min(k, n)  # never need more intervals than timestamps

    INF = float('inf')
    # dp[j][i] = min cost to cover times[i:] using exactly j intervals
    # We reconstruct the split points afterwards.
    dp = [[INF] * (n + 1) for _ in range(k + 1)]
    split = [[n] * (n + 1) for _ in range(k + 1)]  # split[j][i] = best l for j intervals

    # Base: 0 intervals can cover nothing beyond position n
    for i in range(n + 1):
        dp[0][i] = 0 if i == n else INF

    for j in range(1, k + 1):
        for i in range(n - 1, -1, -1):
            # Place an interval starting at times[i], ending at times[l-1]
            for l in range(i + 1, n + 1):
                cost = times[l - 1] - times[i]
                remaining = dp[j - 1][l]
                if remaining == INF:
                    continue
                total = cost + remaining
                if total < dp[j][i]:
                    dp[j][i] = total
                    split[j][i] = l

    # Find minimum cost over all j intervals starting at 0
    best_j, best_cost = 1, dp[1][0]
    for j in range(2, k + 1):
        if dp[j][0] < best_cost:
            best_cost = dp[j][0]
            best_j = j

    # Reconstruct intervals
    intervals = []
    j, i = best_j, 0
    while j > 0 and i < n:
        l = split[j][i]
        intervals.append((times[i], times[l - 1]))
        i = l
        j -= 1

    return intervals


# ---------------------------------------------------------------------------
# Edge-to-node assignment
# ---------------------------------------------------------------------------

def _assign_edges(
    p: torch.Tensor,   # (n_nodes, T) float, 0..1
    edges: torch.Tensor,  # (E, 3): u_idx, v_idx, t_idx
) -> Dict[int, List[int]]:
    """
    For each edge (u, v, t), assign the coverage responsibility:
    - If only u is predicted active (p_u > 0.5), assign to u.
    - If only v is predicted active, assign to v.
    - If both or neither are active, assign to the one with higher p
      (ties go to u).
    Returns {node_idx: [t_idx, ...]} — the timestamp indices each node is
    responsible for.
    """
    n_nodes = p.size(0)
    resp: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}

    if edges.numel() == 0:
        return resp

    for u, v, t in edges.tolist():
        pu, pv = p[u, t].item(), p[v, t].item()
        if pu >= pv:
            resp[u].append(t)
        else:
            resp[v].append(t)

    return resp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def postprocess(
    p: torch.Tensor,        # (n_nodes, T) — model output
    edges: torch.Tensor,    # (E, 3): u_idx, v_idx, t_idx
    k: int,
    unique_times: List[int],  # length T; maps t_idx → actual timestamp
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Convert soft mask + edges into a valid k-interval cover.

    Returns {node_idx: [(start_time, end_time), ...]} using actual timestamps.
    Nodes with no assigned edges get an empty list (not active).
    """
    resp = _assign_edges(p, edges)

    result: Dict[int, List[Tuple[int, int]]] = {}
    for node, t_indices in resp.items():
        if not t_indices:
            result[node] = []
            continue
        # Convert t_indices → actual timestamps
        actual_times = [unique_times[ti] for ti in t_indices]
        result[node] = find_minimal_intervals(actual_times, k)

    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def sum_span(active_intervals: Dict[int, List[Tuple[int, int]]]) -> int:
    """Total span = sum over all nodes of sum of (end - start) for each interval."""
    total = 0
    for intervals in active_intervals.values():
        for s, e in intervals:
            total += e - s
    return total


def coverage_fraction(
    active_intervals: Dict[int, List[Tuple[int, int]]],
    edges: torch.Tensor,    # (E, 3): u_idx, v_idx, t_idx
    unique_times: List[int],
) -> float:
    """
    Fraction of temporal edges (u,v,t) where at least one endpoint is active
    (i.e., t falls within one of that node's intervals).
    """
    if edges.numel() == 0:
        return 1.0

    def is_active(node: int, actual_t: int) -> bool:
        for s, e in active_intervals.get(node, []):
            if s <= actual_t <= e:
                return True
        return False

    covered = 0
    total = 0
    for u, v, ti in edges.tolist():
        t = unique_times[ti]
        if is_active(u, t) or is_active(v, t):
            covered += 1
        total += 1

    return covered / total if total > 0 else 1.0
