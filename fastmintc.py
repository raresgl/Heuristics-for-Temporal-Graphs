"""
FastMinTC+ (Lazzarinetti et al., TIME 2024) — faithful reimplementation.

MinTCover / MinTCover+ is the k=1 timeline-cover problem: each vertex v gets a
single activity interval I_v = [s_v, e_v]; a temporal edge (u,v,t) is covered if
t in [s_u,e_u] or t in [s_v,e_v]; the objective is to minimise the sum-span
S(T) = sum_v (e_v - s_v).

This is the main heuristic baseline that DLMinTC+ compares against, so we need it
to reproduce Lazzarinetti's experimental protocol. The algorithm (their Alg. 1+2):

  InitializeTC (extending): process edges; if an edge is uncovered, add its
    timestamp t to the interval of the higher-degree endpoint.
  InitializeTC (shrinking): drop interval endpoints whose loss is 0.
  Exchange loop (until cutoff): remove the minimum-loss endpoint, remove a
    BMS-selected random endpoint, then re-cover a random uncovered edge via the
    endpoint with greater gain; track the best fully-covering timeline found.

loss(v,t)  = # edges that become uncovered if endpoint (v,t) is removed
gain(v,t)  = # currently-uncovered edges that become covered if (v,t) is added

Public API mirrors the other heuristics (k_inner.runKInner):
  runFastMinTC(timestamps, k=1, cutoff=1.0) -> (Xstart, Xend)
where Xstart/Xend are {node: {0: s_v}} / {node: {0: e_v}} nested dicts.
"""

import random
import time
from bisect import bisect_left, bisect_right
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Problem representation
# ---------------------------------------------------------------------------

class _Instance:
    """Static per-instance data derived from the temporal edge list."""

    def __init__(self, timestamps: List[Tuple[int, int, int]]):
        # timestamps: list of (t, u, v)
        self.edges = [(int(t), int(u), int(v)) for (t, u, v) in timestamps]
        self.nodes = sorted({u for _, u, v in self.edges} | {v for _, u, v in self.edges})
        self.degree: Dict[int, int] = {n: 0 for n in self.nodes}
        # times[v] = sorted unique timestamps at which v has an incident edge
        times: Dict[int, set] = {n: set() for n in self.nodes}
        # inc[v] = list of edge indices incident to v
        self.inc: Dict[int, List[int]] = {n: [] for n in self.nodes}
        for i, (t, u, v) in enumerate(self.edges):
            self.degree[u] += 1
            self.degree[v] += 1
            times[u].add(t)
            times[v].add(t)
            self.inc[u].append(i)
            self.inc[v].append(i)
        self.times: Dict[int, List[int]] = {n: sorted(ts) for n, ts in times.items()}


# ---------------------------------------------------------------------------
# Timeline state: one interval [s_v, e_v] per active vertex + coverage counts
# ---------------------------------------------------------------------------

class _Timeline:
    def __init__(self, inst: _Instance):
        self.inst = inst
        self.s: Dict[int, int] = {}   # start per active vertex
        self.e: Dict[int, int] = {}   # end per active vertex
        # cov[i] = number of endpoints of edge i whose interval currently contains its t
        self.cov = [0] * len(inst.edges)
        self.uncovered = set(range(len(inst.edges)))  # edge indices with cov==0

    # --- coverage bookkeeping -------------------------------------------------
    def _covers(self, v: int, t: int) -> bool:
        return v in self.s and self.s[v] <= t <= self.e[v]

    def _recount_edge(self, i: int) -> None:
        t, u, v = self.inst.edges[i]
        c = (1 if self._covers(u, t) else 0) + (1 if self._covers(v, t) else 0)
        self.cov[i] = c
        if c == 0:
            self.uncovered.add(i)
        else:
            self.uncovered.discard(i)

    def _set_interval(self, v: int, s: int, e: int) -> None:
        """Set/replace v's interval (or remove it if s/e is None) and refresh its edges."""
        if s is None:
            self.s.pop(v, None)
            self.e.pop(v, None)
        else:
            self.s[v] = s
            self.e[v] = e
        for i in self.inst.inc[v]:
            self._recount_edge(i)

    def add_timestamp(self, v: int, t: int) -> None:
        """Extend v's interval to include t (creating it if absent)."""
        if v in self.s:
            self._set_interval(v, min(self.s[v], t), max(self.e[v], t))
        else:
            self._set_interval(v, t, t)

    # --- objective ------------------------------------------------------------
    def sum_span(self) -> int:
        return sum(self.e[v] - self.s[v] for v in self.s)

    def covers_all(self) -> bool:
        return not self.uncovered

    def snapshot(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        return dict(self.s), dict(self.e)


# ---------------------------------------------------------------------------
# loss / gain
# ---------------------------------------------------------------------------

def _loss_of_endpoint(tl: _Timeline, v: int, which: str) -> int:
    """
    # edges that become uncovered if we shrink v's interval by removing the
    endpoint (which in {'s','e'}).  Shrinking replaces the endpoint by the next
    interior edge-timestamp of v; edges of v at timestamps that fall outside the
    shrunken interval and are not covered by their other endpoint are lost.
    """
    if v not in tl.s:
        return 0
    s, e = tl.s[v], tl.e[v]
    inst = tl.inst
    if s == e:
        r_lo, r_hi = None, None  # removing the only timestamp deactivates v
    elif which == 's':
        tv = inst.times[v]
        nxt = tv[bisect_right(tv, s)] if bisect_right(tv, s) < len(tv) else e
        r_lo, r_hi = nxt, e
    else:  # 'e'
        tv = inst.times[v]
        idx = bisect_left(tv, e) - 1
        prev = tv[idx] if idx >= 0 else s
        r_lo, r_hi = s, prev

    loss = 0
    for i in inst.inc[v]:
        t, a, b = inst.edges[i]
        if tl.cov[i] != 1:
            continue  # covered by both (removal is safe) or already uncovered
        # is this edge currently covered *by v*?
        if not (s <= t <= e):
            continue
        # would v still cover it after shrinking?
        still = (r_lo is not None) and (r_lo <= t <= r_hi)
        if not still:
            loss += 1
    return loss


def _gain_of_vertex(tl: _Timeline, v: int, t: int) -> int:
    """# currently-uncovered edges that adding timestamp t to v would cover."""
    inst = tl.inst
    gain = 0
    s = tl.s.get(v)
    e = tl.e.get(v)
    ns = t if s is None else min(s, t)
    ne = t if e is None else max(e, t)
    for i in inst.inc[v]:
        if tl.cov[i] != 0:
            continue
        et = inst.edges[i][0]
        if ns <= et <= ne:
            gain += 1
    return gain


# ---------------------------------------------------------------------------
# Initialization (Algorithm 2): extending + shrinking
# ---------------------------------------------------------------------------

def _initialize(tl: _Timeline) -> None:
    inst = tl.inst
    # --- extending: cover each still-uncovered edge via higher-degree endpoint
    for i, (t, u, v) in enumerate(inst.edges):
        if tl.cov[i] != 0:
            continue
        if inst.degree[u] >= inst.degree[v]:
            tl.add_timestamp(u, t)
        else:
            tl.add_timestamp(v, t)

    # --- shrinking: drop interval endpoints whose loss is 0 (repeat to fixpoint)
    changed = True
    while changed:
        changed = False
        for v in list(tl.s.keys()):
            if v not in tl.s or tl.s[v] == tl.e[v]:
                continue
            for which in ('s', 'e'):
                if v not in tl.s or tl.s[v] == tl.e[v]:
                    break
                if _loss_of_endpoint(tl, v, which) == 0:
                    _shrink(tl, v, which)
                    changed = True


def _shrink(tl: _Timeline, v: int, which: str) -> None:
    """Remove endpoint 's' or 'e' of v, replacing it by the next interior edge-time."""
    s, e = tl.s[v], tl.e[v]
    tv = tl.inst.times[v]
    if which == 's':
        j = bisect_right(tv, s)
        if j < len(tv) and tv[j] <= e:
            tl._set_interval(v, tv[j], e)
        else:
            tl._set_interval(v, None, None)
    else:
        idx = bisect_left(tv, e) - 1
        if idx >= 0 and tv[idx] >= s:
            tl._set_interval(v, s, tv[idx])
        else:
            tl._set_interval(v, None, None)


# ---------------------------------------------------------------------------
# Exchange step (Algorithm 1)
# ---------------------------------------------------------------------------

def _endpoints_list(tl: _Timeline) -> List[Tuple[int, str]]:
    out = []
    for v in tl.s:
        out.append((v, 's'))
        if tl.e[v] != tl.s[v]:
            out.append((v, 'e'))
    return out


def _remove_min_loss(tl: _Timeline) -> None:
    eps = _endpoints_list(tl)
    if not eps:
        return
    best = min(eps, key=lambda ve: _loss_of_endpoint(tl, ve[0], ve[1]))
    _shrink(tl, best[0], best[1])


def _select_rnd_endpoint(tl: _Timeline, bms: int) -> None:
    """BMS: sample `bms` random endpoints, remove the lowest-loss one."""
    eps = _endpoints_list(tl)
    if not eps:
        return
    cand = random.sample(eps, min(bms, len(eps)))
    best = min(cand, key=lambda ve: _loss_of_endpoint(tl, ve[0], ve[1]))
    _shrink(tl, best[0], best[1])


def runFastMinTC(
    timestamps: List[Tuple[int, int, int]],
    k: int = 1,
    cutoff: float = 1.0,
    bms: int = 50,
    seed: int = 0,
) -> Tuple[Dict[int, Dict[int, int]], Dict[int, Dict[int, int]]]:
    """
    FastMinTC+ for the k=1 timeline-cover problem.

    `k` is accepted for interface compatibility with the other heuristics but is
    ignored (the algorithm is single-interval by construction).  `cutoff` is the
    wall-clock budget in seconds for the exchange loop; `bms` is the BMS sample
    size.  Returns (Xstart, Xend) in the repo's nested-dict format.
    """
    random.seed(seed)
    inst = _Instance(timestamps)
    tl = _Timeline(inst)
    if not inst.edges:
        return {}, {}

    _initialize(tl)

    best_s, best_e = tl.snapshot()
    best_span = tl.sum_span() if tl.covers_all() else float('inf')

    start = time.time()
    while time.time() - start < cutoff:
        # FastVC-style search: when the timeline is feasible, greedily shrink by
        # removing the minimum-loss endpoint and re-check (a loss-0 removal is a
        # free span reduction; a loss>0 removal opens a smaller-span search
        # direction).  Only when infeasible do we perform the covering swap.
        if tl.covers_all():
            if tl.sum_span() < best_span:
                best_span = tl.sum_span()
                best_s, best_e = tl.snapshot()
            _remove_min_loss(tl)
            continue

        # infeasible: remove a BMS-selected endpoint, then re-cover a random
        # uncovered edge via its higher-gain endpoint
        _select_rnd_endpoint(tl, bms)
        if tl.uncovered:
            i = random.choice(tuple(tl.uncovered))
            t, u, v = inst.edges[i]
            gu, gv = _gain_of_vertex(tl, u, t), _gain_of_vertex(tl, v, t)
            tl.add_timestamp(u if gu >= gv else v, t)

    # ensure the returned timeline is feasible: fall back to a full re-init if the
    # loop never captured a covering solution (can only happen at tiny cutoffs)
    if best_span == float('inf'):
        tl2 = _Timeline(inst)
        _initialize(tl2)
        best_s, best_e = tl2.snapshot()

    Xstart = {v: {0: best_s[v]} for v in best_s}
    Xend = {v: {0: best_e[v]} for v in best_e}
    return Xstart, Xend
