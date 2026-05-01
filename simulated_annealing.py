import random
import math
from collections import defaultdict, namedtuple
import heapq
import copy

Interval = namedtuple("Interval", ["s", "e"])  # inclusive endpoints

class TemporalGraph:
    def __init__(self, edges):
        # edges: list of (u,v,t) with integer t
        self.edges = edges
        self.vertices = sorted({u for u,v,t in edges} | {v for u,v,t in edges})
        self.times = sorted({t for u,v,t in edges})
        self.tmin = min(self.times)
        self.tmax = max(self.times)
        # map time -> list of (u,v,edge_id)
        self.time_edges = defaultdict(list)
        for idx, (u,v,t) in enumerate(edges):
            self.time_edges[t].append((u,v,idx))
        self.m = len(edges)

class State:
    def __init__(self, graph, k):
        self.graph = graph
        self.k = k
        # dict v -> list of Interval
        self.intervals = {v: [] for v in graph.vertices}
        # track coverage per edge index: True/False
        self.edge_covered = [False] * graph.m
        # counts per time (number of covered edges at that time)
        self.time_covered_counts = {t: 0 for t in graph.times}

    def total_length(self):
        s = 0
        for v, ivs in self.intervals.items():
            for I in ivs:
                s += max(0, I.e - I.s)
        return s

    def uncovered_count(self):
        return sum(0 if c else 1 for c in self.edge_covered)

    def energy(self, penalty_lambda):
        return self.total_length() + penalty_lambda * self.uncovered_count()

    def copy(self):
        st = State(self.graph, self.k)
        st.intervals = {v: [Interval(iv.s, iv.e) for iv in ivs] for v, ivs in self.intervals.items()}
        st.edge_covered = list(self.edge_covered)
        st.time_covered_counts = dict(self.time_covered_counts)
        return st

    # helper: rebuild coverage from scratch (useful after arbitrary changes)
    def rebuild_coverage(self):
        self.edge_covered = [False] * self.graph.m
        self.time_covered_counts = {t:0 for t in self.graph.times}
        # for each vertex interval, mark edges at times within interval as covered if they touch v
        for idx, (u,v,t) in enumerate(self.graph.edges):
            # check whether u or v active at t
            covered = False
            for I in self.intervals[u]:
                if I.s <= t <= I.e:
                    covered = True
                    break
            if not covered:
                for I in self.intervals[v]:
                    if I.s <= t <= I.e:
                        covered = True
                        break
            self.edge_covered[idx] = covered
            if covered:
                self.time_covered_counts[t] += 1

    # fast incremental apply or revert could be implemented; simpler: rebuild when needed

# Greedy initializer: for each uncovered edge add a tiny 1-length interval to one endpoint
def greedy_initial_state(graph, k):
    st = State(graph, k)
    # choose for each time t, for edges at t, assign coverage by preferring vertices that already have intervals
    for t in graph.times:
        for (u,v,idx) in graph.time_edges[t]:
            if st.edge_covered[idx]:
                continue
            # try to attach to u or v: pick the one with < k intervals and with existing interval covering t if possible
            choice = None
            for cand in [u, v]:
                # if cand already has an interval that covers t -> prefer it
                for I in st.intervals[cand]:
                    if I.s <= t <= I.e:
                        choice = cand
                        break
                if choice:
                    break
            if not choice:
                # choose endpoint with fewer intervals
                cu = len(st.intervals[u])
                cv = len(st.intervals[v])
                if cu < k or cv < k:
                    choice = u if cu <= cv and cu < k else v if cv < k else (u if cu <= cv else v)
                else:
                    # both full: assign to endpoint with minimal added length if we extend an existing interval
                    choice = u if cu <= cv else v
            # add a minimal interval [t,t] or extend an existing interval to include t
            added = False
            for i, I in enumerate(st.intervals[choice]):
                if I.s <= t <= I.e:
                    added = True
                    break
                # if t is adjacent, extend
                if I.e + 1 == t:
                    st.intervals[choice][i] = Interval(I.s, t)
                    added = True
                    break
                if t + 1 == I.s:
                    st.intervals[choice][i] = Interval(t, I.e)
                    added = True
                    break
            if not added:
                if len(st.intervals[choice]) < k:
                    st.intervals[choice].append(Interval(t, t))
                else:
                    # fallback: extend a random interval (worst-case)
                    i = random.randrange(len(st.intervals[choice]))
                    I = st.intervals[choice][i]
                    ns = min(I.s, t)
                    ne = max(I.e, t)
                    st.intervals[choice][i] = Interval(ns, ne)
            # mark covered (we'll rebuild after finishing)
    st.rebuild_coverage()
    return st

# Neighbor generator: several move types
def random_neighbor(state):
    g = state.graph
    st = state.copy()
    move_type = random.choices(
        ["shift", "add", "remove", "merge", "split", "reassign", "local_fix"],
        [0.25, 0.15, 0.10, 0.10, 0.10, 0.20, 0.10]
    )[0]

    if move_type == "shift":
        # choose a random interval and shift one endpoint by ±1 (stay in [tmin,tmax])
        v = random.choice(g.vertices)
        if not st.intervals[v]:
            return st
        i = random.randrange(len(st.intervals[v]))
        I = st.intervals[v][i]
        if random.random() < 0.5:
            # shift left endpoint
            delta = random.choice([-1, 1])
            ns = max(g.tmin, I.s + delta)
            if ns <= I.e:
                st.intervals[v][i] = Interval(ns, I.e)
        else:
            # shift right endpoint
            delta = random.choice([-1, 1])
            ne = min(g.tmax, I.e + delta)
            if I.s <= ne:
                st.intervals[v][i] = Interval(I.s, ne)

    elif move_type == "add":
        v = random.choice(g.vertices)
        if len(st.intervals[v]) < st.k:
            # create small interval around a random time (prefer uncovered times)
            if random.random() < 0.7:
                # choose an uncovered edge time (if any)
                unc_times = [t for t in g.times if any(not st.edge_covered[idx] for (_,_,idx) in g.time_edges[t])]
                if unc_times:
                    t = random.choice(unc_times)
                else:
                    t = random.choice(g.times)
            else:
                t = random.choice(g.times)
            s = max(g.tmin, t - random.randint(0,1))
            e = min(g.tmax, t + random.randint(0,1))
            st.intervals[v].append(Interval(s,e))

    elif move_type == "remove":
        v = random.choice(g.vertices)
        if st.intervals[v]:
            i = random.randrange(len(st.intervals[v]))
            st.intervals[v].pop(i)

    elif move_type == "merge":
        v = random.choice(g.vertices)
        if len(st.intervals[v]) >= 2:
            i,j = random.sample(range(len(st.intervals[v])), 2)
            I = st.intervals[v][i]
            J = st.intervals[v][j]
            ns = min(I.s, J.s)
            ne = max(I.e, J.e)
            # merge into one
            st.intervals[v][i] = Interval(ns, ne)
            st.intervals[v].pop(j if j>i else j)  # remove other

    elif move_type == "split":
        v = random.choice(g.vertices)
        if st.intervals[v]:
            i = random.randrange(len(st.intervals[v]))
            I = st.intervals[v][i]
            if I.e - I.s >= 2 and len(st.intervals[v]) < st.k:
                # choose split point
                mid = random.randint(I.s+1, I.e-1)
                st.intervals[v][i] = Interval(I.s, mid)
                st.intervals[v].append(Interval(mid+1, I.e))

    elif move_type == "reassign":
        # move an interval from one vertex to the other endpoint of some uncovered edge
        # pick a random uncovered edge (if none pick any)
        unc_edges = [idx for idx,covered in enumerate(st.edge_covered) if not covered]
        if unc_edges:
            idx = random.choice(unc_edges)
        else:
            idx = random.randrange(g.m)
        u,v,t = g.edges[idx]
        # pick endpoint that has intervals to move (or either)
        src = random.choice([u,v])
        dst = v if src == u else u
        if st.intervals[src]:
            i = random.randrange(len(st.intervals[src]))
            I = st.intervals[src].pop(i)
            # optionally shrink I so it focuses near t
            if I.s <= t <= I.e and random.random() < 0.7:
                I = Interval(max(I.s, t), min(I.e, t))
            if len(st.intervals[dst]) < st.k:
                st.intervals[dst].append(I)
            else:
                # revert move (put back)
                st.intervals[src].append(I)

    elif move_type == "local_fix":
        # add tiny interval at an uncovered time to one of its endpoints
        unc_edges = [idx for idx,covered in enumerate(st.edge_covered) if not covered]
        if unc_edges:
            idx = random.choice(unc_edges)
            u,v,t = g.edges[idx]
            cand = u if len(st.intervals[u]) < st.k else (v if len(st.intervals[v]) < st.k else random.choice([u,v]))
            st.intervals[cand].append(Interval(t,t))

    # after move, rebuild coverage
    st.rebuild_coverage()
    return st

def simulated_annealing(graph, k,
                        max_iters=20000,
                        T0=None,
                        alpha=0.995,
                        penalty_lambda=None,
                        seed=None,
                        verbose=False):
    if seed is not None:
        random.seed(seed)

    if penalty_lambda is None:
        penalty_lambda = 10 * graph.m  # heuristic

    if T0 is None:
        T0 = penalty_lambda * 0.5

    current = greedy_initial_state(graph, k)
    best = current.copy()
    bestE = best.energy(penalty_lambda)
    curE = current.energy(penalty_lambda)

    T = T0
    no_improve = 0
    for it in range(1, max_iters+1):
        nb = random_neighbor(current)
        nbE = nb.energy(penalty_lambda)
        dE = nbE - curE
        accept = False
        if dE <= 0 or random.random() < math.exp(-dE / max(T, 1e-12)):
            current = nb
            curE = nbE
            accept = True
        if curE < bestE:
            best = current.copy()
            bestE = curE
            no_improve = 0
            if verbose:
                print(f"iter {it}: new best E={bestE} len={best.total_length()} unc={best.uncovered_count()}")
        else:
            no_improve += 1

        # cooling
        T *= alpha

        # optional early stopping if feasible and no improvement
        if best.uncovered_count() == 0 and no_improve > 5000:
            break

    # final greedy repair if needed
    if best.uncovered_count() > 0:
        if verbose: print("Repairing final solution...")
        repair_greedy(best)
        best.rebuild_coverage()

    return best

def repair_greedy(state):
    # Add minimal intervals to cover any remaining uncovered edges, prefer endpoints with free slots
    g = state.graph
    for idx, covered in enumerate(state.edge_covered):
        if covered:
            continue
        u,v,t = g.edges[idx]
        chosen = None
        if len(state.intervals[u]) < state.k:
            chosen = u
        elif len(state.intervals[v]) < state.k:
            chosen = v
        else:
            # extend an interval minimally on the endpoint that causes least added length
            # compute cost of extending each interval on u and v to include t
            best_cost = None
            best_v = None
            for cand in [u, v]:
                min_cost = None
                for I in state.intervals[cand]:
                    extra = 0
                    if t < I.s:
                        extra = I.s - t
                    elif t > I.e:
                        extra = t - I.e
                    else:
                        extra = 0
                    if min_cost is None or extra < min_cost:
                        min_cost = extra
                if best_cost is None or min_cost < best_cost:
                    best_cost = min_cost
                    best_v = cand
            chosen = best_v
        # add or extend
        added = False
        for i,I in enumerate(state.intervals[chosen]):
            if I.s <= t <= I.e:
                added = True
                break
            if t < I.s:
                state.intervals[chosen][i] = Interval(t, I.e)
                added = True
                break
            if t > I.e:
                state.intervals[chosen][i] = Interval(I.s, t)
                added = True
                break
        if not added:
            if len(state.intervals[chosen]) < state.k:
                state.intervals[chosen].append(Interval(t,t))
            else:
                # as fallback extend first interval
                I = state.intervals[chosen][0]
                ns = min(I.s, t)
                ne = max(I.e, t)
                state.intervals[chosen][0] = Interval(ns, ne)
    # rebuild coverage
    state.rebuild_coverage()

# Example usage:
if __name__ == "__main__":
    # small toy temporal graph edges
    edges = [
        ("A","B",1), ("A","C",2), ("B","C",2),
        ("C","D",3), ("A","D",4), ("B","D",4)
    ]
    g = TemporalGraph(edges)
    k = 1
    best = simulated_annealing(g, k, max_iters=10000, alpha=0.995, seed=42, verbose=True)
    print("Best total length:", best.total_length())
    for v in g.vertices:
        print(v, best.intervals[v])
    print("Uncovered edges:", best.uncovered_count())
