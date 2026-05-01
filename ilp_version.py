from gurobipy import *
from math import floor


def ilp(timestamps, G, k):

    T = max(t[0] for t in timestamps) + 1
    V = [x for x in range(0,G.number_of_nodes())]
    
    m = Model("Temporal Vertex Cover")
    m._T = T
    m._V = V
    x = m.addVars(V, range(T), vtype=GRB.BINARY, name='x')
    y = m.addVars(V, range(T-1), vtype=GRB.BINARY, name="y")
    #m.setParam('Method', 1)
    m.setObjective(quicksum(x[v,t] for v in V for t in range(T)), GRB.MINIMIZE)

    processed_edges = set()
    for t, u, v in timestamps:
        int_t = floor(t)
        if (int_t, u, v) not in processed_edges:
            m.addConstr(x[u,int_t] + x[v,int_t] >= 1)
            processed_edges.add((int_t, u, v))

    for v in V:
        for t in range(T-1):
            m.addConstr(y[v,t] >= x[v,t] - x[v,t+1])
            m.addConstr(y[v,t] >= x[v,t+1] - x[v,t])
    
    for v in V:
        m.addConstr(quicksum(x[v,t] for t in range(T)) >= 1)
        m.addConstr(quicksum(y[v,t] for t in range(T-1)) <= 2*k)

    return m, x


def active_intervals(m,x):
    if m.status != GRB.OPTIMAL:
        print(f"Warning: Model status is {m.status}, not optimal")
    
    V = m._V
    T = m._T
    active_intervals = {v: [] for v in V}
    for v in V:
        interval_start = None

        for t in range(T):
            is_active = x[v,t].X > 0.5
            
            if is_active and interval_start == None:
                interval_start = t
            elif not is_active and interval_start != None:
                active_intervals[v].append((interval_start, t-1))
                interval_start = None
        if interval_start is not None:
            active_intervals[v].append((interval_start, T-1))
    return active_intervals



