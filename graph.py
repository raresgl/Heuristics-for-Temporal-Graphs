from datetime import datetime, timedelta
import numpy as np
import networkx as nx
from networkx import configuration_model
from networkx.utils import powerlaw_sequence
import random
from random import shuffle
import time
from typing import List, Dict, Tuple, Optional

def readdata(filename: str, unix: bool = True) -> List[Tuple[int, str, str]]:
    """Read temporal network data from file and return sorted timestamps.
    
    Args:
        filename: Path to input file
        unix: Whether timestamps are in unix format or datetime strings
        
    Returns:
        List of (timestamp, node1, node2) tuples sorted by time
    """
    timestamps = []
    with open(filename, 'r') as f:    
        for line in f:                
            line = line.strip().split(' ')
            if not unix:
                tstr = line[0][1:] + ' ' + line[1][:-1]
                timestamp = time.mktime(datetime.strptime(tstr, '%Y-%m-%d %H:%M:%S').timetuple())
                tst, n1, n2 = int(timestamp), line[1], line[2]
            else:
                timestamp = int(line[0])
                tst, n1, n2 = int(line[0]), int(line[1]), int(line[2])
            
            if n1 != n2:  # Skip self-loops
                if n2 < n1:
                    n1, n2 = n2, n1
                timestamps.append((tst, n1, n2))
            
        timestamps.sort()
    return timestamps

def readgraph(filename: str) -> nx.Graph:
    """Read static graph from file.
    
    Args:
        filename: Path to input file
        
    Returns:
        NetworkX undirected graph
    """
    G = nx.Graph()
    with open(filename, 'r') as f:    
        for line in f:                
            line = line.strip().split(' ')
            n1, n2 = int(line[2]), int(line[3])
            if n1 != n2:  # Skip self-loops
                G.add_edge(n1, n2)
    return G

def generateGraph(n: int = 100) -> nx.Graph:
    """Generate random power law graph.
    
    Args:
        n: Number of nodes
        
    Returns:
        NetworkX undirected graph with power law degree distribution
    """
    z = [random.randint(1, 5) for _ in range(n)] 
    if sum(z) % 2 == 1:  
        z[0] += 1  # make the sum even by incrementing first node's degree
    G = nx.configuration_model(z, seed=42)
    G = nx.Graph(G)  # Remove parallel edges
    G.remove_edges_from(list(nx.selfloop_edges(G)))  # Remove self loops
    return G

def generateIntervals(G: nx.Graph, 
                     distance_in: int = 1,
                     event_length: int = 10, 
                     distance_inter: int = 1,
                     overlap: float = 0.1,
                     seed: float = 1.0,
                     number_intervals: int = 10) -> Tuple[List[Tuple[int, int, int]], Dict]:
    """Generate temporal network with intervals of activity.
    
    Args:
        G: Base graph structure
        distance_in: Time between events within interval
        event_length: Number of events per interval
        distance_inter: Time between intervals
        overlap: Fraction of interval overlap
        seed: Random seed
        number_intervals: Number of intervals per node
        
    Returns:
        Tuple of (timestamps, active_periods):
            timestamps: List of (time, node1, node2) events
            active_periods: Dict mapping nodes to their active time intervals
    """
    G.remove_nodes_from(nx.isolates(G)) #removes nodes with no neighbours
    random.seed(seed)
    timestmps = []
    nodes = list(G.nodes())
    active = {n: [] for n in nodes}
    allnodes = []
    for _ in range(number_intervals):
        shuffle(nodes)
        allnodes += nodes
    
    current = 0
    for n in allnodes:
        l = event_length
        s, f = current, current + distance_in*(l-1)
        randt = list(s + np.random.rand(l-2)*(f-s)) + [s, f]
        
        for i in range(l):
            neigh = list(G.neighbors(n))
            n2 = np.random.choice(neigh)
            n1, n2 = min(n, n2), max(n, n2)
            timestmps.append((randt[i], n1, n2))

        if overlap > 0:
            current = max(f - int((f-s)*overlap), 0)
        else:
            current = f + distance_inter
        active[n].append((s,f))
        
    timestmps = sorted(list(set(timestmps)))
    return timestmps, active

def checkCoverage(Xstart: Dict, 
                  Xend: Dict,
                  timeSlackIndex: List[Tuple]) -> List[Tuple]:
    """Check which events are covered by node active periods.
    
    Args:
        Xstart: Dict mapping nodes to list of interval start times
        Xend: Dict mapping nodes to list of interval end times
        timeSlackIndex: List of (time, node1, node2) events to check
        
    Returns:
        List of uncovered events
    """
    uncovered = []    
    for (t,u,v) in timeSlackIndex:
        if not any(s <= t <= e for s, e in zip(Xstart[u], Xend[u])) and \
           not any(s <= t <= e for s, e in zip(Xstart[v], Xend[v])):
            uncovered.append((t,u,v))
    return uncovered

def getCost(Xstart, Xend):
    total_cost = 0
    for node in Xstart.keys():
        for s, e in zip(Xstart[node], Xend[node]):
            total_cost += e - s + 1
    return total_cost

def getMax(Xstart: Dict, Xend: Dict) -> float:
    """Find maximum interval duration.
    
    Args:
        Xstart: Dict mapping nodes to list of interval start times 
        Xend: Dict mapping nodes to list of interval end times
        
    Returns:
        Duration of longest interval
    """
    return max((e - s
               for n in Xstart
               for s, e in zip(Xstart[n].values(), Xend[n].values())
               if s > -np.inf and e < np.inf),
              default=0)

def compareGT(Xstart: Dict, Xend: Dict, active_truth: Dict, timestamps: List[Tuple]) -> Tuple[float, float, float]:
    """Compare predicted intervals against ground truth.
    
    Args:
        Xstart: Dict mapping nodes to interval start times
        Xend: Dict mapping nodes to interval end times 
        active_truth: Dict mapping nodes to list of true active intervals
        timestamps: List of (time, node1, node2) events
        
    Returns:
        Tuple of (precision, recall, f-measure)
    """
    p, r = {}, {}
    TP = {}  # True positives
    P = {}   # Positives in ground truth
    S = {}   # Selected/predicted positives
    
    for (t, n1, n2) in timestamps:
        for n in [n1, n2]:
            active_ints = active_truth[n]
            
            if n not in P:
                P[n] = 0.0
            if n not in S:
                S[n] = 0.0
            if n not in TP:
                TP[n] = 0.0
            
            # Check if time t is in ground truth intervals
            tp = 0.0
            for (s, f) in active_ints:
                if s <= t <= f:
                    tp = 1.0
            P[n] += tp
            
            # Check if time t is in predicted intervals  
            ts = 0.0
            for i in range(len(Xstart.get(int(n), []))): 
                if Xstart[n][i] <= t <= Xend[n][i]:
                    ts = 1.0
            S[n] += ts
            
            # Count true positives
            if ts == 1.0 and tp == 1.0:
                TP[n] += 1.0
                
    # Calculate precision and recall per node
    for n in P:
        p[n] = TP[n]/S[n] if S[n] > 0 else 0.0
        r[n] = TP[n]/P[n] if P[n] > 0 else 0.0
        
    # Calculate average precision and recall
    p, r = np.mean(list(p.values())), np.mean(list(r.values()))
    
    # Calculate F-measure
    f = 2.0 * (p * r)/(p + r) if (p + r) > 0 else 0.0
    
    return p, r, f

def indexMapping(timestamps: List[Tuple[int, str, str]]):
    # Create index mapping nodes to their events
    index_mapping = {}
    for t in timestamps:
        if t[1] not in index_mapping:
            index_mapping[t[1]] = []
        if t[2] not in index_mapping:
            index_mapping[t[2]] = []
        index_mapping[t[1]].append(t)
        index_mapping[t[2]].append(t)
        #The node is the key, the value is a list of tuples (timestamp, node1, node2)
    return index_mapping

def neighborMapping(timestamps: List[Tuple[int, str, str]]) -> Dict[str, List[Tuple[str, int]]]:
    """Create a mapping of nodes to their neighbors and the timestamps of their connections.
    
    Args:
        timestamps: List of (time, node1, node2) events
        
    Returns:
        Dictionary mapping nodes to a list of tuples (neighbor, timestamp)
    """
    neighbor_mapping = {}
    for t in timestamps:
        time, n1, n2 = t
        if n1 not in neighbor_mapping:
            neighbor_mapping[n1] = []
        if n2 not in neighbor_mapping:
            neighbor_mapping[n2] = []
        neighbor_mapping[n1].append((n2, time))
        neighbor_mapping[n2].append((n1, time))
        # The node is the key, the value is a list of tuples (neighbor, timestamp)
    return neighbor_mapping