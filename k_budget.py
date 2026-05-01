from datetime import datetime, timedelta
import numpy as np
import copy
import sys
import pickle
import graph as utils

def runKBudget(timestamps, k, maxiters=10, m=None):
    """
    Implements k-Budget algorithm

    Parameters
    ----------
    timestamps : list of tuples
        Sorted list of interactions [(t1, n1, n2), (t2, n3, n4),...], where t is timestamp, n1 and n2 are interactiong nodes.
        Nodes in the interactions are sorted lexicographically.
    k : int 
        number of activity intervals
    maxiters : int
        maximum number of interactions in binary search and gap points search
    m : dict
        dict of gap points {n: [t1, t2, ..]} where n is a node, list t1, t2 .. is a list of its k-1 inactive (gap) points

    Returns
    -------
    tuple of dicts
        (dict1, dict2): dict1 of a shape {n1: {0: t10, 1: t11, 2: t12}, n2: {0: t20, 1: t21, 2: t22}, ..}, tij is a starting point of jth activity interval of node ni.
                        dict2 of a shape {n1: {0: t10, 1: t11, 2: t12}, n2: {0: t20, 1: t21, 2: t22}, ..}, tij is an ending point of jth activity interval of node ni.

    """
    if m is None:
        m = {}
        
    nodeEdgeIndex = utils.indexMapping(timestamps)

    if not m:
        m = getGapPoints(nodeEdgeIndex, timestamps[0][0], timestamps[-1][0], k)
    Xstart, Xend = runKBudget_BS(timestamps, nodeEdgeIndex, k, maxiters=maxiters, m=m)
    for i in range(maxiters):
        m = getNewGapPoints(nodeEdgeIndex, timestamps[0][0], timestamps[-1][0], Xstart, Xend)
        Xstart, Xend = runKBudget_BS(timestamps, nodeEdgeIndex, k, maxiters=maxiters, m=m)
        
    return Xstart, Xend

def getGapPoints(nodeEdgeIndex, left_border, right_border, k):
    m = {}    
    for n, ts_list in nodeEdgeIndex.items():
        # Extract timestamps from the list of tuples
        timestamps = [t[0] for t in ts_list]
        s = [left_border] + sorted(timestamps) + [right_border]
        
        gaps = np.argsort([(s[i+1] - s[i]) for i in range(len(s)-1)])[::-1]        
        m[n] = sorted([(s[i]+s[i+1])/2.0 for i in gaps[:min((k-1), len(gaps))]])
    return m
    
def getTrueGapPoints(nodeEdgeIndex, active_truths):
    m = {}  
    for n, ints in active_truths.items():
        m[n] = []
        a = ints[0][1]
        for i in range(1, len(ints)):
            b = ints[i][0]            
            m[n].append((a + b)/2.0)
            a = ints[i][1]
    return m
    
def getNewGapPoints(nodeEdgeIndex, left_border, right_border, Xstart, Xend):
    m = {}  
    for n in Xstart.keys():
        m[n] = []
        a = Xend[n][0]
        for i in range(1, len(Xstart[n])):
            b = Xstart[n][i]
            if a == -np.inf:
                a = float(left_border)
            if b == np.inf:
                b = float(right_border)
            m[n].append((a + b)/2.0)
            a = Xend[n][i]
    return m

def setbudget(nodeEdgeIndex, kfactor):
    b = {}    
    for n, ts_list in nodeEdgeIndex.items():
        # Extract timestamps from the list of tuples
        timestamps = [t[0] for t in ts_list]
        b[n] = (max(timestamps) - min(timestamps)) * kfactor
    return b
    
def DFS(v, t, nodeEdgeIndex, O, Orev, b, S, visited_tracker):
    S.append((v, t))
    visited_tracker[v][t] = True    
    
    i = Orev[v][t]
    while O[v][i] and O[v][i][0][0] < (t-b[v][i]):
        (s, u) = O[v][i].pop(0)        
        if not visited_tracker[u].get(s, False):
            DFS(u, s, nodeEdgeIndex, O, Orev, b, S, visited_tracker)
        
    while O[v][i] and O[v][i][-1][0] > (t+b[v][i]):
        (s, u) = O[v][i].pop(-1)
        if not visited_tracker[u].get(s, False):
            DFS(u, s, nodeEdgeIndex, O, Orev, b, S, visited_tracker)
    return
    
def DFS_iterative(v, t, nodeEdgeIndex, O, Orev, b, S, visited_tracker):
    localS = []
    localS.append((v, t))
    while localS:
        (v, t) = localS.pop()
        if not visited_tracker[v].get(t, False):
            visited_tracker[v][t] = True
            S.append((v, t))
            i = Orev[v][t]
            
            tmp = []
            if O[v][i]:
                counter = 0
                while counter < len(O[v][i]) and O[v][i][counter][0] < (t-b[v][i]):
                    tmp.append((O[v][i][counter][1], O[v][i][counter][0]))
                    counter += 1
                counter = -1
                while abs(counter) <= len(O[v][i]) and O[v][i][counter][0] > (t+b[v][i]):
                    tmp.append((O[v][i][counter][1], O[v][i][counter][0]))
                    counter -= 1               
            
            for pair in tmp[::-1]:              
                localS.append(pair)
    return
    
def reverseDFS_iterative(v, t, nodeEdgeIndex, Q, Orev, b, R, visited_tracker):
    localS = []
    localS.append((v, t))
    
    while localS:
        (v, t) = localS.pop()
        if not visited_tracker[v].get(t, False):
            visited_tracker[v][t] = True
            R.append((v, t))
            
            # Find edges that involve this node and timestamp
            edges = []
            for event in nodeEdgeIndex[v]:
                if event[0] == t:
                    edges.append((event[1], event[2]))
                    
            for (n1, n2) in edges:
                u = n2 if n1 == v else n1
                i = Orev[u][t]
                
                tmp = []
                if Q[u][i]:
                    counter = 0
                    while counter < len(Q[u][i]) and Q[u][i][counter] < (t-b[u][i]):
                        tmp.append((u, Q[u][i][counter]))
                        counter += 1
                    counter = -1
                    while abs(counter) <= len(Q[u][i]) and Q[u][i][counter] > (t+b[u][i]):
                        tmp.append((u, Q[u][i][counter]))
                        counter -= 1
                    
                for pair in tmp[::-1]:
                    localS.append(pair)
    return
    
def reverseDFS(v, t, nodeEdgeIndex, Q, Orev, b, R, visited_tracker):
    R.append((v, t))
    visited_tracker[v][t] = True
    
    # Find edges that involve this node and timestamp
    edges = []
    for event in nodeEdgeIndex[v]:
        if event[0] == t:
            edges.append((event[1], event[2]))
            
    for (n1, n2) in edges:
        u = n2 if n1 == v else n1
        i = Orev[u][t]
        
        while Q[u][i] and Q[u][i][0] < (t-b[u][i]):
            s = Q[u][i].pop(0)
            if not visited_tracker[u].get(s, False):
                reverseDFS(u, s, nodeEdgeIndex, Q, Orev, b, R, visited_tracker)
                
        while Q[u][i] and Q[u][i][-1] > (t+b[u][i]):
            s = Q[u][i].pop(-1)
            if not visited_tracker[u].get(s, False):
                reverseDFS(u, s, nodeEdgeIndex, Q, Orev, b, R, visited_tracker)

    return
                   
def clearVisited(nodeEdgeIndex):
    # Create a dictionary to track visited nodes
    visited_tracker = {}
    for node in nodeEdgeIndex:
        visited_tracker[node] = {}
        for event in nodeEdgeIndex[node]:
            timestamp = event[0]
            visited_tracker[node][timestamp] = False
    return visited_tracker

def kbudgetAlgorithm(timestamps, nodeEdgeIndex, b, m):
    # Initialize visited status for all node-timestamp pairs
    visited_tracker = clearVisited(nodeEdgeIndex)
    
    O, Orev = getO(timestamps, m)
    Smain = []
    
    # Adapt the DFS and other functions to work with the visited_tracker
    for n, ts_list in nodeEdgeIndex.items():
        for event in ts_list:
            t = event[0]
            if not visited_tracker[n].get(t, False):
                DFS_iterative(n, t, nodeEdgeIndex, O, Orev, b, Smain, visited_tracker)
    
    # Reset visited status
    visited_tracker = clearVisited(nodeEdgeIndex)
    
    Q = getQ(timestamps, m)
    R = []
    for (node, tm) in Smain[::-1]:
        if not visited_tracker[node].get(tm, False):
            reverseDFS_iterative(node, tm, nodeEdgeIndex, Q, Orev, b, R, visited_tracker)   
    
    p1 = {n: {ind: -np.inf for ind in range(len(mpoints)+1)} for n, mpoints in m.items()}
    i2 = {n: {ind: -np.inf for ind in range(len(mpoints)+1)} for n, mpoints in m.items()}
    p2 = {n: {ind: np.inf for ind in range(len(mpoints)+1)} for n, mpoints in m.items()}
    i1 = {n: {ind: np.inf for ind in range(len(mpoints)+1)} for n, mpoints in m.items()}

    for (v, t) in R[::-1]:
        i = Orev[v][t]
        if p1[v][i] <= t and p2[v][i] >= t:
            p1[v][i] = max(p1[v][i], t - b[v][i])
            p2[v][i] = min(p2[v][i], t + b[v][i])
            i1[v][i] = min(i1[v][i], t)
            i2[v][i] = max(i2[v][i], t)
            
    Xstart = i1
    Xend = i2
    return Xstart, Xend

 
def getO(timestamps, m):   
    O = {n: {ind: [] for ind in range(len(mpoints)+1)} for n, mpoints in m.items()}
    ORev = {n: {} for n in m}
    counter = {n: 0 for n in m}
    
    for (t, n1, n2) in timestamps:
    
        if counter[n1] < len(m[n1]) and t > m[n1][counter[n1]]:
            counter[n1] += 1            
        O[n1][counter[n1]].append((t, n2))        
        ORev[n1][t] = counter[n1]    
        
        if counter[n2] < len(m[n2]) and t > m[n2][counter[n2]]:
            counter[n2] += 1
        O[n2][counter[n2]].append((t, n1))
        ORev[n2][t] = counter[n2]
    return O, ORev
        
def getQ(timestamps, m):
    O = {n: {ind: [] for ind in range(len(mpoints)+1)} for n, mpoints in m.items()}
    counter = {n: 0 for n in m}
    
    for (t, n1, n2) in timestamps:
        if counter[n1] < len(m[n1]) and t > m[n1][counter[n1]]:
            counter[n1] += 1
        O[n1][counter[n1]].append(t)        
                
        if counter[n2] < len(m[n2]) and t > m[n2][counter[n2]]:
            counter[n2] += 1
        O[n2][counter[n2]].append(t)
        
    # Updated dictionary comprehension syntax
    O = {n: {i: sorted(set(l)) for i, l in ind.items()} for n, ind in O.items()}
    return O
    
def runKBudget_fixed_budget(timestamps, nodeEdgeIndex, b=None, m=None, k=10, kfactor=1.0):
    if b is None:
        b = {}
    if m is None:
        m = {}
        
    if not b:
        b = {n: [(timestamps[-1][0]-timestamps[0][0])]*k for n in nodeEdgeIndex}
    if not m:
        m = getGapPoints(nodeEdgeIndex, timestamps[0][0], timestamps[-1][0], k)
    Xstart, Xend = kbudgetAlgorithm(timestamps, nodeEdgeIndex, b, m)
   
    return Xstart, Xend
    
def runKBudget_BS(timestamps, nodeEdgeIndex, kintervals, klow=-1, kup=-1, maxiters=10, m=None):
    if m is None:
        m = {}
        
    maxb = timestamps[-1][0]-timestamps[0][0]
    if klow == -1:
        klow, kup = 1./maxb, 1.0 
    b = {n: [maxb]*kintervals for n in nodeEdgeIndex}
    if not m:
        m = getGapPoints(nodeEdgeIndex, timestamps[0][0], timestamps[-1][0], kintervals)
    LXstart, LXend = kbudgetAlgorithm(timestamps, nodeEdgeIndex, b, m)
    
    alpha = 1.0/maxb
    itercount = 0
    while (kup - klow) > alpha and itercount < maxiters:
        # print('iteration', itercount, kup - klow, alpha)  # Updated print syntax
        itercount += 1
        k = (kup + klow)/2
        b = {n: [maxb*k]*kintervals for n in nodeEdgeIndex}
        Xstart, Xend = kbudgetAlgorithm(timestamps, nodeEdgeIndex, b, m)
        uncovered = utils.checkCoverage(Xstart, Xend, timestamps)
        if uncovered:
            klow = k
        else:
            kup = k
            LXstart, LXend = Xstart, Xend
            
    return LXstart, LXend