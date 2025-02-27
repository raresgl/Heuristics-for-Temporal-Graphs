from datetime import datetime, timedelta
import numpy as np
import copy
import sys
import pickle
import graph as utils
import operator
from typing import List, Tuple, Dict

# def kbaseline(timestamps: List[Tuple[int, str, str]], k: int = 3) -> Tuple[Dict, Dict]:
#     """
#     Implements Baseline algorithm for finding vertex cover with k intervals per node.
    
#     Args:
#         timestamps: List of (time, node1, node2) events to cover, sorted by time
#         k: Maximum number of intervals allowed per node
        
#     Returns:
#         Tuple of (Xstart, Xend) dictionaries mapping nodes to their interval times
#     """
#     # Create index mapping nodes to their events
#     nodeEdgeIndex = {}
#     nodeEdgeIndex = utils.indexMapping(timestamps)
#     #Initialize the dictionaries to store the start and end times of the intervals for each node
#     Xstart = {n: {} for n in nodeEdgeIndex.keys()}
#     Xend = {n: {} for n in nodeEdgeIndex.keys()}
    
#     remaining = timestamps.copy()
#     while remaining:
        
#         nodeEdgeIndex = utils.indexMapping(remaining)

#         # Calculate gain for each node
#         gain = {}
#         for node, events in nodeEdgeIndex.items():
#             total_gap = get_len_wo_kgaps(nodeEdgeIndex, node, k)
#             time_span = events[-1][0] - events[0][0]
#             if time_span - total_gap > 0:
#                 gain[node] = float(len(events))/(time_span - total_gap)
#             else:
#                 gain[node] = float(len(events))

#         # Select node with highest gain
#         node = max(gain.items(), key=operator.itemgetter(1))[0]

#         # Find optimal k intervals
#         s = [remaining[0][0]] + [i[0] for i in nodeEdgeIndex[node]] + [remaining[-1][0]] 
#         #s contains the start time of the first event, all the timestamps for the node with the highest gain, and the timestamp of the last event
#         gaps = np.argsort([(s[i+1] - s[i]) for i in range(len(s)-1)])[::-1]
#         #gaps contains the time differences between timestamps in s
#         starts = [s[0]] + [s[i+1] for i in sorted(gaps[:min((k-1),len(gaps))])]
#         #starts inlcudes the starting point and the timestamps that follow the largest gaps
#         ends = [s[i] for i in sorted(gaps[:min((k-1),len(gaps))])] + [s[-1]]
#         #ends will mark the end of the intervals and the last timestamp
        
#         # Adjust interval boundaries
#         if starts[0] not in [i[0] for i in nodeEdgeIndex[node]]:   
#             starts[0] = s[1]
            
#         if ends[-1] not in [i[0] for i in nodeEdgeIndex[node]]:
#             ends[-1] = s[-2]
       
#         g_num = k-1
#         left, right = 0, 0
#         if starts[0] > ends[0]:            
#             left = 1
#         if starts[-1] > ends[-1]:
#             right = 1
            
#         if g_num + left + right > k-1:
#             starts = [s[0]] + [s[i+1] for i in sorted(gaps[:min((g_num + left + right),len(gaps))])]
#             ends = [s[i] for i in sorted(gaps[:min((g_num + left + right),len(gaps))])] + [s[-1]]
#             if left == 1:
#                 del starts[0]
#                 del ends[0]
#             if right == 1:
#                 del starts[-1]
#                 del ends[-1]
                
#         if starts[0] not in [i[0] for i in nodeEdgeIndex[node]]:   
#             starts[0] = s[1]
            
#         if ends[-1] not in [i[0] for i in nodeEdgeIndex[node]]:
#             ends[-1] = s[-2]
       
#         # Store intervals for selected node
#         Xstart[node] = {i: starts[i] if i<len(starts) else starts[-1] for i in range(len(starts))}
#         Xend[node] = {i: ends[i] if i<len(ends) else ends[-1] for i in range(len(ends))}
        
#         # Remove covered events
#         remaining = [x for x in remaining if x[1] != node and x[2] != node]
        
#     return Xstart, Xend

# def get_len_wo_kgaps(nodeEdgeIndex: Dict, u: str, k: int = 3) -> float:
#     """Calculate total gap length after removing k largest gaps.
    
#     Args:
#         nodeEdgeIndex: Dict mapping nodes to their events
#         u: Node to analyze
#         k: Number of gaps to remove
        
#     Returns:
#         Total length of remaining gaps
#     """
#     s = [i[0] for i in nodeEdgeIndex[u]]
#     gaps = sorted([(s[i+1] - s[i]) for i in range(len(s)-1)])[::-1]
#     return sum(gaps[0:min(k, len(gaps))])
def kbaseline(timestamps, k=3):
    """
    Implements Baseline algorithm
    Parameters
    ----------
    timestamps : list of tuples
        Sorted list of interactions [(t1, n1, n2), (t2, n3, n4),...], where t is timestamp, n1 and n2 are interacting nodes.
        Nodes in the interactions are sorted lexicographically.
    maxiter : int
        maximum number of interactions in binary search
    Returns
    -------
    tuple of dicts
        (dict1, dict2): dict1 of a shape {n1: {0: t10, 1: t11, 2: t12}, n2: {0: t20, 1: t21, 2: t22}, ..}, tij is a starting point of jth activity interval of node ni.
                        dict2 of a shape {n1: {0: t10, 1: t11, 2: t12}, n2: {0: t20, 1: t21, 2: t22}, ..}, tij is an ending point of jth activity interval of node ni.
    """
    nodeEdgeIndex = utils.indexMapping(timestamps)
    Xstart = {n: {} for n in nodeEdgeIndex.keys()}
    Xend = {n: {} for n in nodeEdgeIndex.keys()}
    
    while timestamps:
        nodeEdgeIndex = utils.indexMapping(timestamps)
        gain = {node: np.float64(len(tstmps))/(tstmps[-1][0] - tstmps[0][0] - get_len_wo_kgaps(nodeEdgeIndex, node, k)) 
               for node, tstmps in nodeEdgeIndex.items()}
        node, gain = (sorted(gain.items(), key=operator.itemgetter(1)))[-1]
        s = [timestamps[0][0]] + [i[0] for i in nodeEdgeIndex[node]] + [timestamps[-1][0]] 
        gaps = np.argsort([(s[i+1] - s[i]) for i in range(len(s)-1)])[::-1]
        starts = [s[0]] + [s[i+1] for i in sorted(gaps[:min((k-1),len(gaps))])]
        ends = [s[i] for i in sorted(gaps[:min((k-1),len(gaps))])] + [s[-1]]
        
        if starts[0] not in [i[0] for i in nodeEdgeIndex[node]]:   
            starts[0] = s[1]
            
        if ends[-1] not in [i[0] for i in nodeEdgeIndex[node]]:
            ends[-1] = s[-2]
       
        g_num = k-1
        left, right = 0, 0
        if starts[0] > ends[0]:            
            left = 1
        if starts[-1] > ends[-1]:
            right = 1
        if g_num + left + right > k-1:
            starts = [s[0]] + [s[i+1] for i in sorted(gaps[:min((g_num + left + right),len(gaps))])]
            ends = [s[i] for i in sorted(gaps[:min((g_num + left + right),len(gaps))])] + [s[-1]]
            if left == 1:
                del starts[0]
                del ends[0]
            if right == 1:
                del starts[-1]
                del ends[-1]
                
        if starts[0] not in [i[0] for i in nodeEdgeIndex[node]]:   
            starts[0] = s[1]
            
        if ends[-1] not in [i[0] for i in nodeEdgeIndex[node]]:
            ends[-1] = s[-2]
       
        Xstart[node] = {i: starts[i] if i<len(starts) else ends[-1] for i in range(len(starts))}
        Xend[node] = {i: ends[i] if i<len(ends) else ends[-1] for i in range(len(ends))}
        
        timestamps = [x for x in timestamps if x[1] != node and x[2] != node]
    
    return Xstart, Xend

def get_len_wo_kgaps(nodeEdgeIndex, u, k=3):
    s = [i[0] for i in nodeEdgeIndex[u]]
    gaps = sorted([(s[i+1] - s[i]) for i in range(len(s)-1)])[::-1]
    return sum(gaps[0:min(k, len(gaps))])