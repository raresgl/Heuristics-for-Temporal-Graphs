from datetime import datetime, timedelta
import numpy as np
import copy
import sys
import pickle
import graph as utils
import operator
from baseline import *
from typing import List, Tuple, Dict
import random 
from collections import defaultdict   

def proper_search(timestamps: List[Tuple[int, str, str]], k: int = 3) -> Dict[str, int]:

    Xstart, Xend = kbaseline(timestamps, k)
    nodes_edge_index = defaultdict(list)
    loss_dict = {}
    for node in Xstart:
        for interval in Xstart[node]:
            interval_key = (node, interval)
            loss_dict[interval_key] = 0

    for time in timestamps:
        one_covered = False
        int_key = (0,0)
        both_covered = False
        for node in Xstart.keys():
            if node == time[1]:
                for interval in Xstart[node].keys():
                    if Xstart[node][interval] <= time[0] <= Xend[node][interval]:
                        if one_covered == True:
                            both_covered = True
                            break
                        else:
                            one_covered = True
                            int_key = (node, interval)
                            nodes_edge_index[node].append(time[0])
            if node == time[2]:
                for interval in Xstart[node].keys():
                    if Xstart[node][interval] <= time[0] <= Xend[node][interval]:
                        if one_covered == True:
                            both_covered = True
                            break
                        else:
                            one_covered = True
                            int_key = (node, interval)
                            nodes_edge_index[node].append(time[0])
        if one_covered and not both_covered:
            loss_dict[int_key] += 1

    zero_items = [x for x in loss_dict.items()]
    
    for it in range(5):
        for items in loss_dict.items():
            Xstart[items[0][0]] = {}
            Xend[items[0][0]] = {}
            
            
            intervals = find_minimal_intervals(nodes_edge_index[items[0][0]], k)
            for i, (start, end) in enumerate(intervals, 0):
                Xstart[items[0][0]][i] = start
                Xend[items[0][0]][i] = end


    remaining = []
    for t in timestamps:
        # Check if either node in the event is covered at this timestamp
        node1_covered = False
        node2_covered = False
        
        if t[1] in Xstart and Xstart[t[1]]:
            node1_covered = any(Xstart[t[1]][i] <= t[0] <= Xend[t[1]][i] 
                                for i in Xstart[t[1]].keys())
        if t[2] in Xstart and Xstart[t[2]]:
            node2_covered = any(Xstart[t[2]][i] <= t[0] <= Xend[t[2]][i] 
                                for i in Xstart[t[2]].keys())
        
        # Add to remaining if neither node is covered
        if not node1_covered and not node2_covered:
            remaining.append(t)
    print(remaining)

    return Xstart, Xend


def find_minimal_intervals(timestamps, k):

    n = len(timestamps)
    dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
    # prev[i][j] stores the optimal split point for reconstructing the solution
    prev = [[0] * (k + 1) for _ in range(n + 1)]
    
    # Base cases
    dp[n][0] = 0  # Empty sequence needs 0 length
    for j in range(1, k + 1):
        dp[n][j] = 0  # Empty sequence with any number of intervals
    
    # Fill the dp table
    for i in range(n-1, -1, -1):
        for j in range(1, k + 1):
            for l in range(i + 1, n + 1):
                # Try covering timestamps[i:l] with one interval and timestamps[l:] with j-1 intervals
                if l == i + 1:
                    interval_length = 0
                else:
                    interval_length = timestamps[l-1] - timestamps[i]
                    
                total_length = interval_length + dp[l][j-1]
                
                if total_length < dp[i][j]:
                    dp[i][j] = total_length
                    prev[i][j] = l

    # Reconstruct the solution
    intervals = []
    pos = 0
    remaining = k
    
    while pos < n and remaining > 0:
        next_pos = prev[pos][remaining]
        if next_pos > pos:
            if next_pos - pos > 1:
                intervals.append((timestamps[pos], timestamps[next_pos-1]))
            else:
                intervals.append((timestamps[pos], timestamps[pos]))
        remaining -= 1
        pos = next_pos
    
    return intervals