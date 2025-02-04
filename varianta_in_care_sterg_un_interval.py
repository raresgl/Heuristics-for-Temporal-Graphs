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

def local_search(timestamps: List[Tuple[int, str, str]], k: int = 3, deletions: int = 1) -> Dict[str, int]:
    """
    Calculate the loss for each node, defined as the number of uncovered nodes at each time.
    
    Args:
        Xstart: Dictionary mapping nodes to their start times
        Xend: Dictionary mapping nodes to their end times
        timestamps: List of (time, node1, node2) events
        
    Returns:
        Dictionary mapping nodes to their loss values
    """
    Xstart, Xend = kbaseline(timestamps, k)
    print(Xstart)
    loss_dict = {node: {} for node in Xstart.keys()}
    remaining = []
    
    for node in Xstart.keys():
        loss_dict[node] = {}
        for i in range(len(Xstart[node])):
            uncovered_count = 0
            for time in timestamps:
                if Xstart[node][i] <= time[0] <= Xend[node][i] and (time[1] == node or time[2] == node):
                    uncovered_count += 1
            loss_dict[node][i] = uncovered_count

    for _ in range(deletions):
        min_loss = float('inf')
        min_loss_node = None
        min_interval = None
        for node in loss_dict:
            for interval, loss in loss_dict[node].items():
                if loss < min_loss:
                    min_loss = loss
                    min_loss_node = node
                    min_interval = interval

        remaining_nodes = list(Xstart.keys())
        if min_loss_node in remaining_nodes:
            Xstart[min_loss_node] = {k:v for k,v in Xstart[min_loss_node].items() if k != min_interval}
            Xend[min_loss_node] = {k:v for k,v in Xend[min_loss_node].items() if k != min_interval}
            if not Xstart[min_loss_node]:  # If no intervals left
                remaining_nodes.remove(min_loss_node)
            loss_dict[min_loss_node] = {k:v for k,v in loss_dict[min_loss_node].items() if k != min_interval}
        # Select another random node to eject
        
        if remaining_nodes:
            random_node = random.choice(remaining_nodes)
            random_interval = random.choice(list(Xstart[random_node].keys()))
            Xstart[random_node] = {k:v for k,v in Xstart[random_node].items() if k != random_interval}
            Xend[random_node] = {k:v for k,v in Xend[random_node].items() if k != random_interval}
            if not Xstart[random_node]:  # If no intervals left
                remaining_nodes.remove(random_node)
            
            # Add back events for ejected nodes
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
    # Re-create solution for remaining events if any exist
    while remaining:
        # Create index mapping for remaining events
        nodeEdgeIndex = {}
        for t in remaining:
            if t[1] not in nodeEdgeIndex:
                nodeEdgeIndex[t[1]] = []
            if t[2] not in nodeEdgeIndex:
                nodeEdgeIndex[t[2]] = []
            nodeEdgeIndex[t[1]].append(t)
            nodeEdgeIndex[t[2]].append(t)
        
        # Calculate gain for remaining nodes that aren't already in solution
        gain = {}
        for node, events in nodeEdgeIndex.items():
            if node not in Xstart:  # Only consider nodes not already in solution
                total_gap = get_len_wo_kgaps(nodeEdgeIndex, node, k)
                time_span = events[-1][0] - events[0][0]
                if time_span - total_gap > 0:
                    gain[node] = float(len(events))/(time_span - total_gap)
                else:
                    gain[node] = float(len(events))
        if gain:          
            node = max(gain.items(), key=operator.itemgetter(1))[0]
            
            # Find optimal k intervals
            s = [remaining[0][0]] + [i[0] for i in nodeEdgeIndex[node]] + [remaining[-1][0]]
            gaps = np.argsort([(s[i+1] - s[i]) for i in range(len(s)-1)])[::-1]
            starts = [s[0]] + [s[i+1] for i in sorted(gaps[:min((k-1),len(gaps))])]
            ends = [s[i] for i in sorted(gaps[:min((k-1),len(gaps))])] + [s[-1]]
            
            # Store intervals for selected node
            # If node already exists, we need to be careful not to exceed k total intervals
            current_intervals = len(Xstart[node]) if node in Xstart else 0
            remaining_slots = k - current_intervals
            
            # Only add up to remaining_slots number of new intervals
            for i in range(min(len(starts), remaining_slots)):
                next_idx = current_intervals + i
                if node not in Xstart:
                    # Xstart[node] = {}
                    # Xend[node] = {}
                    Xstart[node] = {i: starts[i] if i<len(starts) else starts[-1] for i in range(len(starts))}
                    Xend[node] = {i: ends[i] if i<len(ends) else ends[-1] for i in range(len(ends))}
                else:
                    Xstart[node][next_idx] = starts[i]
                    Xend[node][next_idx] = ends[i]
                #print(len(Xstart[node]))
        remaining = [x for x in remaining if x[1] != node and x[2] != node]

    return Xstart, Xend
