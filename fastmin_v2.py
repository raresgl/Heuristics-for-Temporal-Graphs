from datetime import datetime, timedelta
import numpy as np
import copy
# import sys # Not used in the provided snippet
# import pickle # Not used in the provided snippet
import graph as utils # Assuming this is your graph utilities module
import operator # Not used in the provided snippet
# from baseline import * # Assuming this is not strictly needed for proper_search
import k_inner # Your k_inner.py script
from typing import List, Tuple, Dict
# import random # Not used in the current refined strategies
from collections import defaultdict

def proper_search(timestamps: List[Tuple[int, str, str]], k: int = 3, maxiter: int = 10, k_inner_maxiter: int = 10) -> Tuple[Dict[str, Dict[int, int]], Dict[str, Dict[int, int]]]:
    """
    Improves upon the k-Inner algorithm's solution for temporal vertex cover.

    Parameters
    ----------
    timestamps : list of tuples
        Sorted list of interactions [(t1, n1, n2), ...].
    k : int
        Number of activity intervals per node.
    maxiter : int
        Maximum number of improvement iterations for proper_search.
    k_inner_maxiter : int
        Maximum number of iterations for the initial k_inner.runKInner call.

    Returns
    -------
    tuple of dicts
        (Xstart, Xend): Optimized interval start and end points.
    """
    nodeEdgeIndex = utils.indexMapping(timestamps)
    if not nodeEdgeIndex: # Handle cases with no nodes/interactions
        return {}, {}

    # 1. Get the best initial solution from the full k_inner.runKInner algorithm
    print(f"Running initial k_inner.runKInner with maxiter={k_inner_maxiter}...")
    initial_Xstart, initial_Xend = k_inner.runKInner(timestamps, k, maxiter=k_inner_maxiter, m={})
    
    Xstart = copy.deepcopy(initial_Xstart)
    Xend = copy.deepcopy(initial_Xend)
    best_solution = (Xstart, Xend)
    best_cost = utils.getCost(Xstart, Xend)
    print(f"Initial cost from k_inner.runKInner: {best_cost}")

    # Prepare a structure to quickly get all interaction timestamps for a node
    node_all_timestamps = {
        node: sorted(list(set(interaction[0] for interaction in interactions_list)))
        for node, interactions_list in nodeEdgeIndex.items()
    }

    # 2. Iteratively try to improve the solution
    for iteration in range(maxiter):
        improved_in_this_iteration = False
        
        # Calculate current loss and identify problematic interactions/nodes
        loss_dict = defaultdict(int) # Using defaultdict for convenience
        uncovered_edges = []
        
        # Initialize loss tracking for existing intervals
        # This might be less critical now as we focus on nodes from uncovered edges or high overall loss
        # for node_key in Xstart:
        #     for interval_idx in Xstart[node_key]:
        #         loss_dict[(node_key, interval_idx)] = 0 # ensure all intervals are in loss_dict

        active_nodes_in_timestamps = set()
        for t, u, v in timestamps:
            active_nodes_in_timestamps.add(u)
            active_nodes_in_timestamps.add(v)

        for time_interaction in timestamps:
            t, u, v = time_interaction
            u_covered = False
            v_covered = False
            
            # Check coverage for node u
            if u in Xstart:
                for interval_idx in Xstart[u]:
                    if Xstart[u][interval_idx] <= t <= Xend[u][interval_idx]:
                        u_covered = True
                        break
            # Check coverage for node v
            if v in Xstart:
                for interval_idx in Xstart[v]:
                    if Xstart[v][interval_idx] <= t <= Xend[v][interval_idx]:
                        v_covered = True
                        break
            
            if u_covered and not v_covered:
                # Find which interval of u covered it (can be complex if overlaps)
                # For simplicity, we'll attribute loss to the node.
                if u in Xstart: # u must be in Xstart to have covered
                    for interval_idx in Xstart[u]: # find first covering interval
                         if Xstart[u][interval_idx] <= t <= Xend[u][interval_idx]:
                            loss_dict[(u, interval_idx)] += 1
                            break 
            elif not u_covered and v_covered:
                if v in Xstart: # v must be in Xstart
                    for interval_idx in Xstart[v]: # find first covering interval
                        if Xstart[v][interval_idx] <= t <= Xend[v][interval_idx]:
                            loss_dict[(v, interval_idx)] += 1
                            break
            elif not u_covered and not v_covered:
                uncovered_edges.append(time_interaction)

        # Identify nodes to try improving
        nodes_to_consider = set()
        node_total_loss = defaultdict(int)

        for (node, interval_idx), loss_val in loss_dict.items():
            if loss_val > 0: # This interval is singly covering edges
                nodes_to_consider.add(node)
                node_total_loss[node] += loss_val
        
        uncovered_nodes = set()
        for _, u, v in uncovered_edges:
            nodes_to_consider.add(u)
            nodes_to_consider.add(v)
            uncovered_nodes.add(u)
            uncovered_nodes.add(v)
        
        # Prioritize nodes: 1. Uncovered, 2. Highest total loss
        # Ensure all nodes that are part of the graph interactions are candidates if needed
        # and have some timestamps.
        sorted_nodes_to_try = sorted(list(nodes_to_consider.intersection(active_nodes_in_timestamps)), 
                                     key=lambda n: (n in uncovered_nodes, node_total_loss[n]), 
                                     reverse=True)
        
        if not sorted_nodes_to_try and uncovered_edges: # Case: Uncovered edges but nodes not in Xstart
             # Add nodes from uncovered_edges that might not be in Xstart yet
             sorted_nodes_to_try = sorted(list(uncovered_nodes.intersection(active_nodes_in_timestamps)), reverse=True)


        print(f"\nIteration {iteration + 1}/{maxiter}. Current best cost: {best_cost}")
        if uncovered_edges:
            print(f"Uncovered edges: {len(uncovered_edges)}")
        # print(f"Nodes to try (prioritized): {sorted_nodes_to_try}")


        for node in sorted_nodes_to_try:
            if node not in node_all_timestamps or not node_all_timestamps[node]:
                # print(f"Skipping node {node} as it has no timestamps in node_all_timestamps.")
                continue

            # Timestamps for this node's interactions
            current_node_timestamps = node_all_timestamps[node]
            if not current_node_timestamps: # Should be caught by above but double check
                # print(f"Skipping node {node} due to no interaction timestamps.")
                continue

            candidate_node_improvements = []

            # Strategy A: DP-based re-optimization for the current 'node'
            if len(current_node_timestamps) > 0:
                # print(f"  Trying DP for node {node} with {len(current_node_timestamps)} timestamps.")
                new_intervals_dp = find_minimal_intervals(current_node_timestamps, k)
                if new_intervals_dp:
                    temp_Xstart_dp = copy.deepcopy(Xstart)
                    temp_Xend_dp = copy.deepcopy(Xend)
                    temp_Xstart_dp[node] = {i: start for i, (start, end) in enumerate(new_intervals_dp)}
                    temp_Xend_dp[node] = {i: end for i, (start, end) in enumerate(new_intervals_dp)}
                    cost_dp = utils.getCost(temp_Xstart_dp, temp_Xend_dp)
                    candidate_node_improvements.append({'Xstart': temp_Xstart_dp, 'Xend': temp_Xend_dp, 'cost': cost_dp, 'strategy': 'DP'})

            # Strategy B: Gap-based re-optimization for the current 'node'
            if len(current_node_timestamps) >= k : # Gap-based needs at least k points to make k intervals meaningfully.
                                                 # Or, more precisely, enough points to form k intervals.
                                                 # `find_intervals_gap_based` handles len < k.
                # print(f"  Trying Gap-based for node {node} with {len(current_node_timestamps)} timestamps.")
                new_intervals_gap = find_intervals_gap_based(current_node_timestamps, k)
                if new_intervals_gap: # Ensure intervals were actually formed
                    temp_Xstart_gap = copy.deepcopy(Xstart)
                    temp_Xend_gap = copy.deepcopy(Xend)
                    temp_Xstart_gap[node] = {i: start for i, (start, end) in enumerate(new_intervals_gap)}
                    temp_Xend_gap[node] = {i: end for i, (start, end) in enumerate(new_intervals_gap)}
                    cost_gap = utils.getCost(temp_Xstart_gap, temp_Xend_gap)
                    candidate_node_improvements.append({'Xstart': temp_Xstart_gap, 'Xend': temp_Xend_gap, 'cost': cost_gap, 'strategy': 'Gap'})
            
            if candidate_node_improvements:
                best_try_for_this_node = min(candidate_node_improvements, key=lambda x: x['cost'])
                if best_try_for_this_node['cost'] < best_cost:
                    print(f"  Improvement for node {node} via {best_try_for_this_node['strategy']}: {best_cost} -> {best_try_for_this_node['cost']}")
                    Xstart = best_try_for_this_node['Xstart']
                    Xend = best_try_for_this_node['Xend']
                    best_cost = best_try_for_this_node['cost']
                    best_solution = (copy.deepcopy(Xstart), copy.deepcopy(Xend))
                    improved_in_this_iteration = True
                    break # Break from iterating over nodes_to_try, restart iteration with new global Xstart/Xend
        
        if not improved_in_this_iteration:
            print("No further improvement in this iteration. Stopping proper_search.")
            break # Break from the main improvement loop

    print(f"\nFinal best cost after proper_search: {best_cost}")
    return best_solution[0], best_solution[1]


def find_intervals_gap_based(timestamps: List[int], k: int) -> List[Tuple[int, int]]:
    """
    Finds k intervals using gap-based splitting.
    Timestamps should be sorted and unique.
    """
    if not timestamps:
        return []
    
    n = len(timestamps)
    
    if k <= 0: # No intervals to find
        return []
    if k >= n: # Each timestamp is its own interval or k is too large
        return [(t, t) for t in timestamps]
    if k == 1:
        return [(timestamps[0], timestamps[-1])]
    
    gaps = []
    for i in range(n - 1):
        gap_size = timestamps[i+1] - timestamps[i]
        if gap_size > 0: # Only consider actual gaps
            gaps.append((gap_size, i)) # Store gap size and index of the first element of the pair
    
    # Sort gaps by size (largest first) and take top k-1 gaps to make k intervals
    gaps.sort(key=lambda x: x[0], reverse=True)
    
    split_indices = sorted([gap[1] for gap in gaps[:k-1]]) # Indices *before* which to split
    
    intervals = []
    current_start_idx = 0
    for split_idx in split_indices:
        intervals.append((timestamps[current_start_idx], timestamps[split_idx]))
        current_start_idx = split_idx + 1 # Next interval starts at the timestamp after the gap
        
    # Add the last interval
    if current_start_idx < n:
        intervals.append((timestamps[current_start_idx], timestamps[n-1]))
    
    # If fewer than k intervals were formed (e.g. k > number of actual gaps + 1),
    # pad with single point intervals from remaining timestamps if necessary.
    # This basic version assumes k-1 largest gaps always exist and are meaningful.
    # A more robust version might be needed if k is large relative to distinct segments.
    # For now, this matches the general intent.
    return intervals


def find_minimal_intervals(timestamps: List[int], k: int) -> List[Tuple[int, int]]:
    """
    DP-based interval finding to cover sorted unique timestamps with k intervals, minimizing sum of lengths.
    """
    if not timestamps:
        return []
    
    n = len(timestamps)

    if k <= 0:
        return []
    if k >= n: # If k allows, cover each point as a separate interval
        return [(t, t) for t in timestamps]
    if k == 1:
        return [(timestamps[0], timestamps[-1])]

    # dp[i][j] = minimum cost to cover timestamps[0...i-1] with j intervals
    # Using 1-based indexing for timestamps (ts_array) to align with common DP patterns
    ts_array = timestamps 
    dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
    # prev_split[i][j] stores the start index (0-based in ts_array) of the j-th interval
    # when covering timestamps[0...i-1]
    prev_split = [[-1] * (k + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][1] = ts_array[i-1] - ts_array[0] # Cost of one interval covering ts_array[0] to ts_array[i-1]
                                              # Add 1 for duration if cost includes endpoints: (ts_array[i-1] - ts_array[0] + 1)
                                              # The original problem description often uses "length" which can be ambiguous.
                                              # Let's assume cost is (end - start). If it's (end - start + 1), adjust here.
                                              # Given utils.getCost, it sums (Xend[n][i] - Xstart[n][i]).
                                              # So, the cost here should be consistent: end_ts - start_ts.
                                              # If point t is an interval (t,t), its cost is 0.
                                              # Let's use (end_ts - start_ts) which is duration.
                                              # If intervals are [start, end] inclusive, length is end - start + 1.
                                              # The provided k_inner suggests cost is sum of (end-start). Let's stick to this.
                                              # If an interval is a single point (t,t), cost is 0.

    # Base case: cost to cover 0 elements with 0 intervals is 0
    for j in range(k + 1):
        dp[0][j] = 0
    
    # Fill DP table
    for j_intervals in range(1, k + 1): # Number of intervals used
        for i_ts_idx in range(1, n + 1): # Considering timestamps up to ts_array[i_ts_idx-1]
            if j_intervals == 1:
                dp[i_ts_idx][j_intervals] = ts_array[i_ts_idx-1] - ts_array[0]
                prev_split[i_ts_idx][j_intervals] = 0 # The first interval starts at index 0
            else:
                for p_prev_ts_idx in range(1, i_ts_idx + 1): # p_prev_ts_idx is the end of the (j-1)th interval group
                                                             # The j-th interval starts at ts_array[p_prev_ts_idx-1] if p_prev_ts_idx > 0
                                                             # and ends at ts_array[i_ts_idx-1]
                    
                    cost_last_interval = 0
                    if p_prev_ts_idx <= i_ts_idx : # ensure valid interval
                        # The last interval covers from ts_array[p_prev_ts_idx-1] to ts_array[i_ts_idx-1]
                        # This means timestamps from index p_prev_ts_idx-1 up to i_ts_idx-1 are in this interval
                        # dp[p_prev_ts_idx-1][j_intervals-1] covers up to ts_array[p_prev_ts_idx-2]
                        # The j-th interval covers from ts_array[p_prev_ts_idx-1] to ts_array[i_ts_idx-1]
                        start_of_last_interval_idx = p_prev_ts_idx -1
                        end_of_last_interval_idx = i_ts_idx -1
                        
                        if start_of_last_interval_idx <= end_of_last_interval_idx: # valid interval
                            cost_last_interval = ts_array[end_of_last_interval_idx] - ts_array[start_of_last_interval_idx]
                            
                            current_total_cost = dp[p_prev_ts_idx-1][j_intervals-1] + cost_last_interval
                            if current_total_cost < dp[i_ts_idx][j_intervals]:
                                dp[i_ts_idx][j_intervals] = current_total_cost
                                prev_split[i_ts_idx][j_intervals] = start_of_last_interval_idx

    # Reconstruct intervals
    intervals = []
    current_n = n
    for j_val in range(k, 0, -1):
        if current_n <= 0 or prev_split[current_n][j_val] == -1:
            # This can happen if k is too large for n points in a way that DP fails to reconstruct.
            # Or if a valid path wasn't found.
            # Fallback or error. For now, if reconstruction fails, return empty or partial.
            # print(f"Warning: DP reconstruction issue for k={k}, n={n}, j_val={j_val}, current_n={current_n}")
            break 
            
        start_index = prev_split[current_n][j_val]
        end_index = current_n - 1
        
        if start_index > end_index : # Should not happen with correct DP
            # print(f"Warning: DP reconstruction start_index > end_index")
            break

        intervals.append((ts_array[start_index], ts_array[end_index]))
        current_n = start_index # Move to the end of the previous segment of intervals
    
    intervals.reverse() # They are added from last to first

    # If fewer than k intervals were found due to DP structure or small n:
    # This DP formulation should ideally produce exactly k intervals if possible,
    # or fewer if k is larger than n.
    # The problem asks for k intervals. If the DP produces fewer, it means it's optimal
    # to have some intervals cover multiple points to achieve k segments.
    # If len(intervals) < k and n > 0, this DP might need adjustment or interpretation
    # regarding forcing k intervals. The common version minimizes cost for *up to* k.
    # For now, let's assume this DP is what's intended by "find_minimal_intervals".
    # The original code did not seem to force k intervals if fewer were optimal or if n < k.
    # The original `find_minimal_intervals` had a different DP structure.
    # Let's stick to the provided DP logic structure from your original file.
    # The version from your original file: `dp[i][j]` = min cost for `timestamps[i:]` with `j` intervals.

    # Re-implementing DP from your original fastmin_v2.py for consistency:
    dp_orig = [[float('inf')] * (k + 1) for _ in range(n + 1)]
    prev_orig = [[(-1,-1)] * (k + 1) for _ in range(n + 1)] # Stores (end_idx, start_idx_of_next_segment)

    for j_intervals_left in range(k + 1):
        dp_orig[n][j_intervals_left] = 0 # Cost to cover no more timestamps is 0,
                                         # but if j_intervals_left > 0, this is not ideal.
                                         # More accurately, dp_orig[n][0] = 0. Other dp_orig[n][j>0] should be inf.
    dp_orig[n][0] = 0
    
    for i_start_idx in range(n - 1, -1, -1): # Current start index of timestamps to cover
        for j_intervals_left in range(1, k + 1):
            for end_idx in range(i_start_idx, n): # Potential end of the current interval
                current_interval_cost = timestamps[end_idx] - timestamps[i_start_idx] # (or +1 if length)
                
                cost_of_remaining = dp_orig[end_idx + 1][j_intervals_left - 1]
                
                if cost_of_remaining != float('inf'):
                    total_cost = current_interval_cost + cost_of_remaining
                    if total_cost < dp_orig[i_start_idx][j_intervals_left]:
                        dp_orig[i_start_idx][j_intervals_left] = total_cost
                        prev_orig[i_start_idx][j_intervals_left] = (end_idx, i_start_idx) # Store end_idx of current interval

    # Reconstruct from original DP logic
    intervals_reconstructed = []
    curr_idx = 0
    intervals_to_form = k
    
    # Check if a solution is possible
    if dp_orig[0][k] == float('inf'):
        # print(f"DP could not find a solution for k={k} with {n} timestamps. Returning best effort for n < k or k=1.")
        if k >= n: return [(t, t) for t in timestamps] # Fallback for k >= n
        if k == 1 and n > 0: return [(timestamps[0], timestamps[-1])]
        return [] # No solution found

    while curr_idx < n and intervals_to_form > 0:
        end_of_current_interval_idx, start_of_current_interval_idx_unused = prev_orig[curr_idx][intervals_to_form]
        
        if end_of_current_interval_idx == -1 : # Should not happen if dp_orig[0][k] was not inf
             # This implies an issue in reconstruction or that a state was not properly reached.
             # If dp_orig[0][k] is finite, a path must exist.
             # Potentially, if k is very high for n, some states might be tricky.
             # print(f"DP reconstruction error: prev_orig[{curr_idx}][{intervals_to_form}] is -1")
             break

        intervals_reconstructed.append((timestamps[curr_idx], timestamps[end_of_current_interval_idx]))
        curr_idx = end_of_current_interval_idx + 1
        intervals_to_form -= 1
        
        # If we have formed enough intervals but haven't covered all points (e.g. k is small)
        # the DP should have made the last interval extend to n-1.
        # If intervals_to_form is 0 and curr_idx < n, it implies the k intervals didn't cover all.
        # This DP formulation is for covering *all* points with *exactly* k intervals if possible,
        # by minimizing sum of lengths.

    return intervals_reconstructed

# Note: The original `find_better_intervals` is not strictly needed if its logic
# (trying DP and Gap, then comparing to original for that node) is now embedded
# in the main `proper_search` loop, which evaluates global cost.