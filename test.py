def find_minimal_intervals(timestamps, k):
    """
    Find k intervals that cover all timestamps with minimum total length.
    
    Args:
        timestamps (list): Sorted list of timestamps to cover
        k (int): Number of intervals to use
        
    Returns:
        list: List of tuples (start, end) representing the optimal intervals
        float: Total length of all intervals
    """
    if k <= 0 or not timestamps:
        return [], 0
    
    if k >= len(timestamps):
        # If we have more intervals than points, we can use zero-length intervals
        return [(t, t) for t in timestamps], 0
    
    n = len(timestamps)
    # dp[i][j] represents the minimum length needed to cover timestamps[i:] with j intervals
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
    
    return intervals, dp[0][k]

def print_solution(timestamps, k):
    """
    Print the solution in a readable format.
    
    Args:
        timestamps (list): List of timestamps
        k (int): Number of intervals
    """
    intervals, total_length = find_minimal_intervals(timestamps, k)
    print(f"\nOptimal solution with {k} intervals:")
    print(f"Total length: {total_length}")
    print("Intervals:")
    for i, (start, end) in enumerate(intervals, 1):
        print(f"Interval {i}: [{start}, {end}] (length: {end-start})")
    print("\nTimestamps covered by each interval:")
    for i, (start, end) in enumerate(intervals, 1):
        covered = [t for t in timestamps if start <= t <= end]
        print(f"Interval {i}: {covered}")

# Your timestamps
t = [0.1, 0.3, 1.2, 1.4, 1.8, 3, 3.9, 4, 5, 6, 8, 10]
k = 3  # Number of intervals

print_solution(t, k)