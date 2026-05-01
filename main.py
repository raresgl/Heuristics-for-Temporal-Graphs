import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from datetime import datetime, timedelta
import numpy as np
import sys
import networkx as nx
import time 
import graph as utils
import baseline
import fastmin_v2
from typing import List, Tuple, Dict
import ilp_version as ilp
import k_budget
import k_inner

def run_experiment(alg, k, event_length, overlap, num_nodes, num_iterations=10):
    """
    Run algorithms for multiple iterations and collect performance metrics
    """
    results = []
    
    for iteration in tqdm(range(num_iterations), desc=f"Running {alg}"):
        # Generate a new graph for each iteration
        G = utils.generateGraph(n=num_nodes)
        
        # Generate intervals for this graph
        timestamps, active_truth = utils.generateIntervals(G, event_length=event_length, 
                                                           overlap=overlap, number_intervals=k)
        
        # Run the specified algorithm
        if alg == 'baseline':
            Xstart, Xend = baseline.kbaseline(timestamps, k)
            
            # Calculate metrics
            relative_cost = utils.getCost(Xstart, Xend)/((event_length-1)*num_nodes)
            total_length = utils.getCost(Xstart, Xend)
            true_length = utils.calculate_length_for_active_truths(active_truth)
            p, r, f = utils.compareGT(Xstart, Xend, active_truth, timestamps)
            
            # Add ILP solution if requested
            if args.use_ilp:
                start_time = time.time()
                m, x = ilp.ilp(timestamps, G, k)
                m.optimize()
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_activations = m.objVal
                active_intervals = ilp.active_intervals(m, x)
                p_ilp, r_ilp, f_ilp = utils.compare_intervals(active_intervals, active_truth, timestamps)
            else:
                p_ilp, r_ilp, f_ilp = 0, 0, 0
                elapsed_time = 0
            
        elif alg == 'heuristic1':
            # Run the heuristic1 algorithm (replace with the actual function)
            Xstart, Xend = fastmin_v2.proper_search(timestamps, k) 
            
            # Calculate metrics (same as above)
            relative_cost = utils.getCost(Xstart, Xend)/((event_length-1)*num_nodes)
            total_length = utils.getCost(Xstart, Xend)
            true_length = utils.calculate_length_for_active_truths(active_truth)
            p, r, f = utils.compareGT(Xstart, Xend, active_truth, timestamps)
            
            # Add ILP solution if requested
            if args.use_ilp:
                start_time = time.time()
                m, x = ilp.ilp(timestamps, G, k)
                m.optimize()
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_activations = m.objVal
                active_intervals = ilp.active_intervals(m, x)
                p_ilp, r_ilp, f_ilp = utils.compare_intervals(active_intervals, active_truth, timestamps)
            else:
                p_ilp, r_ilp, f_ilp = 0, 0, 0
                elapsed_time = 0
        elif alg == 'budget':
            # Run the budget algorithm (replace with the actual function)
            Xstart, Xend = k_budget.runKBudget(timestamps, k) 
            # Debugging: Print Xstart and Xend
            # print(f"Xstart (budget): {Xstart}")
            # print(f"Xend (budget): {Xend}")
            
            # Calculate metrics
            total_length = utils.getCost(Xstart, Xend)
            # print(f"Total length (budget): {total_length}")  # Debugging
            # Calculate metrics (same as above)
            relative_cost = utils.getCost(Xstart, Xend)/((event_length-1)*num_nodes)
            total_length = utils.getCost(Xstart, Xend)
            true_length = utils.calculate_length_for_active_truths(active_truth)
            p, r, f = utils.compareGT(Xstart, Xend, active_truth, timestamps)
            
            # Add ILP solution if requested
            if args.use_ilp:
                start_time = time.time()
                m, x = ilp.ilp(timestamps, G, k)
                m.optimize()
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_activations = m.objVal
                active_intervals = ilp.active_intervals(m, x)
                p_ilp, r_ilp, f_ilp = utils.compare_intervals(active_intervals, active_truth, timestamps)
            else:
                p_ilp, r_ilp, f_ilp = 0, 0, 0
                elapsed_time = 0
        elif alg == 'inner':
            # Run the inner algorithm (replace with the actual function)
            Xstart, Xend = k_inner.runKInner(timestamps, k) 
            # Debugging: Print Xstart and Xend
            # print(f"Xstart (budget): {Xstart}")
            # print(f"Xend (budget): {Xend}")
            
            # Calculate metrics
            total_length = utils.getCost(Xstart, Xend)
            # print(f"Total length (budget): {total_length}")  # Debugging
            # Calculate metrics (same as above)
            relative_cost = utils.getCost(Xstart, Xend)/((event_length-1)*num_nodes)
            total_length = utils.getCost(Xstart, Xend)
            true_length = utils.calculate_length_for_active_truths(active_truth)
            p, r, f = utils.compareGT(Xstart, Xend, active_truth, timestamps)
            
            # Add ILP solution if requested
            if args.use_ilp:
                start_time = time.time()
                m, x = ilp.ilp(timestamps, G, k)
                m.optimize()
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_activations = m.objVal
                active_intervals = ilp.active_intervals(m, x)
                p_ilp, r_ilp, f_ilp = utils.compare_intervals(active_intervals, active_truth, timestamps)
            else:
                p_ilp, r_ilp, f_ilp = 0, 0, 0
                elapsed_time = 0
        
        # Store results for this iteration
        results.append({
            'iteration': iteration,
            'algorithm': alg,
            'k': k,
            'event_length': event_length,
            'overlap': overlap,
            'num_nodes': num_nodes,
            'num_timestamps': len(timestamps),
            'relative_cost': relative_cost,
            'total_length': total_length,
            'true_length': true_length,
            'precision': p,
            'recall': r,
            'f_measure': f,
            'precision_ilp': p_ilp,
            'recall_ilp': r_ilp,
            'f_measure_ilp': f_ilp,
            'ilp_time': elapsed_time
        })
    
    return pd.DataFrame(results)

def parameter_sweep(algorithm, param_name, param_values, fixed_params, num_iterations=5):
    """
    Run experiments for different parameter values while keeping others fixed
    """
    all_results = []
    
    # Map internal parameter names to column names
    param_column_mapping = {
        'k': 'k',
        'event_length': 'event_length',
        'overlap': 'overlap',
        'nnodes': 'num_nodes',
        'intlen': 'event_length'
    }
    
    # Column to use for plotting
    display_param = param_column_mapping.get(param_name, param_name)
    
    for value in tqdm(param_values, desc=f"Sweeping {param_name}"):
        # Create a copy of fixed parameters and update the one we're sweeping
        params = fixed_params.copy()
        
        if param_name == 'nnodes':
            params['num_nodes'] = value
        elif param_name == 'intlen':
            params['event_length'] = value
        else:
            params[param_name] = value
        
        print(f"\nRunning with {param_name}={value}")
        
        # Run experiment with these parameters
        results = run_experiment(
            algorithm, 
            k=params['k'], 
            event_length=params['event_length'], 
            overlap=params['overlap'], 
            num_nodes=params['num_nodes'],
            num_iterations=num_iterations
        )
        
        # Add the sweep parameter value explicitly
        if param_name == 'nnodes':
            results['nnodes'] = value
        elif param_name == 'intlen':
            results['intlen'] = value
            
        all_results.append(results)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Print summary of results for each parameter value
    print("\nSummary of results for each parameter value:")
    for value in param_values:
        if param_name == 'nnodes':
            subset = combined_results[combined_results['num_nodes'] == value]
        elif param_name == 'intlen':
            subset = combined_results[combined_results['event_length'] == value]
        else:
            subset = combined_results[combined_results[param_name] == value]
            
        print(f"\n{param_name} = {value}:")
        for metric in ['precision', 'recall', 'f_measure']:
            mean = subset[metric].mean()
            std = subset[metric].std()
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")
    
    return combined_results

def visualize_results(results, param_name=None):
    """
    Create visualizations based on the collected results
    """
    # Set up the figure
    plt.figure(figsize=(15, 10))
    
    # Create subplots for different metrics
    metrics = ['precision', 'recall', 'f_measure', 'total_length']
    
    # Map command-line parameter names to column names in the DataFrame
    param_column_mapping = {
        'k': 'k',
        'intlen': 'event_length',
        'overlap': 'overlap',
        'nnodes': 'num_nodes'
    }
    
    # Get the correct column name for the sweep parameter
    plot_param = param_column_mapping.get(param_name, param_name)
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        if param_name:
            # Make sure the parameter column exists
            if plot_param not in results.columns:
                if param_name == 'nnodes' and 'num_nodes' in results.columns:
                    # Use num_nodes column for nnodes parameter
                    plot_param = 'num_nodes'
                else:
                    print(f"Warning: Column '{plot_param}' not found in results. Available columns: {results.columns.tolist()}")
                    continue
            
            # For parameter sweep, aggregate the data by the parameter value
            summary = results.groupby(plot_param)[metric].agg(['mean', 'std']).reset_index()
            
            # Plot the mean values
            plt.plot(summary[plot_param], summary['mean'], marker='o', linestyle='-', linewidth=2)
            
            # Add error bars for standard deviation
            plt.fill_between(
                summary[plot_param], 
                summary['mean'] - summary['std'],
                summary['mean'] + summary['std'], 
                alpha=0.3
            )
            
            plt.xlabel(param_name)
            plt.ylabel(metric)
            plt.title(f'{metric.capitalize()} vs {param_name}')
        else:
            # Otherwise show distributions across iterations
            sns.boxplot(data=results, x='algorithm', y=metric)
            plt.title(f'Distribution of {metric.capitalize()}')
        
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'performance_metrics_{time.strftime("%Y%m%d_%H%M%S")}.png')
    print(f"Saved performance metrics plot to performance_metrics_{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    # Create a comparison plot if ILP results are available
    if 'precision_ilp' in results.columns and results['precision_ilp'].sum() > 0:
        plt.figure(figsize=(15, 5))
        
        metrics = ['precision', 'recall', 'f_measure']
        for i, metric in enumerate(metrics):
            plt.subplot(1, 3, i+1)
            
            # Prepare data for comparison
            comparison_data = pd.melt(
                results[['algorithm', metric, f'{metric}_ilp']].rename(
                    columns={metric: 'Heuristic', f'{metric}_ilp': 'ILP'}
                ),
                id_vars=['algorithm'],
                var_name='Method',
                value_name=metric.capitalize()
            )
            
            sns.boxplot(data=comparison_data, x='algorithm', y=metric.capitalize(), hue='Method')
            plt.title(f'Comparison of {metric.capitalize()}')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'ilp_comparison_{time.strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()
    
    # Create a comparison plot if ILP results are available
    if 'precision_ilp' in results.columns and results['precision_ilp'].sum() > 0:
        plt.figure(figsize=(15, 5))
        
        metrics = ['precision', 'recall', 'f_measure']
        for i, metric in enumerate(metrics):
            plt.subplot(1, 3, i+1)
            
            # Prepare data for comparison
            comparison_data = pd.melt(
                results[['algorithm', metric, f'{metric}_ilp']].rename(
                    columns={metric: 'Heuristic', f'{metric}_ilp': 'ILP'}
                ),
                id_vars=['algorithm'],
                var_name='Method',
                value_name=metric.capitalize()
            )
            
            sns.boxplot(data=comparison_data, x='algorithm', y=metric.capitalize(), hue='Method')
            plt.title(f'Comparison of {metric.capitalize()}')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'ilp_comparison_{time.strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()

def save_results(results, filename=None):
    """
    Save results to a CSV file
    """
    if filename is None:
        filename = f'experiment_results_{time.strftime("%Y%m%d_%H%M%S")}.csv'
    
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    # Also append summary statistics to the output file
    with open("out.txt", "a") as f:
        f.write(f"\n===== Experiment Summary {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Number of iterations: {args.iterations}\n")
        f.write(f"Parameters: k={args.k}, event_length={args.intlen}, overlap={args.overlap}, num_nodes={args.nnodes}\n\n")
        
        # Write average metrics
        f.write("Average metrics:\n")
        for metric in ['precision', 'recall', 'f_measure', 'total_length']:
            mean = results[metric].mean()
            std = results[metric].std()
            f.write(f"{metric}: {mean:.4f} ± {std:.4f}\n")
        
        if 'precision_ilp' in results.columns and results['precision_ilp'].sum() > 0:
            f.write("\nILP comparison:\n")
            for metric in ['precision', 'recall', 'f_measure']:
                heuristic_mean = results[metric].mean()
                ilp_mean = results[f'{metric}_ilp'].mean()
                f.write(f"{metric}: Heuristic = {heuristic_mean:.4f}, ILP = {ilp_mean:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("algorithm", choices=["baseline", "heuristic1", "budget", "inner"], help="Algorithm to run")
    parser.add_argument("-k", help="number of intervals", default=10, type=int)
    parser.add_argument("--intlen", help="length of each active interval", default=10, type=int)
    parser.add_argument("--overlap", help="activity intervals overlap parameter, between 0 and 1", default=0.5, type=float)
    parser.add_argument("--nnodes", help="number of nodes in the graph", default=600, type=int)
    parser.add_argument("--iterations", help="number of iterations to run", default=10, type=int)
    parser.add_argument("--use-ilp", help="also run ILP for comparison", action="store_true")
    parser.add_argument("--sweep", help="parameter to sweep (k, intlen, overlap, nnodes)", default=None)
    parser.add_argument("--sweep-values", help="comma-separated values for the sweep parameter", default=None)
    parser.add_argument("--output", help="output file name for results", default=None)
    args = parser.parse_args()

    # Print experiment configuration
    print('=' * 60)
    print(f'Running {args.algorithm} for {args.iterations} iterations')
    print(f'Parameters: k={args.k}, event_length={args.intlen}, overlap={args.overlap}, num_nodes={args.nnodes}')
    print('=' * 60)
    
    # Check if we're doing a parameter sweep
    if args.sweep:
        if not args.sweep_values:
            print("Error: --sweep-values must be provided when using --sweep")
            sys.exit(1)
        
        # Parse the sweep values
        try:
            if args.sweep in ['k', 'intlen', 'nnodes']:
                sweep_values = [int(x) for x in args.sweep_values.split(',')]
            elif args.sweep == 'overlap':
                sweep_values = [float(x) for x in args.sweep_values.split(',')]
            else:
                print(f"Unknown sweep parameter: {args.sweep}")
                sys.exit(1)
        except ValueError:
            print("Error: sweep values must be valid numbers")
            sys.exit(1)
        
        # Set up the fixed parameters with appropriate mapping
        # Map command-line param names to the function parameter names
        param_mapping = {
            'k': 'k',
            'intlen': 'event_length',
            'overlap': 'overlap',
            'nnodes': 'num_nodes'
        }
        
        fixed_params = {
            'k': args.k,
            'event_length': args.intlen,
            'overlap': args.overlap,
            'num_nodes': args.nnodes
        }
        
        # Run the parameter sweep
        print(f"Running parameter sweep for {args.sweep} with values {sweep_values}")
        results = parameter_sweep(args.algorithm, args.sweep, sweep_values, fixed_params, args.iterations)
        
        # Print column names for debugging
        print(f"Available columns in results: {results.columns.tolist()}")
        
        # Visualize the results
        visualize_results(results, param_name=args.sweep)
    else:
        # Run a single experiment with multiple iterations
        results = run_experiment(
            args.algorithm, 
            args.k, 
            args.intlen, 
            args.overlap, 
            args.nnodes,
            args.iterations
        )
        
        # Visualize the results
        visualize_results(results)
    
    # Save the results
    save_results(results, args.output)
    
    print("Experiment completed successfully!")