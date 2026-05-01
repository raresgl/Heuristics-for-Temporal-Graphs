import pandas as pd
import numpy as np
from pathlib import Path

def load_and_process_csv(file_path):
    """Load CSV and ensure proper data types"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path} with {len(df)} rows")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_averages(df, metrics=['total_length', 'precision', 'recall', 'f_measure']):
    """Calculate averages for each unique combination of parameters (excluding iteration)"""
    # Group by all parameters except iteration to calculate averages
    grouping_cols = [col for col in df.columns if col not in ['iteration'] + metrics]
    
    # Calculate averages for the specified metrics
    avg_df = df.groupby(grouping_cols)[metrics].mean().reset_index()
    
    # Add standard deviations for reference
    std_df = df.groupby(grouping_cols)[metrics].std().reset_index()
    
    # Rename std columns
    for metric in metrics:
        std_df = std_df.rename(columns={metric: f'{metric}_std'})
    
    # Merge averages with standard deviations
    result_df = avg_df.merge(std_df, on=grouping_cols)
    
    return result_df

def compare_algorithms(df1, df2, alg1_name=None, alg2_name=None, metrics=['total_length', 'precision', 'recall', 'f_measure']):
    """Compare two algorithms across specified metrics"""
    
    # Auto-detect algorithm names from the data if not provided
    if alg1_name is None:
        alg1_name = df1['algorithm'].iloc[0] if 'algorithm' in df1.columns else 'algorithm1'
    if alg2_name is None:
        alg2_name = df2['algorithm'].iloc[0] if 'algorithm' in df2.columns else 'algorithm2'
    
    print(f"Comparing {alg1_name} vs {alg2_name}")
    
    # Calculate averages for both algorithms
    avg1 = calculate_averages(df1, metrics)
    avg2 = calculate_averages(df2, metrics)
    
    # Add algorithm identifiers
    avg1['algorithm_name'] = alg1_name
    avg2['algorithm_name'] = alg2_name
    
    # Combine the dataframes
    combined = pd.concat([avg1, avg2], ignore_index=True)
    
    # Create comparison table
    comparison_results = []
    
    # Get unique parameter combinations (excluding algorithm)
    param_cols = [col for col in combined.columns if col not in ['algorithm_name'] + metrics + [f'{m}_std' for m in metrics]]
    unique_params = combined[param_cols].drop_duplicates()
    
    for _, params in unique_params.iterrows():
        # Filter data for this parameter combination
        mask = True
        for col in param_cols:
            mask = mask & (combined[col] == params[col])
        
        subset = combined[mask]
        
        if len(subset) == 2:  # Both algorithms present
            alg1_data = subset[subset['algorithm_name'] == alg1_name].iloc[0]
            alg2_data = subset[subset['algorithm_name'] == alg2_name].iloc[0]
            
            result = params.to_dict()
            
            for metric in metrics:
                result[f'{metric}_{alg1_name}'] = alg1_data[metric]
                result[f'{metric}_{alg2_name}'] = alg2_data[metric]
                result[f'{metric}_diff'] = alg2_data[metric] - alg1_data[metric]
                result[f'{metric}_diff_pct'] = ((alg2_data[metric] - alg1_data[metric]) / alg1_data[metric] * 100) if alg1_data[metric] != 0 else 0
                
                # Add standard deviations
                result[f'{metric}_{alg1_name}_std'] = alg1_data[f'{metric}_std']
                result[f'{metric}_{alg2_name}_std'] = alg2_data[f'{metric}_std']
            
            comparison_results.append(result)
    
    return pd.DataFrame(comparison_results), combined

def create_summary_table(comparison_df, alg1_name, alg2_name, metrics=['total_length', 'precision', 'recall', 'f_measure']):
    """Create a summary table showing which algorithm performs better"""
    summary = []
    
    for _, row in comparison_df.iterrows():
        result = {
            'num_nodes': row['num_nodes'],
            'num_timestamps': row['num_timestamps']
        }
        
        for metric in metrics:
            alg1_col = f'{metric}_{alg1_name}'
            alg2_col = f'{metric}_{alg2_name}'
            diff_col = f'{metric}_diff_pct'
            
            if alg1_col not in row or alg2_col not in row:
                print(f"Warning: Missing columns for {metric}. Available columns: {list(row.index)}")
                continue
                
            alg1_val = row[alg1_col]
            alg2_val = row[alg2_col]
            diff_pct = row[diff_col]
            
            if metric == 'total_length':
                # For total_length, lower is better
                better = alg2_name if alg2_val < alg1_val else alg1_name
                result[f'{metric}_better'] = better
                result[f'{metric}_improvement'] = abs(diff_pct)
            else:
                # For precision, recall, f_measure, higher is better
                better = alg2_name if alg2_val > alg1_val else alg1_name
                result[f'{metric}_better'] = better
                result[f'{metric}_improvement'] = abs(diff_pct)
        
        summary.append(result)
    
    return pd.DataFrame(summary)

def main():
    # File paths - modify these to match your CSV files

    csv1_path = "experiment_results/experiment_results_20250514_201432.csv"  # Replace with actual path
    csv2_path = "experiment_results/experiment_results_20250514_202218.csv"  # Replace with actual path
    
    print("Loading CSV files...")
    df1 = load_and_process_csv(csv1_path)
    df2 = load_and_process_csv(csv2_path)
    
    if df1 is None or df2 is None:
        print("Failed to load one or both CSV files. Please check file paths.")
        return
    
    print(f"Algorithm 1 data shape: {df1.shape}")
    print(f"Algorithm 2 data shape: {df2.shape}")
    
    # Metrics to compare
    metrics = ['total_length', 'precision', 'recall', 'f_measure']
    
    # Compare algorithms
    print("\nComparing algorithms...")
    comparison_df, combined_df = compare_algorithms(df1, df2, metrics=metrics)
    
    # Get algorithm names for later use
    alg1_name = df1['algorithm'].iloc[0] if 'algorithm' in df1.columns else 'algorithm1'
    alg2_name = df2['algorithm'].iloc[0] if 'algorithm' in df2.columns else 'algorithm2'
    
    print(f"Algorithm names detected: {alg1_name}, {alg2_name}")
    
    # Check if comparison was successful
    if comparison_df.empty:
        print("Error: No matching parameter combinations found between the two datasets.")
        return
    
    # Create summary table
    summary_df = create_summary_table(comparison_df, alg1_name, alg2_name, metrics)
    
    # Display results
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON RESULTS")
    print("="*80)
    
    print(f"\nDetailed Comparison (averages over iterations):")
    print("-" * 60)
    
    # Format and display the comparison table
    display_cols = ['num_nodes', 'num_timestamps']
    for metric in metrics:
        metric_cols = [f'{metric}_{alg1_name}', f'{metric}_{alg2_name}', f'{metric}_diff_pct']
        # Only add columns that actually exist in the dataframe
        existing_cols = [col for col in metric_cols if col in comparison_df.columns]
        display_cols.extend(existing_cols)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    # Only display columns that exist
    final_display_cols = [col for col in display_cols if col in comparison_df.columns]
    print(comparison_df[final_display_cols].to_string(index=False))
    
    print(f"\nSummary - Which Algorithm Performs Better:")
    print("-" * 50)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No summary data available.")
    
    # Save results to CSV files
    print(f"\nSaving results...")
    comparison_df.to_csv('algorithm_comparison_detailed.csv', index=False)
    summary_df.to_csv('algorithm_comparison_summary.csv', index=False)
    combined_df.to_csv('combined_averages.csv', index=False)
    
    print("Results saved to:")
    print("- algorithm_comparison_detailed.csv")
    print("- algorithm_comparison_summary.csv") 
    print("- combined_averages.csv")
    
    # Overall statistics
    print(f"\nOverall Performance Summary:")
    print("-" * 40)
    
    if not summary_df.empty:
        for metric in metrics:
            better_col = f'{metric}_better'
            improvement_col = f'{metric}_improvement'
            
            if better_col in summary_df.columns and improvement_col in summary_df.columns:
                if metric == 'total_length':
                    wins_alg1 = sum(summary_df[better_col] == alg1_name)
                    wins_alg2 = sum(summary_df[better_col] == alg2_name)
                    avg_improvement = summary_df[improvement_col].mean()
                    print(f"{metric.upper()} (lower is better):")
                    print(f"  {alg1_name} wins: {wins_alg1}, {alg2_name} wins: {wins_alg2}")
                    print(f"  Average improvement: {avg_improvement:.2f}%")
                else:
                    wins_alg1 = sum(summary_df[better_col] == alg1_name)
                    wins_alg2 = sum(summary_df[better_col] == alg2_name)
                    avg_improvement = summary_df[improvement_col].mean()
                    print(f"{metric.upper()} (higher is better):")
                    print(f"  {alg1_name} wins: {wins_alg1}, {alg2_name} wins: {wins_alg2}")
                    print(f"  Average improvement: {avg_improvement:.2f}%")
    else:
        print("No summary statistics available.")

if __name__ == "__main__":
    main()