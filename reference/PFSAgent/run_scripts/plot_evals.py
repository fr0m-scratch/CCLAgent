import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import json
import os
import argparse



def plot_means_with_ci(workload, data_file, save_path):
    """
    Plot means with 90% confidence intervals for each scenario in the data.
    
    Args:
        data_file: Path to JSON file containing the stats
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save the figure (if None, will show instead)
    """
        # Set seaborn style for professional looking plots
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=2)
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    data = data[workload]
    
    scenarios = list(data.keys())
    means = [data[scenario]['mean'] for scenario in scenarios]
    
    # Calculate 90% confidence intervals using t-distribution
    n = 3  # Sample size - adjust this based on your actual data
    confidence = 0.9
    t_value = stats.t.ppf((1 + confidence) / 2, n-1)
    
    ci_errors = []
    for scenario in scenarios:
        stddev = data[scenario]['stddev']
        ci = t_value * (stddev / np.sqrt(n))
        ci_errors.append(ci)
    
    # Create the plot with seaborn
    plt.figure(figsize=(5, 6))
    
    # Create dataframe for seaborn
    plot_data = pd.DataFrame({
        'Scenario': scenarios,
        'Mean': means,
        'CI': ci_errors
    })
    colors = sns.color_palette("colorblind", 3)
    # Use seaborn's barplot with error bars
    ax = sns.barplot(x='Scenario', y='Mean', data=plot_data, 
                    palette=['#348ABD', '#988ED5', '#E24A33'], errorbar=None, alpha=0.85)
    
    # Add error bars manually to have more control
    for i, bar in enumerate(ax.patches):
        ax.errorbar(i, means[i], yerr=ci_errors[i], fmt='none', 
                   color='black', capsize=8, elinewidth=2)
    
    # Add mean values on top of bars
    #for i, bar in enumerate(ax.patches):
    #    height = bar.get_height()
    #    ax.text(bar.get_x() + bar.get_width()/2., height + ci_errors[i] + 0.02*max(means),
    #           f'{means[i]:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Set labels and title with improved styling
    ax.set_xlabel("")
    ax.set_ylabel("Walltime (seconds)")
    ax.set_title(workload, fontsize=14, fontweight='bold', pad=20)
    
    # Enhance x-axis labels if needed
    plt.xticks(rotation=0, fontweight='bold')
    
    # Add a subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
    
    plt.tight_layout()
    
    # Save or show the plot
    plt.savefig(f"{save_path}/{workload}_walltime_comparison.png", dpi=300, bbox_inches='tight')


def plot_app_speedups(app, data_file, save_path):
    with open(data_file, "r") as f:
        data = json.load(f)
    
    # Modify the initial speedup values to be 1.0 instead of 0
    # Set seaborn theme for a professional look
    sns.set_theme(style="whitegrid", context="paper", font_scale=2.2)
    
    # Modify the initial speedup values to be 1.0 instead of 0
    no_rules_speedup = [1.0] + data[app]["No Rules"]["calculated_speedup"]
    with_rules_speedup = [1.0] + data[app]["With Rules"]["calculated_speedup"]
    
    # Get the default_mean_performance for the application
    default_perf = data[app]["No Rules"]["default_mean_performance"]
    
    # Create DataFrame for seaborn (better for styling and plotting)
    df_list = []
    
    if len(no_rules_speedup) > 1:
        for i, val in enumerate(no_rules_speedup):
            df_list.append({'Iteration': i, 'Speedup': val, 'Method': 'No Rules'})
            
    if len(with_rules_speedup) > 1:
        for i, val in enumerate(with_rules_speedup):
            df_list.append({'Iteration': i, 'Speedup': val, 'Method': 'With Rules'})
    
    df = pd.DataFrame(df_list)
    
    # Find the minimum speedup across both datasets
    all_speedups = no_rules_speedup + with_rules_speedup
    min_speedup = min(all_speedups)
    max_speedup = max(all_speedups)
    speedup_range = max_speedup - min_speedup
    
    # Create figure with higher DPI for better resolution
    plt.figure(figsize=(8, 6), dpi=120)
    
    # Use seaborn's lineplot with improved styling
    ax = sns.lineplot(
        data=df, x='Iteration', y='Speedup', hue='Method', 
        style='Method', markers=['D', 'o'], dashes=False,
        palette=['#348ABD', '#E24A33', '#988ED5', '#FBC15E'] ,
        linewidth=3.5, markersize=14,
        err_style=None
    )
    
    # Only show baseline line if there's a speedup less than 1
    if min_speedup < 1.0:
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, 
                   linewidth=2.5, label='Baseline')
        
    # Enhance the title and axis labels
    print(f"{app} (Default: {default_perf:.2f})")
    #plt.title(f"{plot_name} (Default: {default_perf:.2f})", 
    #         fontsize=22, fontweight='bold', pad=20)
    #plt.xlabel('Iteration', fontsize=18, fontweight='bold')
    #plt.ylabel('Speedup', fontsize=18, fontweight='bold')
    
    # Format axis ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Set y-axis to start at the minimum speedup value with a buffer
    y_min = min_speedup - 0.05 * speedup_range
    y_max = max_speedup + 0.10 * speedup_range  # Add more space at the top for annotations
    plt.ylim(bottom=y_min, top=y_max)
    
    # Enhance the legend
    plt.legend(title=None, frameon=True, facecolor='white', 
              edgecolor='lightgray', shadow=True, 
              loc='lower right', fontsize=24)
    
    # Annotate the best speedup for each method
    if len(no_rules_speedup) > 1:
        best_idx = np.argmax(no_rules_speedup)
        best_val = no_rules_speedup[best_idx]
        print(best_val)
        """
        plt.annotate(f'{best_val:.2f}x', 
                    xy=(best_idx, best_val),
                    xytext=(5, 10), textcoords='offset points',
                    fontsize=20, fontweight='bold', 
                    color=sns.color_palette("colorblind")[0],
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        """
    if len(with_rules_speedup) > 1:
        best_idx = np.argmax(with_rules_speedup)
        best_val = with_rules_speedup[best_idx]
        print(best_val)
        """
        plt.annotate(f'{best_val:.2f}x', 
                    xy=(best_idx, best_val),
                    xytext=(5, 10), textcoords='offset points',
                    fontsize=20, fontweight='bold', 
                    color=sns.color_palette("colorblind")[1],
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        """
    # Add subtle border with rounded corners
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
        spine.set_linewidth(1.5)

    plt.xlabel("Iteration (1st iteration is baseline)")
    plt.ylabel("Speedup (compared to baseline)")
    plt.title(f"{app} (Default: {default_perf:.2f}s)")
    
    # Add subtle grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/{app}_speedup.png", bbox_inches='tight')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_idx", type=int, choices=[1, 2, 3], required=True)
    args = parser.parse_args()


    save_path = f"/custom-install/eval_results/plots/Eval_{args.eval_idx}"
    if args.eval_idx == 1:
        data_file = "/custom-install/eval_results/baseline_tuning_performance.json"
    else:
        data_file = "/custom-install/eval_results/tuning_speedups.json"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    if args.eval_idx == 1 or args.eval_idx == 2:
        workloads = ["IO500", "IOR_64k", "IOR_16m", "MDWorkbench_2k", "MDWorkbench_8k"]
    else:
        workloads = ["AMReX", "MACSio_512K", "MACSio_16M"]
    for app in workloads:
        if args.eval_idx == 1:
            plot_means_with_ci(app, data_file, save_path)
        else:
            plot_app_speedups(app, data_file, save_path)