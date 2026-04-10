"""
Generates the Grouped Bar Chart with Standard Deviation for Phase Analysis
Usage:
    python scripts/analysis/plot_phase_variance.py --input_dir <csv_dir> --output_dir <plot_dir>
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WARMUP_STEPS = 6

def get_steady(df):
    return df.iloc[WARMUP_STEPS:] if len(df) > WARMUP_STEPS else df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the CSVs')
    parser.add_argument('--output_dir', type=str, required=True, help='Output dir for plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    batch_sizes = [8, 4, 2]
    
    phases = {'fwd': 'forward_time_ms', 'bwd': 'backward_time_ms', 'opt': 'optimizer_time_ms'}
    phase_labels = {'fwd': 'Forward Pass', 'bwd': 'Backward Pass', 'opt': 'Optimizer Update'}
    colors = {'fwd': '#3498db', 'bwd': '#e74c3c', 'opt': '#2ecc71'}

    means = {'fwd': [], 'bwd': [], 'opt': []}
    stds = {'fwd': [], 'bwd': [], 'opt': []}

    print("Calculating means and standard deviations across all steady-state steps...")

    for bs in batch_sizes:
        for phase_abbr, col_name in phases.items():
            # Grab all 3 repetitions for this phase and batch size
            files = glob.glob(os.path.join(args.input_dir, f"hardware_stats_bs{bs}_rep*_{phase_abbr}.csv"))
            all_vals = []
            
            for f in files:
                try:
                    df = pd.read_csv(f)
                    steady_df = get_steady(df)
                    if col_name in steady_df.columns:
                        # Append all individual step times to our master list
                        all_vals.extend(steady_df[col_name].dropna().tolist())
                except Exception as e:
                    print(f"Error reading {f}: {e}")

            if all_vals:
                means[phase_abbr].append(np.mean(all_vals))
                stds[phase_abbr].append(np.std(all_vals))
            else:
                means[phase_abbr].append(0)
                stds[phase_abbr].append(0)

    # --- Plotting the Grouped Bar Chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(batch_sizes))
    width = 0.25  # Width of the individual bars

    # Plot each phase side-by-side with yerr (Standard Deviation)
    ax.bar(x - width, means['fwd'], width, yerr=stds['fwd'], label=phase_labels['fwd'], 
           color=colors['fwd'], capsize=5, edgecolor='black', zorder=3)
    ax.bar(x, means['bwd'], width, yerr=stds['bwd'], label=phase_labels['bwd'], 
           color=colors['bwd'], capsize=5, edgecolor='black', zorder=3)
    ax.bar(x + width, means['opt'], width, yerr=stds['opt'], label=phase_labels['opt'], 
           color=colors['opt'], capsize=5, edgecolor='black', zorder=3)

    ax.set_ylabel('Execution Time (ms)', fontweight='bold')
    ax.set_xlabel('Batch Size', fontweight='bold')
    ax.set_title('Per-Phase Execution Time with Standard Deviation', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"BS={bs}" for bs in batch_sizes])
    ax.legend()
    
    # Add a subtle grid behind the bars for easier reading
    ax.grid(True, axis='y', alpha=0.3, zorder=0)

    fig.tight_layout()
    output_path = os.path.join(args.output_dir, "Figure_8_Phase_Variance.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    
    print(f"Successfully generated standard deviation plot at: {output_path}")

if __name__ == '__main__':
    main()
