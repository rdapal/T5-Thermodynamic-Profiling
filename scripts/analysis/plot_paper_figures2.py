"""
Usage:
    python scripts/analysis/plot_advanced_figures.py --input_dir <csv_dir> --log_dir <log_dir> --output_dir <plot_dir>
"""
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WARMUP_STEPS = 6

def get_steady(df):
    return df.iloc[WARMUP_STEPS:] if len(df) > WARMUP_STEPS else df

def parse_noop_logs(log_dir, bs):
    """Parses the NOOP text logs"""
    search_path = os.path.join(log_dir, f"e2e_time_bs{bs}_rep*.log")
    log_files = glob.glob(search_path)
    steps = []
    pattern = re.compile(r"limit at step (\d+)")
    for lf in log_files:
        try:
            with open(lf, 'r', encoding='utf-8') as f:
                content = f.read()
                match = pattern.search(content)
                if match:
                    steps.append(int(match.group(1)))
        except Exception as e:
            print(f"Error reading log {lf}: {e}")
            
    if not steps:
        print(f"WARNING: No NOOP steps found for BS={bs} at {search_path}")
    return np.mean(steps) if steps else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    batch_sizes = [8, 4, 2]

    print("Generating Figure 5: Instrumentation Overhead...")
    noop_steps = []
    timeline_steps = []

    for bs in batch_sizes:
        n_steps = parse_noop_logs(args.log_dir, bs)
        noop_steps.append(n_steps)

        tl_files = glob.glob(os.path.join(args.input_dir, f"hardware_stats_bs{bs}_rep*_timeline.csv"))
        t_steps = []
        for tf in tl_files:
            df = pd.read_csv(tf)
            t_steps.append(df['step_num'].max() if 'step_num' in df.columns else len(df))
        timeline_steps.append(np.mean(t_steps) if t_steps else 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(batch_sizes))
    width = 0.35

    # Blue for Baseline/NOOP, Red for Instrumented
    ax.bar(x - width/2, noop_steps, width, label='NOOP (0% Overhead)', color='#3498db', edgecolor='black')
    ax.bar(x + width/2, timeline_steps, width, label='Hardware Timeline Polling', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Total Steps Completed in 300s', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"BS={bs}" for bs in batch_sizes])
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "Figure_5_Instrumentation_Overhead.png"), dpi=300)
    plt.close(fig)

    print("Generating Figure 6: Throughput vs. Efficiency...")
    samples_per_sec = []
    joules_per_sample = []

    for idx, bs in enumerate(batch_sizes):
        avg_steps = timeline_steps[idx]
        total_samples = avg_steps * bs
        sps = total_samples / 300.0  
        samples_per_sec.append(sps)

        tl_files = glob.glob(os.path.join(args.input_dir, f"hardware_stats_bs{bs}_rep*_timeline.csv"))
        energies = []
        for tf in tl_files:
            df = pd.read_csv(tf)
            if 'energy_step_j' in df.columns:
                energies.append(df['energy_step_j'].sum())
        avg_energy = np.mean(energies) if energies else 0
        
        jps = avg_energy / total_samples if total_samples > 0 else 0
        joules_per_sample.append(jps)

    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    ax1.plot(batch_sizes, samples_per_sec, marker='o', markersize=10, color='#2ecc71', lw=3, label='Samples / Sec')
    ax1.set_xlabel('Batch Size', fontweight='bold')
    ax1.set_ylabel('Throughput: Samples / Second', color='#27ae60', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#27ae60')
    ax1.set_xticks(batch_sizes)
    ax1.invert_xaxis() 

    ax2 = ax1.twinx()
    ax2.plot(batch_sizes, joules_per_sample, marker='s', markersize=10, color='#e74c3c', lw=3, label='Joules / Sample')
    ax2.set_ylabel('Efficiency: Joules / Sample', color='#c0392b', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#c0392b')

    ax1.grid(True, alpha=0.3)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "Figure_6_Throughput_Efficiency.png"), dpi=300)
    plt.close(fig)

    print("Generating Figure 7: The 'Zero Dark Time' Validation...")
    bs_labels = [f"BS={bs}" for bs in batch_sizes]
    timeline_means = []
    summed_phases = []

    phases = {'fwd': 'forward_time_ms', 'bwd': 'backward_time_ms', 'opt': 'optimizer_time_ms', 'timeline': 'step_time_ms'}

    for bs in batch_sizes:
        bs_means = {}
        for phase_abbr, col_name in phases.items():
            files = glob.glob(os.path.join(args.input_dir, f"hardware_stats_bs{bs}_rep*_{phase_abbr}.csv"))
            all_vals = []
            for f in files:
                df = pd.read_csv(f)
                steady_df = get_steady(df)
                if col_name in steady_df.columns:
                    all_vals.extend(steady_df[col_name].tolist())
            bs_means[phase_abbr] = np.mean(all_vals) if all_vals else 0
        
        timeline_means.append(bs_means['timeline'])
        summed_phases.append(bs_means['fwd'] + bs_means['bwd'] + bs_means['opt'])

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(batch_sizes))
    width = 0.35

    ax.bar(x - width/2, timeline_means, width, label='Un-instrumented Step Baseline', color='#3498db', edgecolor='black')
    ax.bar(x + width/2, summed_phases, width, label='Sum of Isolated Phases (Fwd+Bwd+Opt)', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Execution Time (ms)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bs_labels)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "Figure_7_Isolation_Validation.png"), dpi=300)
    plt.close(fig)

    print("Plots generated successfully!")

if __name__ == '__main__':
    main()
