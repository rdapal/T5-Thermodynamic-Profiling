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

    print("Generating Figure 1: End-to-End Macro Throughput...")
    fig1_data = []
    for bs in batch_sizes:
        timeline_files = glob.glob(os.path.join(args.input_dir, f"hardware_stats_bs{bs}_rep*_timeline.csv"))
        steps, energies = [], []
        for tf in timeline_files:
            df = pd.read_csv(tf)
            steps.append(df['step_num'].max() if 'step_num' in df.columns else len(df))
            if 'energy_step_j' in df.columns:
                energies.append(df['energy_step_j'].sum())
        avg_steps = np.mean(steps) if steps else 0
        avg_energy = np.mean(energies) if energies else 0
        fig1_data.append((bs, avg_steps, avg_energy))

    bs_labels = [f"BS={bs}" for bs in batch_sizes]
    steps_data = [d[1] for d in fig1_data]
    energy_data = [d[2] for d in fig1_data]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    x = np.arange(len(batch_sizes))
    width = 0.35
    ax1.bar(x - width/2, steps_data, width, label='Total Steps', color='#3498db', edgecolor='black')
    ax1.set_ylabel('Total Steps Completed', color='#3498db', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bs_labels)

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, energy_data, width, label='Total Energy (J)', color='#e74c3c', edgecolor='black')
    ax2.set_ylabel('Total Energy Consumed (Joules)', color='#e74c3c', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "Figure_1_Macro_Throughput.png"), dpi=300)
    plt.close(fig)

    print("Generating Figure 2: Per-Phase Temporal Scaling...")
    phases = {'fwd': 'forward_time_ms', 'bwd': 'backward_time_ms', 'opt': 'optimizer_time_ms'}
    fwd_means, bwd_means, opt_means = [], [], []
    for bs in batch_sizes:
        bs_means = {}
        for phase_abbr, col_name in phases.items():
            phase_files = glob.glob(os.path.join(args.input_dir, f"hardware_stats_bs{bs}_rep*_{phase_abbr}.csv"))
            all_vals = []
            for pf in phase_files:
                df = pd.read_csv(pf)
                steady_df = get_steady(df)
                if col_name in steady_df.columns:
                    all_vals.extend(steady_df[col_name].tolist())
            bs_means[phase_abbr] = np.mean(all_vals) if all_vals else 0
        fwd_means.append(bs_means['fwd'])
        bwd_means.append(bs_means['bwd'])
        opt_means.append(bs_means['opt'])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(bs_labels, fwd_means, label='Forward', color='#3498db', edgecolor='black')
    ax.bar(bs_labels, bwd_means, bottom=fwd_means, label='Backward', color='#e74c3c', edgecolor='black')
    bottom_opt = np.array(fwd_means) + np.array(bwd_means)
    ax.bar(bs_labels, opt_means, bottom=bottom_opt, label='Optimizer', color='#2ecc71', edgecolor='black')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "Figure_2_Temporal_Scaling.png"), dpi=300)
    plt.close(fig)

    print("Generating Figure 3: The Checkpoint I/O Stall...")
    timeline_file = os.path.join(args.input_dir, "hardware_stats_bs2_rep3_timeline.csv")
    ckpt_file = os.path.join(args.input_dir, "hardware_stats_bs2_rep3_ckpt.csv")

    if os.path.exists(timeline_file) and os.path.exists(ckpt_file):
        df_tl = pd.read_csv(timeline_file)
        df_ckpt = pd.read_csv(ckpt_file)
        
        # RUTHLESS FIX: step_time_ms already includes the IO stall! Do not double count!
        time_tl = df_tl['step_time_ms'].cumsum() / 1000.0
        time_ckpt = df_ckpt['step_time_ms'].cumsum() / 1000.0

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        if 'gpu_utilization' in df_tl.columns:
            ax1.plot(time_tl, df_tl['gpu_utilization'], label='Baseline (No Ckpt)', color='#3498db', lw=2)
        if 'gpu_utilization' in df_ckpt.columns:
            ax1.plot(time_ckpt, df_ckpt['gpu_utilization'], label='With Checkpoint', color='#e74c3c', lw=2)
        ax1.set_ylabel('GPU Utilization (%)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        temp_col = 'gpu_temperature_c' if 'gpu_temperature_c' in df_tl.columns else 'gpu_temperature'
        if temp_col in df_tl.columns:
            ax2.plot(time_tl, df_tl[temp_col], label='Baseline Temp', color='#3498db', linestyle='--', lw=2)
        if temp_col in df_ckpt.columns:
            ax2.plot(time_ckpt, df_ckpt[temp_col], label='Ckpt Temp', color='#e74c3c', linestyle='--', lw=2)
            
        ax2.set_ylabel('GPU Temperature (°C)', fontweight='bold')
        ax2.set_xlabel('Time (Seconds)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "Figure_3_Checkpoint_Stall.png"), dpi=300)
        plt.close(fig)

    print("Generating Figure 4: Resource Utilization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = {8: '#3498db', 4: '#e74c3c', 2: '#2ecc71'}

    for bs in batch_sizes:
        timeline_files = glob.glob(os.path.join(args.input_dir, f"hardware_stats_bs{bs}_rep*_timeline.csv"))
        if not timeline_files: 
            continue
        
        dfs = [pd.read_csv(tf) for tf in timeline_files]
        min_len = min(len(df) for df in dfs)
        dfs_trunc = [df.iloc[:min_len].reset_index(drop=True) for df in dfs]
        
        df_concat = pd.concat(dfs_trunc)
        df_avg = df_concat.groupby(level=0).mean(numeric_only=True)

        # RUTHLESS FIX: Pure cumulative sum using the properly bounded step limits
        time_axis = df_avg['step_time_ms'].cumsum() / 1000.0
        
        if 'gpu_memory_allocated_mb' in df_avg.columns:
            ax1.plot(time_axis, df_avg['gpu_memory_allocated_mb'], label=f'Allocated BS={bs}', color=colors[bs], lw=2)
        if 'gpu_memory_reserved_mb' in df_avg.columns:
            ax1.plot(time_axis, df_avg['gpu_memory_reserved_mb'], linestyle='--', color=colors[bs], alpha=0.6)
        
        cpu_col = 'cpu_utilization' if 'cpu_utilization' in df_avg.columns else 'cpu_percent'
        if cpu_col in df_avg.columns:
            ax2.plot(time_axis, df_avg[cpu_col], label=f'CPU Util BS={bs}', color=colors[bs], lw=2)

    ax1.set_ylabel('GPU Memory (MB)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax2.set_ylabel('CPU Utilization (%)', fontweight='bold')
    ax2.set_xlabel('Time (Seconds)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "Figure_4_Resource_Utilization.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Core Plots generated successfully.")

if __name__ == '__main__':
    main()
