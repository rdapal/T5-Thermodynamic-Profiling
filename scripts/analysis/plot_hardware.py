"""
Plot hardware monitoring results for T5 energy analysis v.3

Reads the 500ms CSV output from hardware.py
Removes invalid sub-millisecond per-phase energy/power visualisations,
focusing on accurate macro-timelines as directed by Prof. Balmau

Plots Generated:
  1. Phase Durations         — if profile_phase was 'all' or specific phase
  2. Energy per Step         — dual-axis plot: per-step + cumulative
  3. CO2 Emissions           — dual-axis plot: per-step + cumulative
  4. GPU Memory              — line plot: allocated / reserved / peak
  5. GPU & CPU Utilization   — line plot with steady-state averages
  6. GPU Temperature         — line plot with thermal rise annotation

Usage:
    python scripts/analysis/plot_hardware.py --csv <path> --output <dir> --run_id <id>
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================
WARMUP_STEPS = 6

# Consistent colour scheme across all plots
C_FORWARD       = '#3498db'
C_BACKWARD      = '#e74c3c'
C_OPTIMIZER     = '#2ecc71'
C_TOTAL         = '#9b59b6'
C_ENERGY        = '#e74c3c'
C_CUMULATIVE    = '#3498db'
C_CO2           = '#27ae60'
C_CO2_CUM       = '#2c3e50'
C_TEMPERATURE   = '#e67e22'
C_UTILIZATION   = '#e67e22'
C_CPU           = '#7f8c8d'


# ============================================================
# Helpers
# ============================================================

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def steady(df):
    return df.iloc[WARMUP_STEPS:]

def savefig(fig, output_dir, name, run_id):
    path = os.path.join(output_dir, f"{name}_{run_id}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")

# ============================================================
# Plot 1: Phase Durations
# ============================================================
def plot_phase_durations(df, output_dir, run_id):
    """Plots timing. Note: In isolated mode, un-profiled phases will be 0."""
    fig, ax = plt.subplots(figsize=(12, 6))

    if df['forward_time_ms'].mean() > 0:
        ax.plot(df['step_num'], df['forward_time_ms'], label='Forward', lw=2, color=C_FORWARD)
    if df['backward_time_ms'].mean() > 0:
        ax.plot(df['step_num'], df['backward_time_ms'], label='Backward', lw=2, color=C_BACKWARD)
    if df['optimizer_time_ms'].mean() > 0:
        ax.plot(df['step_num'], df['optimizer_time_ms'], label='Optimizer', lw=2, color=C_OPTIMIZER)

    ax.plot(df['step_num'], df['step_time_ms'], label='Total Step', lw=2, color=C_TOTAL, linestyle='--', alpha=0.7)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Duration (ms)', fontsize=12)
    ax.set_title('Step & Phase Durations', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    savefig(fig, output_dir, 'phase_durations', run_id)

# ============================================================
# Plot 2: Energy per Step + Cumulative
# ============================================================
def plot_energy_per_step(df, output_dir, run_id):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    energy_mj = df['energy_step_j'] * 1000
    cumulative_j = df['energy_step_j'].cumsum()
    ss = steady(df)
    avg_mj = ss['energy_step_j'].mean() * 1000

    ax1.plot(df['step_num'], energy_mj, color=C_ENERGY, lw=1.5, alpha=0.8, label='Per-step energy')
    ax1.axhline(y=avg_mj, color=C_ENERGY, linestyle='--', alpha=0.5, label=f'Steady avg: {avg_mj:.1f} mJ/step')

    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Energy per Step (mJ)', fontsize=12, color=C_ENERGY)
    ax1.tick_params(axis='y', labelcolor=C_ENERGY)

    ax2 = ax1.twinx()
    ax2.plot(df['step_num'], cumulative_j, color=C_CUMULATIVE, lw=2, linestyle='--', alpha=0.7, label='Cumulative energy')
    ax2.set_ylabel('Cumulative Energy (J)', fontsize=12, color=C_CUMULATIVE)
    ax2.tick_params(axis='y', labelcolor=C_CUMULATIVE)

    total_j = cumulative_j.iloc[-1]
    ax2.annotate(f'Total: {total_j:.0f} J ({total_j/3600:.2f} Wh)',
                 xy=(df['step_num'].iloc[-1], total_j),
                 xytext=(-120, -20), textcoords='offset points',
                 fontsize=10, color=C_CUMULATIVE,
                 arrowprops=dict(arrowstyle='->', color=C_CUMULATIVE))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

    ax1.set_title('Energy Consumption (500ms Hardware Polling)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    savefig(fig, output_dir, 'energy_per_step', run_id)

# ============================================================
# Plot 3: CO2 Emissions
# ============================================================
def plot_co2_emissions(df, output_dir, run_id):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    co2_mg = df['co2_step_mg']
    cumulative_mg = co2_mg.cumsum()
    ss = steady(df)
    avg_co2 = ss['co2_step_mg'].mean()

    ax1.plot(df['step_num'], co2_mg, color=C_CO2, lw=1.5, alpha=0.8, label='Per-step CO₂')
    ax1.axhline(y=avg_co2, color=C_CO2, linestyle='--', alpha=0.5, label=f'Steady avg: {avg_co2:.4f} mg/step')
    
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('CO₂ per Step (mg)', fontsize=12, color=C_CO2)
    ax1.tick_params(axis='y', labelcolor=C_CO2)

    ax2 = ax1.twinx()
    ax2.plot(df['step_num'], cumulative_mg, color=C_CO2_CUM, lw=2, linestyle='--', alpha=0.7, label='Cumulative CO₂')
    ax2.set_ylabel('Cumulative CO₂ (mg)', fontsize=12, color=C_CO2_CUM)
    ax2.tick_params(axis='y', labelcolor=C_CO2_CUM)

    total_mg = cumulative_mg.iloc[-1]
    ax2.annotate(f'Total: {total_mg:.2f} mg\n({total_mg/1000:.4f} g CO₂eq)',
                 xy=(df['step_num'].iloc[-1], total_mg),
                 xytext=(-140, -25), textcoords='offset points',
                 fontsize=10, color=C_CUMULATIVE,
                 arrowprops=dict(arrowstyle='->', color=C_CO2_CUM))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

    ax1.set_title('CO₂ Emissions', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    savefig(fig, output_dir, 'co2_emissions', run_id)

# ============================================================
# Plot 4: GPU Memory
# ============================================================
def plot_gpu_memory(df, output_dir, run_id):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['step_num'], df['gpu_memory_allocated_mb'], label='Allocated (Tensors)', lw=2, color=C_OPTIMIZER)
    ax.plot(df['step_num'], df['gpu_memory_reserved_mb'], label='Reserved (Caching Alloc)', lw=2, color=C_FORWARD, linestyle='--')
    ax.plot(df['step_num'], df['gpu_memory_peak_mb'], label='Peak (Step Max)', lw=2, color=C_BACKWARD, linestyle=':')

    ax.axhline(y=32768, color='black', linestyle='--', alpha=0.3, lw=1)
    ax.text(df['step_num'].iloc[-1], 32768 + 300, '32 GB VRAM', ha='right', fontsize=9, color='#7f8c8d')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('GPU Memory (MB)', fontsize=12)
    ax.set_title('GPU Memory Footprint', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    savefig(fig, output_dir, 'gpu_memory', run_id)

# ============================================================
# Plot 5: GPU/CPU Utilization
# ============================================================
def plot_utilization(df, output_dir, run_id):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['step_num'], df['gpu_utilization'], lw=2, color=C_UTILIZATION, label='GPU Compute Util')
    if 'cpu_utilization' in df.columns:
        ax.plot(df['step_num'], df['cpu_utilization'], lw=1.5, color=C_CPU, linestyle=':', label='CPU Util')

    ss = steady(df)
    avg_gpu = ss['gpu_utilization'].mean()
    ax.axhline(y=avg_gpu, color='red', linestyle='--', label=f'GPU Steady Avg: {avg_gpu:.1f}%')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Utilization (%)', fontsize=12)
    ax.set_title('Compute Utilization', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    savefig(fig, output_dir, 'utilization', run_id)

# ============================================================
# Plot 6: GPU Temperature
# ============================================================
def plot_gpu_temperature(df, output_dir, run_id):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['step_num'], df['gpu_temperature_c'], lw=2, color=C_TEMPERATURE)

    t_start = df['gpu_temperature_c'].iloc[0]
    t_end = df['gpu_temperature_c'].iloc[-1]

    ax.annotate(f'{t_start:.0f}°C', xy=(0, t_start), xytext=(15, -20),
                textcoords='offset points', fontsize=11, arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate(f'{t_end:.0f}°C', xy=(df['step_num'].iloc[-1], t_end),
                xytext=(-40, -20), textcoords='offset points', fontsize=11, arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('GPU Thermal Rise', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    savefig(fig, output_dir, 'gpu_temperature', run_id)


def main():
    parser = argparse.ArgumentParser(description='Plot V3 hardware stats')
    parser.add_argument('--csv', type=str, required=True, help='Path to hardware_stats CSV')
    parser.add_argument('--output', type=str, default='./plots', help='Output dir')
    parser.add_argument('--run_id', type=str, default='v3', help='Run identifier')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = load_data(args.csv)

    print(f"Generating V3 plots for {args.csv}...")
    plot_phase_durations(df, args.output, args.run_id)
    plot_energy_per_step(df, args.output, args.run_id)
    plot_co2_emissions(df, args.output, args.run_id)
    plot_gpu_memory(df, args.output, args.run_id)
    plot_utilization(df, args.output, args.run_id)
    plot_gpu_temperature(df, args.output, args.run_id)
    print("Done!")

if __name__ == '__main__':
    main()