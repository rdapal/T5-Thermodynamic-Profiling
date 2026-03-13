#!/bin/bash

# ==============================================================================
# T5 MilaBench Sustainability Analysis Experiment Matrix
# ==============================================================================
#
# DESCRIPTION:
# This is the master execution script for T5 benchmarking.
# Automates a 3x3 experiment matrix (3 Batch Sizes x 3 Repetitions) to 
# generate statistically significant, hardware-level profiling data for the 
# T5 NLP workload.
#
# METHODOLOGY & COMPLIANCE:
# We adhere to a <5% measurement overhead threshold and a 500ms 
# hardware polling limit, this script isolates the metrics gathering. Each "Repetition" 
# here consists of 6 strictly distinct 5-minute training loops:
#
#   1. End-to-End Time: Baseline with zero telemetry overhead
#   2. End-to-End Energy: CodeCarbon integration for collecting coarse workload energy
#   3. Fine-Grained Timelines: Runs our `hardware` v3 stats. Disables all CUDA 
#      phase synchronization to allow native async GPU execution. Polls NVML 
#      and PyTorch memory strictly every >= 500ms to eliminate API blocking
#   4. Forward Pass Profiling: Only injects `torch.cuda.synchronize()` around 
#      the Forward Pass
#   5. Backward Pass Profiling: Only injects `torch.cuda.synchronize()` around 
#      the Backward Pass
#   6. Optimizer Profiling: Only injects `torch.cuda.synchronize()` around 
#      the Optimizer step
#
# DATA FORMAT:
#   - Synthetic Data: Pre-generated tensors (seq_len=512, vocab=32128) bypassing I/O
#   - Time Limit: Hard-coded 5:00 minutes (300s) inside Python loop. SLURM config
#     allocation is set at 10:00 to buffer our run time
#   - Outputs: Logs (.log), CodeCarbon files, and V3 Hardware Stats (.csv & .json)
# ==============================================================================

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))

# ------------------------------------------------------------------------------
# FOLDER STRUCTURE SETUP
# ------------------------------------------------------------------------------
DATE_FORMAT=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="hardware_stats/experiments_${DATE_FORMAT}"
RAW_DIR="${BASE_DIR}/raw_data"
PLOT_DIR="${BASE_DIR}/plots"

# Ensure output directories exist locally
mkdir -p "${RAW_DIR}"
mkdir -p "${PLOT_DIR}"

# ------------------------------------------------------------------------------
# EXPERIMENT PARAMETERS
# ------------------------------------------------------------------------------
BATCH_SIZES=(16 8 4)
REPETITIONS=(1 2 3)

echo "========================================================="
echo "Starting T5 Experiment Suite"
echo "Output Directory: ${BASE_DIR}"
echo "========================================================="

for bs in "${BATCH_SIZES[@]}"; do
    for rep in "${REPETITIONS[@]}"; do
        
        echo "========================================================="
        echo "  BATCH SIZE: ${bs} | REPETITION: ${rep}/3"
        echo "========================================================="

        # 1. End-to-End Time (NOOP)
        echo "  -> Run 1/6: End-to-End Time (NOOP)"
        ${SCRIPTS_DIR}/srun.sh --logging.level INFO --model t5 --trainer simple --data synthetic \
            --batch_size $bs --learning_rate 1e-6 --data_configs.dataset.split '"train[:50000]"' \
            --trainer_stats noop > "${OUT_DIR}/e2e_time_bs${bs}_rep${rep}.log"

        # 2. End-to-End Energy (CodeCarbon)
        echo "  -> Run 2/6: End-to-End Energy (CodeCarbon)"
        ${SCRIPTS_DIR}/srun.sh --logging.level INFO --model t5 --trainer simple --data synthetic \
            --batch_size $bs --learning_rate 1e-6 --data_configs.dataset.split '"train[:50000]"' \
            --trainer_stats codecarbon \
            --trainer_stats_configs.codecarbon.output_dir "${OUT_DIR}/codecarbon_bs${bs}_rep${rep}"

        # 3. Fine-Grained Timelines (Hardware stats, no phase syncs)
        # By passing "timeline", simple.py bypasses all phase syncs, leaving only 500ms background polling.
        echo "  -> Run 3/6: Fine-Grained Timelines (500ms Hardware Polling)"
        ${SCRIPTS_DIR}/srun.sh --logging.level INFO --model t5 --trainer simple --data synthetic \
            --batch_size $bs --learning_rate 1e-6 --data_configs.dataset.split '"train[:50000]"' \
            --trainer_stats hardware \
            --trainer_stats_configs.hardware.output_dir "${OUT_DIR}" \
            --trainer_configs.simple.profile_phase "timeline" \
            --trainer_stats_configs.hardware.run_id "timeline_bs${bs}_rep${rep}"

        # 4. Phase Profiling: FORWARD
        echo "  -> Run 4/6: Phase Profiling (Forward)"
        ${SCRIPTS_DIR}/srun.sh --logging.level INFO --model t5 --trainer simple --data synthetic \
            --batch_size $bs --learning_rate 1e-6 --data_configs.dataset.split '"train[:50000]"' \
            --trainer_stats hardware \
            --trainer_stats_configs.hardware.output_dir "${OUT_DIR}" \
            --trainer_configs.simple.profile_phase "forward" \
            --trainer_stats_configs.hardware.run_id "fwd_bs${bs}_rep${rep}"

        # 5. Phase Profiling: BACKWARD
        echo "  -> Run 5/6: Phase Profiling (Backward)"
        ${SCRIPTS_DIR}/srun.sh --logging.level INFO --model t5 --trainer simple --data synthetic \
            --batch_size $bs --learning_rate 1e-6 --data_configs.dataset.split '"train[:50000]"' \
            --trainer_stats hardware \
            --trainer_stats_configs.hardware.output_dir "${OUT_DIR}" \
            --trainer_configs.simple.profile_phase "backward" \
            --trainer_stats_configs.hardware.run_id "bwd_bs${bs}_rep${rep}"

        # 6. Phase Profiling: OPTIMIZER
        echo "  -> Run 6/6: Phase Profiling (Optimizer)"
        ${SCRIPTS_DIR}/srun.sh --logging.level INFO --model t5 --trainer simple --data synthetic \
            --batch_size $bs --learning_rate 1e-6 --data_configs.dataset.split '"train[:50000]"' \
            --trainer_stats hardware \
            --trainer_stats_configs.hardware.output_dir "${OUT_DIR}" \
            --trainer_configs.simple.profile_phase "optimizer" \
            --trainer_stats_configs.hardware.run_id "opt_bs${bs}_rep${rep}"

    done
done

echo "========================================================="
echo " GENERATING PLOTS"
echo "========================================================="
# Loop through all CSV files generated in the raw_data directory
for csv_file in ${RAW_DIR}/*.csv; do
    if [ -f "$csv_file" ]; then
        # Extract filename to use as the plot identifier
        filename=$(basename -- "$csv_file")
        run_id="${filename%.*}"
        run_id="${run_id#hardware_stats_}"
        
        echo "  -> Plotting ${run_id}..."
        python ${SCRIPTS_DIR}/analysis/plot_hardware.py --csv "$csv_file" --output "${PLOT_DIR}" --run_id "${run_id}"
    fi
done

echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo " --> ALL EXPERIMENTS COMPLETE <---"
echo "Results stored in: ${OUT_DIR}"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"