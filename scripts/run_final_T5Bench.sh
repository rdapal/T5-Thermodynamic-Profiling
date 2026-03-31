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
#   7. Phase Profiling: Utilize Checkpoint I/O stalls to profile potential checkpointing bottlenecks in workload
#
# DATA FORMAT:
#   - Synthetic Data: Pre-generated tensors (seq_len=512, vocab=32128) bypassing I/O
#   - To consider I/O we run our own Checkpoint Profiling to imitate non-synthetic data envrionments (I/O bottlenecks)
#   - Time Limit: Hard-coded 5:00 minutes (300s) inside Python loop. SLURM config
#     allocation is set at 10:00 to buffer our run time
#   - Outputs: Logs (.log), CodeCarbon files, and V3 Hardware Stats (.csv & .json)
#
# USAGE:
#   Run full matrix:  ./scripts/run_final_T5Bench.sh
#   Run specific BS:  ./scripts/run_final_T5Bench.sh --batch_size 8
#   Run specific Rep: ./scripts/run_final_T5Bench.sh --rep 2
#   Run specific Run: ./scripts/run_final_T5Bench.sh --run_type forward
#   Combine flags:    ./scripts/run_final_T5Bench.sh -b 4 -r 1 -t timeline
# ==============================================================================

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( cd "${SCRIPTS_DIR}/.." >/dev/null 2>&1 && pwd )"

# ------------------------------------------------------------------------------
# ARGUMENT PARSING
# ------------------------------------------------------------------------------
TARGET_BS=""
TARGET_REP=""
TARGET_RUN=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -b|--batch_size) TARGET_BS="$2"; shift ;;
        -r|--rep) TARGET_REP="$2"; shift ;;
        -t|--run_type) TARGET_RUN="$2"; shift ;;
        *) echo "[ERROR] Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Set arrays based on flags, or default to the full matrix
if [ -n "$TARGET_BS" ]; then BATCH_SIZES=($TARGET_BS); else BATCH_SIZES=(8 4 2); fi
if [ -n "$TARGET_REP" ]; then REPETITIONS=($TARGET_REP); else REPETITIONS=(1 2 3); fi

# ------------------------------------------------------------------------------
# FOLDER STRUCTURE SETUP
# ------------------------------------------------------------------------------
DATE_FORMAT=$(date +"%Y%m%d_%H%M%S")

# Remote SLURM Shared Storage
REMOTE_DIR="/home/slurm/comp597/students/rdapal/experiments_${DATE_FORMAT}"
REMOTE_INDIV_DIR="${REMOTE_DIR}/individual_experiments"

# Local Storage for final plots, logs, and CSV backups
LOCAL_BASE_DIR="${PROJECT_ROOT}/hardware_stats/experiments_${DATE_FORMAT}"
LOCAL_INDIV_DIR="${LOCAL_BASE_DIR}/individual_experiments"
LOCAL_PLOT_DIR="${LOCAL_BASE_DIR}/plots"
LOCAL_LOG_DIR="${LOCAL_BASE_DIR}/logs"

# Ensure directories exist
mkdir -p "${LOCAL_INDIV_DIR}" "${LOCAL_PLOT_DIR}" "${LOCAL_LOG_DIR}"
${SCRIPTS_DIR}/bash_srun.sh "mkdir -p ${REMOTE_INDIV_DIR}"

export COMP597_SLURM_TIME_LIMIT="10:00" # 10 minute allocation to pad time

echo "========================================================="
echo "Starting T5 Experiment Suite"
echo "   Target Batch Sizes : ${BATCH_SIZES[*]}"
echo "   Target Repetitions : ${REPETITIONS[*]}"
if [ -n "$TARGET_RUN" ]; then echo "   Target Run Type    : ${TARGET_RUN}"; fi
echo "   Remote Storage     : ${REMOTE_INDIV_DIR}"
echo "========================================================="

for bs in "${BATCH_SIZES[@]}"; do
    for rep in "${REPETITIONS[@]}"; do
        
        echo "========================================================="
        echo "  BATCH SIZE: ${bs} | REPETITION: ${rep}/3"
        echo "========================================================="

        # 1. End-to-End Time (NOOP)
        if [[ -z "$TARGET_RUN" || "$TARGET_RUN" == "noop" ]]; then
            echo "  -> Run: End-to-End Time (NOOP)"
            ${SCRIPTS_DIR}/srun.sh --logging.level INFO --model t5 --trainer simple --data synthetic \
                --batch_size $bs --learning_rate 1e-6 --data_configs.dataset.split '"train[:50000]"' \
                --trainer_stats noop > "${LOCAL_LOG_DIR}/e2e_time_bs${bs}_rep${rep}.log"
        fi

        # 2. End-to-End Energy (CodeCarbon)
        if [[ -z "$TARGET_RUN" || "$TARGET_RUN" == "codecarbon" ]]; then
            echo "  -> Run: End-to-End Energy (CodeCarbon)"
            ${SCRIPTS_DIR}/srun.sh --logging.level INFO --model t5 --trainer simple --data synthetic \
                --batch_size $bs --learning_rate 1e-6 --data_configs.dataset.split '"train[:50000]"' \
                --trainer_stats codecarbon \
                --trainer_stats_configs.codecarbon.output_dir "${REMOTE_INDIV_DIR}/codecarbon_bs${bs}_rep${rep}"
        fi

        # 3. Fine-Grained Timelines (Hardware stats, no phase syncs)
        if [[ -z "$TARGET_RUN" || "$TARGET_RUN" == "timeline" ]]; then
            echo "  -> Run: Fine-Grained Timelines (500ms Hardware Polling)"
            ${SCRIPTS_DIR}/srun.sh --logging.level INFO --model t5 --trainer simple --data synthetic \
                --batch_size $bs --learning_rate 1e-6 --data_configs.dataset.split '"train[:50000]"' \
                --trainer_stats hardware \
                --trainer_stats_configs.hardware.output_dir "${REMOTE_INDIV_DIR}" \
                --trainer_configs.simple.profile_phase "timeline" \
                --trainer_stats_configs.hardware.run_id "bs${bs}_rep${rep}_timeline"
        fi

        # 4. Phase Profiling: FORWARD
        if [[ -z "$TARGET_RUN" || "$TARGET_RUN" == "forward" ]]; then
            echo "  -> Run: Phase Profiling (Forward)"
            ${SCRIPTS_DIR}/srun.sh --logging.level INFO --model t5 --trainer simple --data synthetic \
                --batch_size $bs --learning_rate 1e-6 --data_configs.dataset.split '"train[:50000]"' \
                --trainer_stats hardware \
                --trainer_stats_configs.hardware.output_dir "${REMOTE_INDIV_DIR}" \
                --trainer_configs.simple.profile_phase "forward" \
                --trainer_stats_configs.hardware.run_id "bs${bs}_rep${rep}_fwd"
        fi

        # 5. Phase Profiling: BACKWARD
        if [[ -z "$TARGET_RUN" || "$TARGET_RUN" == "backward" ]]; then
            echo "  -> Run: Phase Profiling (Backward)"
            ${SCRIPTS_DIR}/srun.sh --logging.level INFO --model t5 --trainer simple --data synthetic \
                --batch_size $bs --learning_rate 1e-6 --data_configs.dataset.split '"train[:50000]"' \
                --trainer_stats hardware \
                --trainer_stats_configs.hardware.output_dir "${REMOTE_INDIV_DIR}" \
                --trainer_configs.simple.profile_phase "backward" \
                --trainer_stats_configs.hardware.run_id "bs${bs}_rep${rep}_bwd"
        fi

        # 6. Phase Profiling: OPTIMIZER
        if [[ -z "$TARGET_RUN" || "$TARGET_RUN" == "optimizer" ]]; then
            echo "  -> Run: Phase Profiling (Optimizer)"
            ${SCRIPTS_DIR}/srun.sh --logging.level INFO --model t5 --trainer simple --data synthetic \
                --batch_size $bs --learning_rate 1e-6 --data_configs.dataset.split '"train[:50000]"' \
                --trainer_stats hardware \
                --trainer_stats_configs.hardware.output_dir "${REMOTE_INDIV_DIR}" \
                --trainer_configs.simple.profile_phase "optimizer" \
                --trainer_stats_configs.hardware.run_id "bs${bs}_rep${rep}_opt"
        fi

	# 7. Phase Profiling: CHECKPOINT I/O STALL
        if [[ -z "$TARGET_RUN" || "$TARGET_RUN" == "ckpt" ]]; then
            echo "  -> Run: Phase Profiling (Checkpoint I/O Stall)"
            ${SCRIPTS_DIR}/srun.sh --logging.level INFO --model t5 --trainer simple --data synthetic \
                --batch_size $bs --learning_rate 1e-6 --data_configs.dataset.split '"train[:50000]"' \
                --trainer_stats hardware \
                --trainer_stats_configs.hardware.output_dir "${REMOTE_INDIV_DIR}" \
                --trainer_configs.simple.profile_phase "ckpt" \
                --trainer_stats_configs.hardware.run_id "bs${bs}_rep${rep}_ckpt"
        fi

    done
done

echo "========================================================="
echo " FETCHING DATA FROM SLURM STORAGE"
echo "========================================================="
# Pull the generated CSVs back into the local raw_data directory
${SCRIPTS_DIR}/bash_srun.sh "bash -c 'cp ${REMOTE_INDIV_DIR}/*.csv ${LOCAL_INDIV_DIR}/ 2>/dev/null || true'"
# Pull the generated JSON metadata files as well
${SCRIPTS_DIR}/bash_srun.sh "bash -c 'cp ${REMOTE_INDIV_DIR}/*.json ${LOCAL_INDIV_DIR}/ 2>/dev/null || true'"

echo "========================================================="
echo " GENERATING PLOTS"
echo "========================================================="
# Safely attempt to plot the fetched data
if [ "$(ls -A ${LOCAL_INDIV_DIR}/*.csv 2>/dev/null)" ]; then
    echo "  -> Aggregating data and generating plots per batch size..."
    ${PROJECT_ROOT}/venv/bin/python ${SCRIPTS_DIR}/analysis/plot_hardware.py \
        --input_dir "${LOCAL_INDIV_DIR}" \
        --output_dir "${LOCAL_PLOT_DIR}"
else
    echo "[WARNING] No CSV files were found in ${LOCAL_INDIV_DIR} to plot."
fi

echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo " --> ALL EXPERIMENTS COMPLETE <---"
echo "Results stored locally in: ${LOCAL_BASE_DIR}"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
