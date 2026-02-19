#!/bin/bash
#SBATCH --job-name=wings_gpu
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_L40,aoraki_gpu,aoraki_gpu_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=64GB
#SBATCH --output=wings_gpu_%j.log
#SBATCH --error=wings_gpu_%j.err
# ============================================================
# W.I.N.G.S. — GPU-Accelerated Wolbachia Spread Simulation
# ============================================================
#
# Scheduling strategy: SINGLE JOB, MULTIPLE GPUs, PARALLEL LOOP
#
# Instead of 160 separate array tasks (each needing its own
# scheduler allocation + conda init + filesystem wait), this
# runs ALL 160 combinations inside ONE job with 4 GPUs.
# 4 simulations run in parallel (one per GPU), cycling through
# all combos in ~40 serial batches.
#
# Advantages over array jobs:
#   - 1 scheduler allocation instead of 160
#   - 1 conda activation, 1 filesystem wait
#   - Already-completed runs are automatically skipped
#   - Progress counter shows exactly where you are
#
# Setup (run ONCE):
#   conda env create -f wings_gpu.yml
#   conda activate wings-gpu
#
# Submit:
#   sbatch submit_wings.sh
#
# Quick test (single combo, 30 days):
#   sbatch --time=00:15:00 --gpus-per-task=1 \
#          --export=ALL,QUICK=1,COMBO_RANGE="0-0",NREPS=1 submit_wings.sh
#
# Custom range (e.g., only combos 0-3 with 5 reps):
#   sbatch --export=ALL,COMBO_RANGE="0-3",NREPS=5 submit_wings.sh
# ============================================================

# --- Paths ---
PROJECT_DIR="/projects/sciences/zoology/geurten_lab/wolbachia_spread_model"
CODE_DIR="/home/geuba03p/PyProjects/WINGS"
SCRIPT="${CODE_DIR}/gpu/gpu_simulation.py"
OUTDIR="${PROJECT_DIR}/gpu_results_${SLURM_JOB_ID}"

# --- Configuration (override via --export) ---
N_GPUS=${SLURM_GPUS_ON_NODE:-4}
NREPS=${NREPS:-10}             # replicates per combo
NCOMBOS=16                     # 2^4 effect combinations
COMBO_START=0
COMBO_END=15
if [ -n "${COMBO_RANGE}" ]; then
    COMBO_START=$(echo "${COMBO_RANGE}" | cut -d- -f1)
    COMBO_END=$(echo "${COMBO_RANGE}" | cut -d- -f2)
fi
DAYS=${DAYS:-365}
if [ "${QUICK}" = "1" ]; then
    DAYS=30
fi

TOTAL_RUNS=$(( (COMBO_END - COMBO_START + 1) * NREPS ))

# --- Wait for filesystem mount ---
sleep 5

# --- Activate environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wings-gpu

# --- Diagnostics ---
echo "============================================"
echo "  W.I.N.G.S. GPU Simulation"
echo "============================================"
echo "  Job ID:       ${SLURM_JOB_ID}"
echo "  Node:         $(hostname)"
echo "  GPUs:         ${N_GPUS}x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  CUDA driver:  $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
echo "  Python:       $(which python)"
echo "  PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A")' 2>&1)"
echo "  Output dir:   ${OUTDIR}"
echo "  Combos:       ${COMBO_START}–${COMBO_END} ($(( COMBO_END - COMBO_START + 1 )) combos)"
echo "  Replicates:   ${NREPS}"
echo "  Total runs:   ${TOTAL_RUNS}"
echo "  Days/run:     ${DAYS}"
echo "  GPUs in use:  ${N_GPUS} (parallel)"
echo "  Quick mode:   ${QUICK:-0}"
echo "============================================"
echo ""

mkdir -p "${OUTDIR}"

# --- Build the list of all (combo, replicate) pairs ---
declare -a TASKS=()
for COMBO_ID in $(seq ${COMBO_START} ${COMBO_END}); do
    for REP_ID in $(seq 0 $((NREPS - 1))); do
        # Decode combo ID to 4 binary flags
        CI=$(( (COMBO_ID >> 3) & 1 ))
        MK=$(( (COMBO_ID >> 2) & 1 ))
        ER=$(( (COMBO_ID >> 1) & 1 ))
        IE=$(( (COMBO_ID >> 0) & 1 ))

        # Build output filename (matches original WINGS naming convention)
        FNAME="cytoplasmic_incompatibility_$([ $CI -eq 1 ] && echo True || echo False)"
        FNAME="${FNAME}_male_killing_$([ $MK -eq 1 ] && echo True || echo False)"
        FNAME="${FNAME}_increased_exploration_rate_$([ $ER -eq 1 ] && echo True || echo False)"
        FNAME="${FNAME}_increased_eggs_$([ $IE -eq 1 ] && echo True || echo False)"
        FNAME="${FNAME}_${REP_ID}.csv"

        # Skip if already completed
        if [ -f "${OUTDIR}/${FNAME}" ]; then
            continue
        fi

        TASKS+=("${COMBO_ID}|${REP_ID}|${FNAME}")
    done
done

REMAINING=${#TASKS[@]}
SKIPPED=$((TOTAL_RUNS - REMAINING))

echo "  Skipping ${SKIPPED} already-completed runs."
echo "  Remaining: ${REMAINING} runs"
echo "  Estimated batches: $(( (REMAINING + N_GPUS - 1) / N_GPUS )) (${N_GPUS} parallel)"
echo "============================================"
echo ""

if [ ${REMAINING} -eq 0 ]; then
    echo "All runs already completed! Nothing to do."
    exit 0
fi

# --- Run tasks in parallel batches of N_GPUS ---
COMPLETED=0
START_TIME=$(date +%s)
TASK_IDX=0

while [ ${TASK_IDX} -lt ${REMAINING} ]; do
    PIDS=()
    BATCH_SIZE=0

    for GPU_SLOT in $(seq 0 $((N_GPUS - 1))); do
        IDX=$((TASK_IDX + GPU_SLOT))
        if [ ${IDX} -ge ${REMAINING} ]; then
            break
        fi

        # Parse task
        IFS='|' read -r COMBO_ID REP_ID FNAME <<< "${TASKS[$IDX]}"
        CI=$(( (COMBO_ID >> 3) & 1 ))
        MK=$(( (COMBO_ID >> 2) & 1 ))
        ER=$(( (COMBO_ID >> 1) & 1 ))
        IE=$(( (COMBO_ID >> 0) & 1 ))

        # Build effect flags
        FLAGS=""
        [ $CI -eq 1 ] && FLAGS="$FLAGS --ci"
        [ $MK -eq 1 ] && FLAGS="$FLAGS --mk"
        [ $ER -eq 1 ] && FLAGS="$FLAGS --er"
        [ $IE -eq 1 ] && FLAGS="$FLAGS --ie"

        SEED=$((42 + COMBO_ID * 100 + REP_ID))

        # Run on specific GPU via CUDA_VISIBLE_DEVICES
        echo "[$(date +%H:%M:%S)] Starting combo=${COMBO_ID} rep=${REP_ID} on GPU ${GPU_SLOT}  (CI=$CI MK=$MK ER=$ER IE=$IE)"
        CUDA_VISIBLE_DEVICES=${GPU_SLOT} python ${SCRIPT} \
            --population 50 \
            --max-pop 20000 \
            --grid-size 500 \
            --days ${DAYS} \
            --mortality cannibalism \
            --backend cell_list \
            --device cuda \
            --seed ${SEED} \
            ${FLAGS} \
            --output "${OUTDIR}/${FNAME}" \
            > "${OUTDIR}/${FNAME%.csv}.log" 2>&1 &

        PIDS+=($!)
        BATCH_SIZE=$((BATCH_SIZE + 1))
    done

    # Wait for this batch to finish
    for PID in "${PIDS[@]}"; do
        wait ${PID}
    done
    COMPLETED=$((COMPLETED + BATCH_SIZE))
    TASK_IDX=$((TASK_IDX + BATCH_SIZE))

    # Progress report
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))
    if [ ${COMPLETED} -gt 0 ]; then
        RATE=$(echo "scale=1; ${ELAPSED} / ${COMPLETED}" | bc)
        ETA=$(echo "scale=0; ${RATE} * (${REMAINING} - ${COMPLETED})" | bc)
        ETA_MIN=$(echo "scale=1; ${ETA} / 60" | bc)
    else
        RATE="?"
        ETA_MIN="?"
    fi
    echo "[$(date +%H:%M:%S)] ── Progress: ${COMPLETED}/${REMAINING} done (${RATE}s/run, ETA: ${ETA_MIN} min) ──"
done

# --- Summary ---
END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "============================================"
echo "  ALL DONE"
echo "============================================"
echo "  Completed: ${COMPLETED} runs"
echo "  Skipped:   ${SKIPPED} (already existed)"
echo "  Wall time: $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s"
echo "  Output:    ${OUTDIR}"
echo "  Files:     $(ls ${OUTDIR}/*.csv 2>/dev/null | wc -l) CSV files"
echo "============================================"
