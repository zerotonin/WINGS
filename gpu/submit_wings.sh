#!/bin/bash
#SBATCH --job-name=wings_gpu
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_L40,aoraki_gpu,aoraki_gpu_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --mem=64GB
#SBATCH --output=wings_gpu_%j.log
#SBATCH --error=wings_gpu_%j.err
# ============================================================
# W.I.N.G.S. — GPU-Accelerated Wolbachia Spread Simulation
# ============================================================
#
# GPU Packing Strategy
# --------------------
# DCGM profiling shows each simulation uses only ~16% SM and
# ~550 MB VRAM.  So we pack SIMS_PER_GPU concurrent simulations
# on each GPU (default: 6), giving 4 GPUs × 6 = 24 sims in
# parallel.  This raises effective GPU utilization to ~90%+ and
# finishes 160 runs in ~7 batches instead of ~40.
#
# Setup (run ONCE):
#   conda env create -f wings_gpu.yml
#   conda activate wings-gpu
#
# Submit:
#   sbatch submit_wings.sh
#
# Quick test:
#   sbatch --time=00:15:00 --gpus-per-task=1 \
#          --export=ALL,QUICK=1,COMBO_RANGE="0-0",NREPS=1,SIMS_PER_GPU=1 submit_wings.sh
#
# Tune packing density (e.g. on H100 with more VRAM):
#   sbatch --export=ALL,SIMS_PER_GPU=8 submit_wings.sh
# ============================================================

# --- Paths ---
PROJECT_DIR="/projects/sciences/zoology/geurten_lab/wolbachia_spread_model"
CODE_DIR="/home/geuba03p/PyProjects/WINGS"
SCRIPT="${CODE_DIR}/gpu/gpu_simulation.py"
OUTDIR="${PROJECT_DIR}/gpu_results_${SLURM_JOB_ID}"

# --- Configuration (override via --export) ---
N_GPUS=${SLURM_GPUS_ON_NODE:-4}
SIMS_PER_GPU=${SIMS_PER_GPU:-6}       # concurrent sims sharing one GPU
PARALLEL=$((N_GPUS * SIMS_PER_GPU))   # total parallel slots

NREPS=${NREPS:-10}
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
echo ""
echo "  ── Schedule ──"
echo "  Combos:         ${COMBO_START}–${COMBO_END} ($(( COMBO_END - COMBO_START + 1 )) combos)"
echo "  Replicates:     ${NREPS}"
echo "  Total runs:     ${TOTAL_RUNS}"
echo "  GPUs:           ${N_GPUS}"
echo "  Sims per GPU:   ${SIMS_PER_GPU}"
echo "  Parallel slots: ${PARALLEL}"
echo "  Days/run:       ${DAYS}"
echo "  Quick mode:     ${QUICK:-0}"
echo "============================================"
echo ""

mkdir -p "${OUTDIR}"

# --- Build the list of all (combo, replicate) pairs ---
declare -a TASKS=()
for COMBO_ID in $(seq ${COMBO_START} ${COMBO_END}); do
    for REP_ID in $(seq 0 $((NREPS - 1))); do
        CI=$(( (COMBO_ID >> 3) & 1 ))
        MK=$(( (COMBO_ID >> 2) & 1 ))
        ER=$(( (COMBO_ID >> 1) & 1 ))
        IE=$(( (COMBO_ID >> 0) & 1 ))

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
N_BATCHES=$(( (REMAINING + PARALLEL - 1) / PARALLEL ))

echo "  Skipping:       ${SKIPPED} already-completed runs"
echo "  Remaining:      ${REMAINING} runs"
echo "  Batches:        ${N_BATCHES} (${PARALLEL} sims/batch)"
echo "============================================"
echo ""

if [ ${REMAINING} -eq 0 ]; then
    echo "All runs already completed! Nothing to do."
    exit 0
fi

# --- Run tasks in parallel batches ---
COMPLETED=0
START_TIME=$(date +%s)
TASK_IDX=0
BATCH_NUM=0

while [ ${TASK_IDX} -lt ${REMAINING} ]; do
    PIDS=()
    BATCH_SIZE=0
    BATCH_NUM=$((BATCH_NUM + 1))

    echo "[$(date +%H:%M:%S)] ── Batch ${BATCH_NUM}/${N_BATCHES} ──"

    for SLOT in $(seq 0 $((PARALLEL - 1))); do
        IDX=$((TASK_IDX + SLOT))
        if [ ${IDX} -ge ${REMAINING} ]; then
            break
        fi

        # Assign this slot to a GPU (round-robin)
        GPU_ID=$((SLOT % N_GPUS))

        # Parse task
        IFS='|' read -r COMBO_ID REP_ID FNAME <<< "${TASKS[$IDX]}"
        CI=$(( (COMBO_ID >> 3) & 1 ))
        MK=$(( (COMBO_ID >> 2) & 1 ))
        ER=$(( (COMBO_ID >> 1) & 1 ))
        IE=$(( (COMBO_ID >> 0) & 1 ))

        FLAGS=""
        [ $CI -eq 1 ] && FLAGS="$FLAGS --ci"
        [ $MK -eq 1 ] && FLAGS="$FLAGS --mk"
        [ $ER -eq 1 ] && FLAGS="$FLAGS --er"
        [ $IE -eq 1 ] && FLAGS="$FLAGS --ie"

        SEED=$((42 + COMBO_ID * 100 + REP_ID))

        echo "  slot ${SLOT}: combo=${COMBO_ID} rep=${REP_ID} → GPU ${GPU_ID}  (CI=$CI MK=$MK ER=$ER IE=$IE)"

        CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT} \
            --population 50 \
            --max-pop 20000 \
            --grid-size 500 \
            --days ${DAYS} \
            --mortality cannibalism \
            --backend brute \
            --device cuda \
            --seed ${SEED} \
            ${FLAGS} \
            --output "${OUTDIR}/${FNAME}" \
            > "${OUTDIR}/${FNAME%.csv}.log" 2>&1 &

        PIDS+=($!)
        BATCH_SIZE=$((BATCH_SIZE + 1))
    done

    # Wait for entire batch
    FAILED=0
    for PID in "${PIDS[@]}"; do
        wait ${PID} || FAILED=$((FAILED + 1))
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
    FAIL_MSG=""
    if [ ${FAILED} -gt 0 ]; then
        FAIL_MSG="  (${FAILED} FAILED — check logs)"
    fi
    echo "[$(date +%H:%M:%S)] ── Progress: ${COMPLETED}/${REMAINING} done | ${RATE}s/run | ETA: ${ETA_MIN} min${FAIL_MSG} ──"
    echo ""
done

# --- Summary ---
END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( END_TIME - START_TIME ))
N_CSV=$(ls "${OUTDIR}"/*.csv 2>/dev/null | wc -l)
echo "============================================"
echo "  ALL DONE"
echo "============================================"
echo "  Completed: ${COMPLETED} runs in this session"
echo "  Skipped:   ${SKIPPED} (already existed)"
echo "  Wall time: $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s"
echo "  Output:    ${OUTDIR}"
echo "  CSV files: ${N_CSV}"
echo "============================================"
