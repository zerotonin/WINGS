#!/bin/bash
#SBATCH --job-name=wings_gpu
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_L40,aoraki_gpu,aoraki_gpu_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=40:00:00
#SBATCH --mem=16GB
#SBATCH --array=0-26
#SBATCH --output=wings_gpu_%A_%a.log
#SBATCH --error=wings_gpu_%A_%a.err
# ============================================================
# W.I.N.G.S. — GPU Simulation  (Array + Packing Hybrid)
# ============================================================
#
# Strategy: best of both worlds.
#
#   - Array job: each task requests only 1 GPU → schedules fast.
#   - Packing:   each task runs 6 sims concurrently on that GPU
#                (each sim uses ~16% SM / ~550 MB VRAM).
#
#   160 total runs  ÷  6 per task  =  27 array tasks (0–26)
#   Each task takes ~10–20 min for 365 days.
#   Up to 4 tasks run simultaneously (your allocation limit).
#
# Setup (run ONCE):
#   conda env create -f wings_gpu.yml
#
# Submit:
#   sbatch submit_wings.sh
#
# Quick test (1 task, 30 days):
#   sbatch --array=0 --time=00:10:00 \
#          --export=ALL,QUICK=1 submit_wings.sh
#
# Custom packing (e.g. 4 sims/GPU if memory-tight):
#   sbatch --export=ALL,SIMS_PER_GPU=4 submit_wings.sh
# ============================================================

# --- Paths ---
PROJECT_DIR="/projects/sciences/zoology/geurten_lab/wolbachia_spread_model"
CODE_DIR="/home/geuba03p/PyProjects/WINGS"
SCRIPT="${CODE_DIR}/gpu/gpu_simulation.py"
OUTDIR="${PROJECT_DIR}/gpu_results"

# --- Configuration ---
SIMS_PER_GPU=${SIMS_PER_GPU:-6}
NREPS=${NREPS:-200}
NCOMBOS=16
TOTAL_RUNS=$((NCOMBOS * NREPS))    # 160
DAYS=${DAYS:-365}
if [ "${QUICK}" = "1" ]; then
    DAYS=30
fi

# --- Wait for filesystem mount ---
sleep 5

# --- Activate environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wings-gpu

# --- Which runs does THIS array task own? ---
# Task 0 gets runs 0–5, task 1 gets 6–11, ..., task 26 gets 156–159
TASK_ID=${SLURM_ARRAY_TASK_ID}
RUN_START=$((TASK_ID * SIMS_PER_GPU))
RUN_END=$((RUN_START + SIMS_PER_GPU - 1))
# Clamp to total runs
if [ ${RUN_END} -ge ${TOTAL_RUNS} ]; then
    RUN_END=$((TOTAL_RUNS - 1))
fi

# --- Diagnostics ---
echo "============================================"
echo "  W.I.N.G.S. GPU Simulation"
echo "============================================"
echo "  Array job:    ${SLURM_ARRAY_JOB_ID}"
echo "  Array task:   ${TASK_ID} / $((TOTAL_RUNS / SIMS_PER_GPU))"
echo "  Node:         $(hostname)"
echo "  GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  VRAM:         $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "  Python:       $(which python)"
echo "  Output dir:   ${OUTDIR}"
echo ""
echo "  Runs:         ${RUN_START}–${RUN_END} ($(( RUN_END - RUN_START + 1 )) sims packed on 1 GPU)"
echo "  Days/run:     ${DAYS}"
echo "  Sims per GPU: ${SIMS_PER_GPU}"
echo "  Quick mode:   ${QUICK:-0}"
echo "============================================"
echo ""

mkdir -p "${OUTDIR}"

# --- Map run index → (combo, replicate) and launch ---
PIDS=()
FNAMES=()
LAUNCHED=0
SKIPPED=0

for RUN_IDX in $(seq ${RUN_START} ${RUN_END}); do
    COMBO_ID=$((RUN_IDX / NREPS))
    REP_ID=$((RUN_IDX % NREPS))

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
        echo "  [skip] ${FNAME} (already exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    FLAGS=""
    [ $CI -eq 1 ] && FLAGS="$FLAGS --ci"
    [ $MK -eq 1 ] && FLAGS="$FLAGS --mk"
    [ $ER -eq 1 ] && FLAGS="$FLAGS --er"
    [ $IE -eq 1 ] && FLAGS="$FLAGS --ie"

    SEED=$((42 + COMBO_ID * 100 + REP_ID))

    echo "  [launch] run=${RUN_IDX} combo=${COMBO_ID} rep=${REP_ID}  CI=$CI MK=$MK ER=$ER IE=$IE"

    python ${SCRIPT} \
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
    FNAMES+=("${FNAME}")
    LAUNCHED=$((LAUNCHED + 1))
done

echo ""
echo "  Launched: ${LAUNCHED}  |  Skipped: ${SKIPPED}"
echo "  Waiting for ${LAUNCHED} concurrent simulations..."
echo ""

# --- Wait for all and report ---
FAILED=0
T_START=$(date +%s)
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    FNAME=${FNAMES[$i]}
    if wait ${PID}; then
        echo "  [done]   ${FNAME}"
    else
        echo "  [FAILED] ${FNAME} (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
done
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

echo ""
echo "============================================"
echo "  Task ${TASK_ID} complete"
echo "============================================"
echo "  Launched:  ${LAUNCHED}"
echo "  Succeeded: $((LAUNCHED - FAILED))"
echo "  Failed:    ${FAILED}"
echo "  Skipped:   ${SKIPPED}"
echo "  Wall time: $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "============================================"
