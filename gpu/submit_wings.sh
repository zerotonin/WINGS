#!/bin/bash
#SBATCH --job-name=wings_gpu
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_L40,aoraki_gpu,aoraki_gpu_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --array=1-160
#SBATCH --output=wings_gpu_%A_%a.log
#SBATCH --error=wings_gpu_%A_%a.err
# ============================================================
# W.I.N.G.S. — GPU-Accelerated Wolbachia Spread Simulation
# ============================================================
#
# Setup (run ONCE before first submission):
#
#   conda env create -f wings_gpu.yml
#   conda activate wings-gpu
#   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
#
# Submit:
#   sbatch submit_wings.sh
#
# Quick test (single combo, 30 days):
#   sbatch --array=1 --time=00:15:00 --export=ALL,QUICK=1 submit_wings.sh
#
# Array layout: 16 effect combinations × 10 replicates = 160 tasks
# ============================================================

# --- Paths ---
PROJECT_DIR="/projects/sciences/zoology/geurten_lab/wolbachia_spread_model"
CODE_DIR="/home/geuba03p/PyProjects/WINGS"
SCRIPT="${CODE_DIR}/gpu/gpu_simulation.py"
OUTDIR="${PROJECT_DIR}/gpu_results_${SLURM_ARRAY_JOB_ID}"

# --- Wait for filesystem mount ---
sleep 5

# --- Activate environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wings-gpu

# --- Diagnostics ---
echo "============================================"
echo "Job ID:       ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "CUDA version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python:       $(which python)"
echo "Torch CUDA:   $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A")' 2>&1)"
echo "Output dir:   ${OUTDIR}"
echo "============================================"

mkdir -p "${OUTDIR}"

# --- Decode array task ID into combo + replicate ---
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))   # 0-indexed
COMBO_ID=$((TASK_ID / 10))              # 0..15
REP_ID=$((TASK_ID % 10))               # 0..9

# --- Map COMBO_ID to the 4 binary flags ---
#     bit 3 = CI, bit 2 = MK, bit 1 = ER, bit 0 = IE
CI=$(( (COMBO_ID >> 3) & 1 ))
MK=$(( (COMBO_ID >> 2) & 1 ))
ER=$(( (COMBO_ID >> 1) & 1 ))
IE=$(( (COMBO_ID >> 0) & 1 ))

# Build CLI flags
FLAGS=""
[ $CI -eq 1 ] && FLAGS="$FLAGS --ci"
[ $MK -eq 1 ] && FLAGS="$FLAGS --mk"
[ $ER -eq 1 ] && FLAGS="$FLAGS --er"
[ $IE -eq 1 ] && FLAGS="$FLAGS --ie"

# Filename matching the original WINGS naming convention
FNAME="cytoplasmic_incompatibility_$([ $CI -eq 1 ] && echo True || echo False)"
FNAME="${FNAME}_male_killing_$([ $MK -eq 1 ] && echo True || echo False)"
FNAME="${FNAME}_increased_exploration_rate_$([ $ER -eq 1 ] && echo True || echo False)"
FNAME="${FNAME}_increased_eggs_$([ $IE -eq 1 ] && echo True || echo False)"
FNAME="${FNAME}_${REP_ID}.csv"

echo "Task $SLURM_ARRAY_TASK_ID: combo=$COMBO_ID rep=$REP_ID"
echo "  CI=$CI MK=$MK ER=$ER IE=$IE"
echo "  Output: ${OUTDIR}/${FNAME}"
echo "============================================"

# --- Build command ---
DAYS=365
if [ "${QUICK}" = "1" ]; then
    DAYS=30
    echo ">>> QUICK MODE (30 days) <<<"
fi

CMD="python ${SCRIPT}"
CMD="${CMD} --population 50"
CMD="${CMD} --max-pop 25000"
CMD="${CMD} --grid-size 500"
CMD="${CMD} --days ${DAYS}"
CMD="${CMD} --backend cell_list"
CMD="${CMD} --device cuda"
CMD="${CMD} --seed $((42 + SLURM_ARRAY_TASK_ID))"
CMD="${CMD} ${FLAGS}"
CMD="${CMD} --output ${OUTDIR}/${FNAME}"

echo "Running: ${CMD}"
echo "============================================"

# --- Run ---
time ${CMD}

echo ""
echo "============================================"
echo "Done. Results in: ${OUTDIR}/${FNAME}"
echo "============================================"
