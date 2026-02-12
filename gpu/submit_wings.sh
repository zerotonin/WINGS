#!/bin/bash
#SBATCH --job-name=wings_gpu
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_L40
#SBATCH --gres=gpu:l40s:1          # request 1 L40S GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-160              # 16 effect combos × 10 replicates
#SBATCH --output=logs/wings_%A_%a.out
#SBATCH --error=logs/wings_%A_%a.err

# ---------------------------------------------------------------
# WINGS GPU Simulation  –  SLURM array job
#
# Maps SLURM_ARRAY_TASK_ID → (effect combination, replicate).
# 16 combinations × 10 replicates = 160 tasks.
# Each task runs independently on one GPU for ~1-5 min at N=20 000.
# ---------------------------------------------------------------

module load anaconda3          # adjust to your HPC module system
module load cuda/12.1          # or whichever CUDA version matches your PyTorch
conda activate wings

mkdir -p logs data/gpu_results

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

# Filename matching the original naming convention
FNAME="cytoplasmic_incompatibility_$([ $CI -eq 1 ] && echo True || echo False)"
FNAME="${FNAME}_male_killing_$([ $MK -eq 1 ] && echo True || echo False)"
FNAME="${FNAME}_increased_exploration_rate_$([ $ER -eq 1 ] && echo True || echo False)"
FNAME="${FNAME}_increased_eggs_$([ $IE -eq 1 ] && echo True || echo False)"
FNAME="${FNAME}_${REP_ID}.csv"

echo "Task $SLURM_ARRAY_TASK_ID: combo=$COMBO_ID rep=$REP_ID"
echo "  CI=$CI MK=$MK ER=$ER IE=$IE"
echo "  Output: data/gpu_results/$FNAME"

python gpu_simulation.py \
    --population 20000 \
    --max-pop 25000 \
    --grid-size 500 \
    --days 365 \
    --backend cell_list \
    --device cuda \
    --seed $((42 + SLURM_ARRAY_TASK_ID)) \
    $FLAGS \
    --output "data/gpu_results/$FNAME"

echo "Done."
