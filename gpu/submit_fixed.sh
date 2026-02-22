#!/bin/bash
#SBATCH --job-name=wings_fixed
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem=1GB
#SBATCH --output=wings_fixed_%j.log
#SBATCH --error=wings_fixed_%j.err
# ============================================================
# W.I.N.G.S. — Fixed-size discrete-generation model
# ============================================================
#
# Wright-Fisher: 50 adults per generation, 12 generations (≈ 1 year).
# 16 combos × 200 reps = 3200 runs.  Takes ~5 seconds on 1 CPU core.
#
# No GPU needed — pure NumPy on 50 beetles is instant.
#
# Submit:
#   sbatch submit_fixed.sh
#
# Or just run interactively (it's fast enough):
#   python fixed_generation_sim.py --run-all --nreps 200
# ============================================================

CODE_DIR="/home/geuba03p/PyProjects/WINGS"
SCRIPT="${CODE_DIR}/gpu/fixed_generation_sim.py"
OUTDIR="/projects/sciences/zoology/geurten_lab/wolbachia_spread_model/gpu_results_50beetles"
NREPS=${NREPS:-200}

# --- Activate environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wings-gpu

echo "============================================"
echo "  W.I.N.G.S. Fixed-Size Model"
echo "============================================"
echo "  Node:     $(hostname)"
echo "  Python:   $(which python)"
echo "  Script:   ${SCRIPT}"
echo "  Output:   ${OUTDIR}"
echo "  Reps:     ${NREPS}"
echo "============================================"
echo ""

mkdir -p "${OUTDIR}"

python "${SCRIPT}" \
    --run-all \
    --nreps ${NREPS} \
    --population 50 \
    --max-generations 12 \
    --initial-infection-freq 0.5 \
    --outdir "${OUTDIR}"

echo ""
echo "  Files produced: $(ls ${OUTDIR}/*.csv 2>/dev/null | wc -l)"
echo "  Done."
