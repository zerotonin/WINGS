#!/bin/bash
#SBATCH --job-name=wings_fixed
# Account + partition come from local_paths.json via load_paths.sh
# (exported as SBATCH_ACCOUNT / SBATCH_PARTITION). Run
#   source slurm/load_paths.sh
# before sbatch, or override per run with: sbatch -A <acct> -p <part> ...
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

# --- Machine-specific paths (see slurm/load_paths.sh) ---
if [ -z "${WINGS_DATA_ROOT:-}" ]; then
    _here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
    if [ -f "${_here}/load_paths.sh" ]; then
        source "${_here}/load_paths.sh"
    elif [ -n "${WINGS_CODE_ROOT:-}" ] && [ -f "${WINGS_CODE_ROOT}/slurm/load_paths.sh" ]; then
        source "${WINGS_CODE_ROOT}/slurm/load_paths.sh"
    fi
fi
if [ -z "${WINGS_DATA_ROOT:-}" ]; then
    echo "ERROR: WINGS paths not loaded. Run 'source slurm/load_paths.sh' before sbatch." >&2
    exit 1
fi
CODE_DIR="${WINGS_CODE_ROOT}"
SCRIPT="${CODE_DIR}/wings/models/wfm.py"
OUTDIR="${WINGS_DATA_ROOT}/gpu_results_50beetles"
NREPS=${NREPS:-200}

# --- Activate environment ---
source "${WINGS_CONDA_SETUP}"
conda activate "${WINGS_CONDA_ENV}"

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
