#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=wings_lf_analyse
#SBATCH --output=logs/wings_lf_analyse_%j.out
#SBATCH --error=logs/wings_lf_analyse_%j.out
# ============================================================
#  W.I.N.G.S. — low-frequency meta-analysis (runs on a COMPUTE node)
# ------------------------------------------------------------
#  The analysis reads ~90 k trajectory CSVs and renders figures — a
#  CPU/IO job, but it must NOT run on the login node.  Submit it:
#
#     source slurm/load_paths.sh
#     sbatch -A "$SBATCH_ACCOUNT" -p aoraki_short slurm/analyse_lowfreq.sh
#
#  Or run interactively on a node with srun (see CheatSheet).  Output
#  lands in results/threshold_lowfreq/ on the shared filesystem; pull it
#  to the workstation with rsync (a transfer, login-safe).
# ============================================================
# No `set -u` — conda's MKL activation hook references unbound vars.
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

if [ -z "${WINGS_DATA_ROOT:-}" ]; then
    _here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
    [ -f "${_here}/load_paths.sh" ] && source "${_here}/load_paths.sh"
fi
: "${WINGS_DATA_ROOT:?run 'source slurm/load_paths.sh' before sbatch}"

source "${WINGS_CONDA_SETUP}"
conda activate "${WINGS_CONDA_ENV}"

mkdir -p logs results/threshold_lowfreq
echo "── analysing low-freq sweep on $(hostname) ──"
python -u -m wings.analysis.threshold_lowfreq \
    --data-root "${WINGS_DATA_ROOT}" \
    --outdir results/threshold_lowfreq
echo "── done ──"
