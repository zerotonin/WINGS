#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=wings_dp
#SBATCH --output=logs/wings_dp_%j.out
#SBATCH --error=logs/wings_dp_%j.err
# ============================================================
# W.I.N.G.S. — Δp opportunistic per-task (one mu × phenotype × fraction)
# ============================================================
#  Runs NREPS sims for a single (MU, PHENO, FRAC) cell, packing
#  SIMS_PER_GPU concurrently on one GPU.  Account and the multi-partition
#  opportunistic list come from local_paths.json via load_paths.sh
#  (SBATCH_ACCOUNT / SBATCH_PARTITION) — set slurm_partition there to the
#  opportunistic list (see the cheat sheet).  Nothing is hard-coded here.
#
#  Injected by the submitter via --export:
#     MU (e.g. 0.03), PHENO (CI|ER|CI_ER), FRAC (e.g. 0.10)
#
#  Run one by hand:
#     source slurm/load_paths.sh
#     sbatch --export=ALL,MU=0.03,PHENO=CI_ER,FRAC=0.10 slurm/dp_single.sh
# ============================================================

# NB: no `set -e` at the top — a single transient failure must not tank
# the whole packed batch (lab convention).
cd "${SLURM_SUBMIT_DIR:-.}"

# --- Machine paths (account/partition already consumed by sbatch) ---
if [ -z "${WINGS_DATA_ROOT:-}" ]; then
    _here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
    if [ -f "${_here}/load_paths.sh" ]; then
        source "${_here}/load_paths.sh"
    elif [ -n "${WINGS_CODE_ROOT:-}" ] && [ -f "${WINGS_CODE_ROOT}/slurm/load_paths.sh" ]; then
        source "${WINGS_CODE_ROOT}/slurm/load_paths.sh"
    fi
fi
: "${WINGS_DATA_ROOT:?run 'source slurm/load_paths.sh' before sbatch}"

source "${WINGS_CONDA_SETUP}"
conda activate "${WINGS_CONDA_ENV}"

SCRIPT="${WINGS_CODE_ROOT}/wings/models/gpu_abm.py"
MU=${MU:?MU must be set}
PHENO=${PHENO:?PHENO must be set (CI|ER|CI_ER)}
FRAC=${FRAC:?FRAC must be set}
NREPS=${NREPS:-200}
SIMS_PER_GPU=${SIMS_PER_GPU:-6}
DAYS=${DAYS:-365}

OUTDIR="${WINGS_DATA_ROOT}/abm_delta_p_mu${MU}"
mkdir -p "${OUTDIR}" logs

# Phenotype → flags + stable index (order must match the submitter)
PHENO_LABELS=("CI" "ER" "CI_ER")
case "${PHENO}" in
    CI)    FLAGS="--ci";        PHENO_ID=0 ;;
    ER)    FLAGS="--er";        PHENO_ID=1 ;;
    CI_ER) FLAGS="--ci --er";   PHENO_ID=2 ;;
    *) echo "unknown PHENO=${PHENO}"; exit 1 ;;
esac

# Fraction → 3-digit tag and index (matches submit_delta_p.sh seeding)
FRACTIONS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
NFRAC=${#FRACTIONS[@]}
FRAC_STR=$(printf "%03d" "$(awk "BEGIN{printf \"%d\", ${FRAC}*100}")")
FRAC_ID=-1
for i in "${!FRACTIONS[@]}"; do
    [ "${FRACTIONS[$i]}" = "${FRAC}" ] && FRAC_ID=$i
done
[ "${FRAC_ID}" -lt 0 ] && { echo "unknown FRAC=${FRAC}"; exit 1; }
CONDITION_ID=$((PHENO_ID * NFRAC + FRAC_ID))

echo "── Job ${SLURM_JOB_ID:-local}: mu=${MU} pheno=${PHENO} frac=${FRAC} ──"
echo "── Partition=${SLURM_JOB_PARTITION:-?}  node=$(hostname)  reps=${NREPS} ──"

PIDS=(); LAUNCHED=0; SKIPPED=0
for REP_ID in $(seq 0 $((NREPS - 1))); do
    FNAME="${PHENO}_frac${FRAC_STR}_rep${REP_ID}.csv"
    if [ -s "${OUTDIR}/${FNAME}" ]; then
        SKIPPED=$((SKIPPED + 1)); continue
    fi
    SEED=$((42 + CONDITION_ID * 1000 + REP_ID))
    python "${SCRIPT}" \
        --population 50 --max-pop 20000 --grid-size 500 \
        --days "${DAYS}" --infected-fraction "${FRAC}" --mu "${MU}" \
        --mortality cannibalism --backend brute --device cuda \
        --seed "${SEED}" ${FLAGS} \
        --output "${OUTDIR}/${FNAME}" \
        > "${OUTDIR}/${FNAME%.csv}.log" 2>&1 &
    PIDS+=($!); LAUNCHED=$((LAUNCHED + 1))
    # Throttle to SIMS_PER_GPU concurrent
    if [ "${#PIDS[@]}" -ge "${SIMS_PER_GPU}" ]; then
        wait "${PIDS[@]}"; PIDS=()
    fi
done
[ "${#PIDS[@]}" -gt 0 ] && wait "${PIDS[@]}"

echo "── done: launched=${LAUNCHED} skipped=${SKIPPED} ──"
