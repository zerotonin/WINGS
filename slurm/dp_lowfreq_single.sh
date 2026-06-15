#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=wings_lf
#SBATCH --output=logs/wings_lf_%j.out
#SBATCH --error=logs/wings_lf_%j.out
#SBATCH --requeue
# ============================================================
# W.I.N.G.S. — low-frequency CI-only Turelli-threshold per-task
# ============================================================
#  One (MU, FRAC) cell.  Splits NREPS replicates across SIMS_PER_GPU
#  concurrent *batched* processes (run_dp_batch), each importing torch
#  once — so the ~8 s startup is paid a handful of times per cell, not
#  once per replicate.  Account + opportunistic partition list come from
#  local_paths.json via load_paths.sh; nothing machine-specific here.
#
#  Injected by the submitter via --export:
#     MU (e.g. 0.03), FRAC (e.g. 0.005)
# ============================================================

# No `set -e` — a single transient sim failure must not tank the batch.
cd "${SLURM_SUBMIT_DIR:-.}"

if [ -z "${WINGS_DATA_ROOT:-}" ]; then
    _here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
    [ -f "${_here}/load_paths.sh" ] && source "${_here}/load_paths.sh"
fi
: "${WINGS_DATA_ROOT:?run 'source slurm/load_paths.sh' before sbatch}"

source "${WINGS_CONDA_SETUP}"
conda activate "${WINGS_CONDA_ENV}"

MU=${MU:?MU must be set}
FRAC=${FRAC:?FRAC must be set}
NREPS=${NREPS:-500}
# One sim saturates the GPU (brute O(N^2) mating at N=20000), so packing
# several per GPU only slows each one down and overruns the wall. Keep 1.
SIMS_PER_GPU=${SIMS_PER_GPU:-1}
REP_START=${REP_START:-0}
REP_END=${REP_END:-$((NREPS - 1))}
DAYS=${DAYS:-72}
POP=${POP:-2000}

OUTDIR="${WINGS_DATA_ROOT}/abm_dp_lowfreq_mu${MU}"
mkdir -p "${OUTDIR}" logs

echo "── Job ${SLURM_JOB_ID:-local}: mu=${MU} frac=${FRAC} reps=${REP_START}-${REP_END} days=${DAYS} pop=${POP} ──"
echo "── Partition=${SLURM_JOB_PARTITION:-?}  node=$(hostname)  procs=${SIMS_PER_GPU} ──"

# Split [REP_START, REP_END] into SIMS_PER_GPU contiguous blocks.
TOTAL_REPS=$((REP_END - REP_START + 1))
CHUNK=$(( (TOTAL_REPS + SIMS_PER_GPU - 1) / SIMS_PER_GPU ))
PIDS=()
for ((START = REP_START; START <= REP_END; START += CHUNK)); do
    END=$((START + CHUNK - 1))
    [ "${END}" -gt "${REP_END}" ] && END=${REP_END}
    python -m wings.models.run_dp_batch \
        --mu "${MU}" --frac "${FRAC}" \
        --rep-start "${START}" --rep-end "${END}" \
        --population "${POP}" --days "${DAYS}" \
        --outdir "${OUTDIR}" --device cuda \
        > "${OUTDIR}/.log_${FRAC}_${START}-${END}.txt" 2>&1 &
    PIDS+=($!)
done
wait "${PIDS[@]}"

echo "── done: $(ls "${OUTDIR}"/CI_frac*_rep*.csv 2>/dev/null | wc -l) CSVs in cell dir ──"
