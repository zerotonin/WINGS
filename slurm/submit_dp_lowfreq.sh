#!/usr/bin/env bash
# ============================================================
# W.I.N.G.S. — low-frequency CI-only Turelli-threshold submitter
# ============================================================
#  Tests whether the ABM reproduces the WFM threshold p̂ ≈ µ by seeding
#  CI-only infections densely across p = 0.5–5 % (where p̂ lives for
#  µ ≤ 0.05) and measuring the initial Δp.  One opportunistic single-GPU
#  job per (µ, fraction) cell; each cell runs NREPS batched replicates.
#
#  Account + opportunistic partition list come from local_paths.json via
#  load_paths.sh (SBATCH_ACCOUNT / SBATCH_PARTITION).  After submission the
#  gated RTX6000 partition is re-added to pending jobs (see CheatSheet §2.1).
#
#  Usage:
#     source slurm/load_paths.sh
#     bash slurm/submit_dp_lowfreq.sh --dry
#     bash slurm/submit_dp_lowfreq.sh
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "${SCRIPT_DIR}/.."
source "${SCRIPT_DIR}/load_paths.sh"
: "${WINGS_DATA_ROOT:?run 'source slurm/load_paths.sh' first}"
: "${SBATCH_PARTITION:?set slurm_partition (opportunistic list) in local_paths.json}"

MU_LIST=${MU_LIST:-"0.01 0.02 0.03 0.04 0.05"}
FRACTIONS=(0.005 0.010 0.015 0.020 0.025 0.030 0.035 0.040 0.045 0.050 0.055 0.060)
NREPS=${NREPS:-500}
CHUNK_SIZE=${CHUNK_SIZE:-150}   # reps/job; 150 × ~14 s ≈ 35 min, well under the 1:30 wall

DRY=false
[ "${1:-}" = "--dry" ] && DRY=true

echo "Partitions: ${SBATCH_PARTITION}"
echo "mu list:    ${MU_LIST}"
echo "fractions:  ${FRACTIONS[*]}"
echo "reps/cell:  ${NREPS}  chunk=${CHUNK_SIZE}  (CI-only, 72-day, pop 2000, 1 sim/GPU)"

SUBMITTED=0; SKIPPED=0
for MU in ${MU_LIST}; do
    OUTDIR="${WINGS_DATA_ROOT}/abm_dp_lowfreq_mu${MU}"
    for FRAC in "${FRACTIONS[@]}"; do
        PERMILLE=$(printf "%04d" "$(awk "BEGIN{printf \"%d\", ${FRAC}*1000}")")
        for ((START = 0; START < NREPS; START += CHUNK_SIZE)); do
            END=$((START + CHUNK_SIZE - 1))
            [ "${END}" -ge "${NREPS}" ] && END=$((NREPS - 1))
            # Chunk-level resume: skip if every rep CSV in [START, END] exists
            need=$((END - START + 1)); have=0
            for ((r = START; r <= END; r++)); do
                [ -s "${OUTDIR}/CI_frac${PERMILLE}_rep${r}.csv" ] && have=$((have + 1))
            done
            if [ "${have}" -ge "${need}" ]; then
                SKIPPED=$((SKIPPED + 1)); continue
            fi
            if $DRY; then
                echo "  would submit mu=${MU} frac=${FRAC} reps ${START}-${END} (have ${have}/${need})"
                continue
            fi
            sbatch \
                --export=ALL,MU="${MU}",FRAC="${FRAC}",NREPS="${NREPS}",REP_START="${START}",REP_END="${END}" \
                --job-name="wings_lf_mu${MU}_f${PERMILLE}_r${START}-${END}" \
                slurm/dp_lowfreq_single.sh
            SUBMITTED=$((SUBMITTED + 1))
        done
    done
done

echo "Submitted: ${SUBMITTED}  Skipped (complete): ${SKIPPED}"

# Reclaim the gated RTX6000 partition (stripped at submit; see CheatSheet §2.1).
if ! $DRY && [ "${SUBMITTED}" -gt 0 ]; then
    sleep 5
    READDED=0
    for jid in $(squeue -u "${USER}" -h -t PENDING -o '%i %j' 2>/dev/null \
                 | awk '$2 ~ /^wings_lf_/ {print $1}'); do
        scontrol update jobid="${jid}" partition="${SBATCH_PARTITION}" 2>/dev/null \
            && READDED=$((READDED + 1))
    done
    echo "Re-added RTX6000 to ${READDED} pending wings_lf jobs."
fi
