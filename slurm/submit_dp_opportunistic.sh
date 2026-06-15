#!/usr/bin/env bash
# ============================================================
# W.I.N.G.S. — Δp sweep, opportunistic multi-partition submitter
# ============================================================
#  Lab opportunistic pattern (see CheatSheet "Aoraki Opportunistic GPU
#  Submission"): fire single-GPU sbatch jobs and let SLURM bin-pack them
#  across every GPU partition your account can use. Each (mu, phenotype,
#  fraction) cell is split into CHUNK_SIZE-replicate jobs so every job
#  finishes inside the wall-clock instead of timing out at 200 reps.
#  Set slurm_partition in local_paths.json to the full opportunistic
#  list, e.g.:
#     aoraki_gpu_H100,aoraki_gpu_A100_80GB,aoraki_gpu_A100_40GB,\
#     aoraki_gpu_RTX6000,aoraki_gpu_L40,aoraki_gpu_L4_24GB,aoraki_gpu_RTX3090
#  load_paths.sh exports it as SBATCH_PARTITION (+ SBATCH_ACCOUNT), so the
#  per-task script needs no hard-coded partition/account.
#
#  Usage:
#     source slurm/load_paths.sh                 # sets SBATCH_*, WINGS_*
#     bash slurm/submit_dp_opportunistic.sh --dry   # list what would fire
#     bash slurm/submit_dp_opportunistic.sh         # default: mu in {0.01..0.05}
#
#  Default mu grid is {0.01, 0.02, 0.03, 0.04, 0.05}. mu=0 is the cost-free
#  case and is NOT re-run — reuse the existing abm_delta_p/ sweep as the
#  mu=0 baseline. Override the grid with MU_LIST, e.g. MU_LIST="0.03" bash ...
#
#  Resume gate: replicate chunks whose CSVs already exist are skipped, so
#  re-running only queues the gaps (per-rep, composes with partial cells).
#
#  After submission the gated RTX6000 partition (aoraki45/46) is re-added
#  to every pending job: it is silently stripped at submit because its
#  AllowQos excludes our default QOS, but queued jobs carry gpu_unlimited
#  which IS allowed there. See CheatSheet § 2.1.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "${SCRIPT_DIR}/.."

# Load machine paths (also exports SBATCH_ACCOUNT / SBATCH_PARTITION).
source "${SCRIPT_DIR}/load_paths.sh"
: "${WINGS_DATA_ROOT:?run 'source slurm/load_paths.sh' first}"
: "${SBATCH_PARTITION:?set slurm_partition (opportunistic list) in local_paths.json}"

MU_LIST=${MU_LIST:-"0.01 0.02 0.03 0.04 0.05"}   # mu=0 reused from abm_delta_p/, not re-run
PHENOS=("CI" "ER" "CI_ER")
FRACTIONS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
NREPS=${NREPS:-200}
CHUNK_SIZE=${CHUNK_SIZE:-40}   # reps per job; 40 ≈ 55 min on the slowest GPU, well under the 1:30 wall

DRY=false
[ "${1:-}" = "--dry" ] && DRY=true

echo "Partitions: ${SBATCH_PARTITION}"
echo "mu list:    ${MU_LIST}"
echo "chunking:   ${CHUNK_SIZE} reps/job  (${NREPS} reps/cell)"
SUBMITTED=0; SKIPPED=0
for MU in ${MU_LIST}; do
    OUTDIR="${WINGS_DATA_ROOT}/abm_delta_p_mu${MU}"
    for PHENO in "${PHENOS[@]}"; do
        for FRAC in "${FRACTIONS[@]}"; do
            FRAC_STR=$(printf "%03d" "$(awk "BEGIN{printf \"%d\", ${FRAC}*100}")")
            for ((START = 0; START < NREPS; START += CHUNK_SIZE)); do
                END=$((START + CHUNK_SIZE - 1))
                [ "${END}" -ge "${NREPS}" ] && END=$((NREPS - 1))
                # Chunk complete if every replicate CSV in [START, END] is present
                need=$((END - START + 1)); have=0
                for ((r = START; r <= END; r++)); do
                    [ -s "${OUTDIR}/${PHENO}_frac${FRAC_STR}_rep${r}.csv" ] && have=$((have + 1))
                done
                if [ "${have}" -ge "${need}" ]; then
                    SKIPPED=$((SKIPPED + 1)); continue
                fi
                if $DRY; then
                    echo "  would submit mu=${MU} ${PHENO} frac=${FRAC} reps ${START}-${END} (have ${have}/${need})"
                    continue
                fi
                sbatch \
                    --export=ALL,MU="${MU}",PHENO="${PHENO}",FRAC="${FRAC}",NREPS="${NREPS}",REP_START="${START}",REP_END="${END}" \
                    --job-name="wings_dp_mu${MU}_${PHENO}_f${FRAC_STR}_r${START}-${END}" \
                    slurm/dp_single.sh
                SUBMITTED=$((SUBMITTED + 1))
            done
        done
    done
done

echo "Submitted: ${SUBMITTED}  Skipped (complete chunks): ${SKIPPED}"

# ── Reclaim the gated RTX6000 partition (aoraki45/46) ───────────────────
# A fresh submit silently strips it (AllowQos excludes our default QOS).
# Queued jobs carry gpu_unlimited, which IS allowed there, so scontrol
# re-adds it now — getting us onto those GPUs earlier. See CheatSheet § 2.1.
if ! $DRY && [ "${SUBMITTED}" -gt 0 ]; then
    sleep 5   # let the scheduler register the freshly-submitted jobs
    READDED=0
    for jid in $(squeue -u "${USER}" -h -t PENDING -o '%i %j' 2>/dev/null \
                 | awk '$2 ~ /^wings_dp_/ {print $1}'); do
        scontrol update jobid="${jid}" partition="${SBATCH_PARTITION}" 2>/dev/null \
            && READDED=$((READDED + 1))
    done
    echo "Re-added RTX6000 to ${READDED} pending wings_dp jobs."
fi
