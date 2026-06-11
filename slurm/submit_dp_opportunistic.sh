#!/usr/bin/env bash
# ============================================================
# W.I.N.G.S. — Δp sweep, opportunistic multi-partition submitter
# ============================================================
#  Lab opportunistic pattern (see CheatSheet "Aoraki Opportunistic GPU
#  Submission"): fire one single-GPU sbatch per (mu, phenotype, fraction)
#  cell and let SLURM bin-pack them across every GPU partition your
#  account can use. Set slurm_partition in local_paths.json to the full
#  opportunistic list, e.g.:
#     aoraki_gpu_H100,aoraki_gpu_A100_80GB,aoraki_gpu_A100_40GB,\
#     aoraki_gpu_RTX6000,aoraki_gpu_L40,aoraki_gpu_L4_24GB,aoraki_gpu_RTX3090
#  load_paths.sh exports it as SBATCH_PARTITION (+ SBATCH_ACCOUNT), so the
#  per-task script needs no hard-coded partition/account.
#
#  Usage:
#     source slurm/load_paths.sh                 # sets SBATCH_*, WINGS_*
#     bash slurm/submit_dp_opportunistic.sh --dry   # list what would fire
#     bash slurm/submit_dp_opportunistic.sh         # default: mu=0.03 only
#
#  mu=0 is the cost-free case and is NOT re-run here — reuse the existing
#  abm_delta_p/ sweep as the mu=0 baseline. Override with MU_LIST if you
#  ever do need to regenerate it: MU_LIST="0.0 0.03" bash ...
#
#  Resume gate: cells whose 200 CSVs already exist are skipped, so re-
#  running only queues the gaps.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "${SCRIPT_DIR}/.."

# Load machine paths (also exports SBATCH_ACCOUNT / SBATCH_PARTITION).
source "${SCRIPT_DIR}/load_paths.sh"
: "${WINGS_DATA_ROOT:?run 'source slurm/load_paths.sh' first}"
: "${SBATCH_PARTITION:?set slurm_partition (opportunistic list) in local_paths.json}"

MU_LIST=${MU_LIST:-"0.03"}     # mu=0 reused from existing abm_delta_p/, not re-run
PHENOS=("CI" "ER" "CI_ER")
FRACTIONS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
NREPS=${NREPS:-200}

DRY=false
[ "${1:-}" = "--dry" ] && DRY=true

echo "Partitions: ${SBATCH_PARTITION}"
echo "mu list:    ${MU_LIST}"
SUBMITTED=0; SKIPPED=0
for MU in ${MU_LIST}; do
    OUTDIR="${WINGS_DATA_ROOT}/abm_delta_p_mu${MU}"
    for PHENO in "${PHENOS[@]}"; do
        for FRAC in "${FRACTIONS[@]}"; do
            FRAC_STR=$(printf "%03d" "$(awk "BEGIN{printf \"%d\", ${FRAC}*100}")")
            # Cell complete if every replicate CSV is present
            have=$(ls "${OUTDIR}/${PHENO}_frac${FRAC_STR}_rep"*.csv 2>/dev/null | wc -l)
            if [ "${have}" -ge "${NREPS}" ]; then
                SKIPPED=$((SKIPPED + 1)); continue
            fi
            if $DRY; then
                echo "  would submit mu=${MU} ${PHENO} frac=${FRAC} (have ${have}/${NREPS})"
                continue
            fi
            sbatch \
                --export=ALL,MU="${MU}",PHENO="${PHENO}",FRAC="${FRAC}",NREPS="${NREPS}" \
                --job-name="wings_dp_mu${MU}_${PHENO}_f${FRAC_STR}" \
                slurm/dp_single.sh
            SUBMITTED=$((SUBMITTED + 1))
        done
    done
done

echo "Submitted: ${SUBMITTED}  Skipped (complete): ${SKIPPED}"
