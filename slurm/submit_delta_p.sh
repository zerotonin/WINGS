#!/bin/bash
#SBATCH --job-name=wings_dp
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_L40,aoraki_gpu,aoraki_gpu_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --output=wings_dp_%A_%a.log
#SBATCH --error=wings_dp_%A_%a.err
# ============================================================
# W.I.N.G.S. — Δp sweep (smart two-phase submission)
# ============================================================
#
# Phase 1 (interactive): Scans OUTDIR, writes manifest of missing
#          runs, calculates exact array size, submits only what's
#          needed.  No GPU reservation wasted.
#
# Phase 2 (SLURM):       Each array task reads the manifest and
#          runs its assigned sims (6 per GPU).
#
# Usage:
#   bash submit_delta_p.sh              # scan → submit missing
#   DRY_RUN=1 bash submit_delta_p.sh    # just list what's missing
#   QUICK=1 bash submit_delta_p.sh      # 30-day test runs
# ============================================================

# --- Paths ---
PROJECT_DIR="/projects/sciences/zoology/geurten_lab/wolbachia_spread_model"
CODE_DIR="/home/geuba03p/PyProjects/WINGS"
SCRIPT="${CODE_DIR}/wings/models/gpu_abm.py"
OUTDIR="${PROJECT_DIR}/abm_delta_p"
MANIFEST="${OUTDIR}/.missing_manifest.txt"

# --- Configuration ---
SIMS_PER_GPU=${SIMS_PER_GPU:-6}
NREPS=${NREPS:-200}
DAYS=${DAYS:-365}
DRY_RUN=${DRY_RUN:-0}
if [ "${QUICK}" = "1" ]; then
    DAYS=30
fi

# Phenotype definitions
PHENO_LABELS=("CI" "ER" "CI_ER")
PHENO_FLAGS=("--ci" "--er" "--ci --er")
NPHENO=${#PHENO_LABELS[@]}

# Infection fractions
FRACTIONS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
NFRAC=${#FRACTIONS[@]}

# ============================================================
# PHASE 1: Scan & Submit (runs when NOT inside SLURM)
# ============================================================
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then

    echo "============================================"
    echo "  W.I.N.G.S. — Δp Sweep (Phase 1: Scan)"
    echo "============================================"

    mkdir -p "${OUTDIR}"

    # --- Scan for missing files ---
    > "${MANIFEST}"
    N_MISSING=0
    N_EXIST=0

    for PHENO_ID in $(seq 0 $((NPHENO - 1))); do
        LABEL=${PHENO_LABELS[$PHENO_ID]}

        for FRAC_ID in $(seq 0 $((NFRAC - 1))); do
            FRAC=${FRACTIONS[$FRAC_ID]}
            FRAC_STR=$(printf "%03d" $(echo "${FRAC} * 100" | bc | cut -d. -f1))

            for REP_ID in $(seq 0 $((NREPS - 1))); do
                FNAME="${LABEL}_frac${FRAC_STR}_rep${REP_ID}.csv"

                if [ -f "${OUTDIR}/${FNAME}" ]; then
                    N_EXIST=$((N_EXIST + 1))
                else
                    echo "${PHENO_ID} ${FRAC_ID} ${REP_ID}" >> "${MANIFEST}"
                    N_MISSING=$((N_MISSING + 1))
                fi
            done
        done
    done

    TOTAL=$((NPHENO * NFRAC * NREPS))
    echo "  Output dir:  ${OUTDIR}"
    echo "  Expected:    ${TOTAL}"
    echo "  Existing:    ${N_EXIST}"
    echo "  Missing:     ${N_MISSING}"
    echo "============================================"

    if [ ${N_MISSING} -eq 0 ]; then
        echo "  All runs complete — nothing to submit!"
        rm -f "${MANIFEST}"
        exit 0
    fi

    if [ "${DRY_RUN}" = "1" ]; then
        echo ""
        echo "  Missing runs (first 50):"
        head -50 "${MANIFEST}" | while IFS=' ' read -r PI FI RI; do
            echo "    ${PHENO_LABELS[$PI]} frac=${FRACTIONS[$FI]} rep=${RI}"
        done
        [ ${N_MISSING} -gt 50 ] && echo "    ... and $((N_MISSING - 50)) more"
        echo ""
        echo "  DRY_RUN=1 → not submitting."
        rm -f "${MANIFEST}"
        exit 0
    fi

    # --- Calculate exact array size ---
    N_TASKS=$(( (N_MISSING + SIMS_PER_GPU - 1) / SIMS_PER_GPU ))
    ARRAY_MAX=$((N_TASKS - 1))

    echo ""
    echo "  Submitting ${N_TASKS} array tasks (${SIMS_PER_GPU} sims/GPU)..."

    sbatch \
        --array=0-${ARRAY_MAX} \
        --export=ALL,DAYS=${DAYS} \
        "$0"

    echo "  Manifest written: ${MANIFEST} (${N_MISSING} entries)"
    exit 0
fi

# ============================================================
# PHASE 2: Execute (runs inside SLURM array task)
# ============================================================

sleep 5
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wings-gpu

TASK_ID=${SLURM_ARRAY_TASK_ID}

if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: manifest not found at ${MANIFEST}"
    exit 1
fi

mapfile -t MANIFEST_LINES < "${MANIFEST}"
N_MISSING=${#MANIFEST_LINES[@]}

RUN_START=$((TASK_ID * SIMS_PER_GPU))
RUN_END=$((RUN_START + SIMS_PER_GPU - 1))
if [ ${RUN_END} -ge ${N_MISSING} ]; then
    RUN_END=$((N_MISSING - 1))
fi

echo "============================================"
echo "  W.I.N.G.S. — Δp Sweep (Phase 2: Run)"
echo "============================================"
echo "  Array task:   ${TASK_ID}"
echo "  Node:         $(hostname)"
echo "  GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Manifest:     ${N_MISSING} missing runs total"
echo "  This task:    runs ${RUN_START}–${RUN_END}"
echo "  Days:         ${DAYS}"
echo "============================================"
echo ""

PIDS=()
FNAMES=()
LAUNCHED=0

for RUN_IDX in $(seq ${RUN_START} ${RUN_END}); do
    LINE="${MANIFEST_LINES[$RUN_IDX]}"
    PHENO_ID=$(echo "$LINE" | cut -d' ' -f1)
    FRAC_ID=$(echo "$LINE" | cut -d' ' -f2)
    REP_ID=$(echo "$LINE" | cut -d' ' -f3)

    LABEL=${PHENO_LABELS[$PHENO_ID]}
    FLAGS=${PHENO_FLAGS[$PHENO_ID]}
    FRAC=${FRACTIONS[$FRAC_ID]}
    FRAC_STR=$(printf "%03d" $(echo "${FRAC} * 100" | bc | cut -d. -f1))

    FNAME="${LABEL}_frac${FRAC_STR}_rep${REP_ID}.csv"

    # Race-condition guard
    if [ -f "${OUTDIR}/${FNAME}" ]; then
        echo "  [skip] ${FNAME} (appeared since scan)"
        continue
    fi

    CONDITION_ID=$((PHENO_ID * NFRAC + FRAC_ID))
    SEED=$((42 + CONDITION_ID * 1000 + REP_ID))

    echo "  [launch] ${LABEL} frac=${FRAC} rep=${REP_ID}"

    python ${SCRIPT} \
        --population 50 \
        --max-pop 20000 \
        --grid-size 500 \
        --days ${DAYS} \
        --infected-fraction ${FRAC} \
        --mortality cannibalism \
        --backend brute \
        --device cuda \
        --seed ${SEED} \
        ${FLAGS} \
        --output "${OUTDIR}/${FNAME}" \
        > "${OUTDIR}/${FNAME%.csv}.log" 2>&1 &

    PIDS+=($!)
    FNAMES+=("${FNAME}")
    LAUNCHED=$((LAUNCHED + 1))
done

echo ""
echo "  Launched: ${LAUNCHED}"
echo "  Waiting..."

FAILED=0
T_START=$(date +%s)
for i in "${!PIDS[@]}"; do
    if wait ${PIDS[$i]}; then
        echo "  [done]   ${FNAMES[$i]}"
    else
        echo "  [FAILED] ${FNAMES[$i]} (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

echo ""
echo "============================================"
echo "  Task ${TASK_ID}: ${LAUNCHED} launched, $((LAUNCHED-FAILED)) ok, ${FAILED} failed"
echo "  Wall time: $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "============================================"