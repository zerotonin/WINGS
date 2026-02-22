#!/bin/bash
#SBATCH --job-name=wings_rerun
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_L40,aoraki_gpu,aoraki_gpu_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --output=wings_rerun_%j.log
#SBATCH --error=wings_rerun_%j.err
# ============================================================
# W.I.N.G.S. — Rerun missing simulations
# ============================================================
# Scans OUTDIR for missing CSVs across all 16 combos × 200 reps,
# then launches them 6-at-a-time on a single GPU.
#
# Usage:
#   sbatch rerun_missing.sh            # auto-detect & run missing
#   DRY_RUN=1 bash rerun_missing.sh    # just list what's missing
# ============================================================

PROJECT_DIR="/projects/sciences/zoology/geurten_lab/wolbachia_spread_model"
CODE_DIR="/home/geuba03p/PyProjects/WINGS"
SCRIPT="${CODE_DIR}/gpu/gpu_simulation.py"
OUTDIR="${PROJECT_DIR}/gpu_results"

NREPS=${NREPS:-200}
NCOMBOS=16
SIMS_PER_GPU=${SIMS_PER_GPU:-6}
DAYS=${DAYS:-365}
DRY_RUN=${DRY_RUN:-0}

# --- Activate environment ---
if [ "${DRY_RUN}" != "1" ]; then
    sleep 5
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate wings-gpu
fi

# --- Scan for missing files ---
MISSING_RUNS=()
for COMBO_ID in $(seq 0 $((NCOMBOS - 1))); do
    CI=$(( (COMBO_ID >> 3) & 1 ))
    MK=$(( (COMBO_ID >> 2) & 1 ))
    ER=$(( (COMBO_ID >> 1) & 1 ))
    IE=$(( (COMBO_ID >> 0) & 1 ))

    PREFIX="cytoplasmic_incompatibility_$([ $CI -eq 1 ] && echo True || echo False)"
    PREFIX="${PREFIX}_male_killing_$([ $MK -eq 1 ] && echo True || echo False)"
    PREFIX="${PREFIX}_increased_exploration_rate_$([ $ER -eq 1 ] && echo True || echo False)"
    PREFIX="${PREFIX}_increased_eggs_$([ $IE -eq 1 ] && echo True || echo False)"

    for REP_ID in $(seq 0 $((NREPS - 1))); do
        FNAME="${PREFIX}_${REP_ID}.csv"
        if [ ! -f "${OUTDIR}/${FNAME}" ]; then
            MISSING_RUNS+=("${COMBO_ID}:${REP_ID}")
        fi
    done
done

N_MISSING=${#MISSING_RUNS[@]}
echo "============================================"
echo "  W.I.N.G.S. — Missing Run Scanner"
echo "============================================"
echo "  Output dir:  ${OUTDIR}"
echo "  Expected:    $((NCOMBOS * NREPS)) files"
echo "  Missing:     ${N_MISSING}"
echo "============================================"

if [ ${N_MISSING} -eq 0 ]; then
    echo "  All runs complete — nothing to do!"
    exit 0
fi

# List missing runs
for entry in "${MISSING_RUNS[@]}"; do
    COMBO_ID=${entry%%:*}
    REP_ID=${entry##*:}
    CI=$(( (COMBO_ID >> 3) & 1 ))
    MK=$(( (COMBO_ID >> 2) & 1 ))
    ER=$(( (COMBO_ID >> 1) & 1 ))
    IE=$(( (COMBO_ID >> 0) & 1 ))
    echo "  missing: combo=${COMBO_ID} (CI=$CI MK=$MK ER=$ER IE=$IE) rep=${REP_ID}"
done
echo ""

if [ "${DRY_RUN}" = "1" ]; then
    echo "  DRY_RUN=1 → exiting without launching."
    exit 0
fi

# --- Launch missing runs in batches of SIMS_PER_GPU ---
TOTAL_LAUNCHED=0
TOTAL_FAILED=0
T_GLOBAL_START=$(date +%s)

for (( BATCH_START=0; BATCH_START < N_MISSING; BATCH_START += SIMS_PER_GPU )); do
    BATCH_END=$((BATCH_START + SIMS_PER_GPU - 1))
    if [ ${BATCH_END} -ge ${N_MISSING} ]; then
        BATCH_END=$((N_MISSING - 1))
    fi
    BATCH_SIZE=$((BATCH_END - BATCH_START + 1))

    echo "--- Batch: runs $((BATCH_START+1))–$((BATCH_END+1)) of ${N_MISSING} (${BATCH_SIZE} sims) ---"

    PIDS=()
    FNAMES=()

    for (( i=BATCH_START; i<=BATCH_END; i++ )); do
        entry="${MISSING_RUNS[$i]}"
        COMBO_ID=${entry%%:*}
        REP_ID=${entry##*:}

        CI=$(( (COMBO_ID >> 3) & 1 ))
        MK=$(( (COMBO_ID >> 2) & 1 ))
        ER=$(( (COMBO_ID >> 1) & 1 ))
        IE=$(( (COMBO_ID >> 0) & 1 ))

        FNAME="cytoplasmic_incompatibility_$([ $CI -eq 1 ] && echo True || echo False)"
        FNAME="${FNAME}_male_killing_$([ $MK -eq 1 ] && echo True || echo False)"
        FNAME="${FNAME}_increased_exploration_rate_$([ $ER -eq 1 ] && echo True || echo False)"
        FNAME="${FNAME}_increased_eggs_$([ $IE -eq 1 ] && echo True || echo False)"
        FNAME="${FNAME}_${REP_ID}.csv"

        FLAGS=""
        [ $CI -eq 1 ] && FLAGS="$FLAGS --ci"
        [ $MK -eq 1 ] && FLAGS="$FLAGS --mk"
        [ $ER -eq 1 ] && FLAGS="$FLAGS --er"
        [ $IE -eq 1 ] && FLAGS="$FLAGS --ie"

        SEED=$((42 + COMBO_ID * 100 + REP_ID))

        echo "  [launch] combo=${COMBO_ID} rep=${REP_ID}  CI=$CI MK=$MK ER=$ER IE=$IE"

        python ${SCRIPT} \
            --population 50 \
            --max-pop 20000 \
            --grid-size 500 \
            --days ${DAYS} \
            --mortality cannibalism \
            --backend brute \
            --device cuda \
            --seed ${SEED} \
            ${FLAGS} \
            --output "${OUTDIR}/${FNAME}" \
            > "${OUTDIR}/${FNAME%.csv}.log" 2>&1 &

        PIDS+=($!)
        FNAMES+=("${FNAME}")
        TOTAL_LAUNCHED=$((TOTAL_LAUNCHED + 1))
    done

    # Wait for this batch
    for j in "${!PIDS[@]}"; do
        if wait ${PIDS[$j]}; then
            echo "  [done]   ${FNAMES[$j]}"
        else
            echo "  [FAILED] ${FNAMES[$j]} (exit $?)"
            TOTAL_FAILED=$((TOTAL_FAILED + 1))
        fi
    done
    echo ""
done

T_GLOBAL_END=$(date +%s)
ELAPSED=$((T_GLOBAL_END - T_GLOBAL_START))

echo "============================================"
echo "  Rerun complete"
echo "============================================"
echo "  Launched:  ${TOTAL_LAUNCHED}"
echo "  Succeeded: $((TOTAL_LAUNCHED - TOTAL_FAILED))"
echo "  Failed:    ${TOTAL_FAILED}"
echo "  Wall time: $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "============================================"
