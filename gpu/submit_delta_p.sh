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
#SBATCH --array=0-474
#  ↑ Formula: ceil(TOTAL_RUNS / SIMS_PER_GPU) - 1
#    Default: ceil(3 * 19 * 50 / 6) - 1 = ceil(2850/6)-1 = 474
#    Quick test (10 reps):  ceil(3*19*10/6)-1 = 94  → --array=0-94
#    Full run (50 reps):    ceil(3*19*50/6)-1 = 474 → --array=0-474
#SBATCH --output=wings_dp_%A_%a.log
#SBATCH --error=wings_dp_%A_%a.err
# ============================================================
# W.I.N.G.S. — Δp sweep (Array + Packing Hybrid)
# ============================================================
#
# Sweeps initial infection fractions for CI-only, ER-only, CI+ER
# to generate Δp vs p data for the complementary-strategies figure.
#
# Design:
#   3 phenotypes × 19 initial fractions × 50 reps = 2,850 sims
#   Packed 6 per GPU → 475 array tasks
#
# Phenotypes (COMBO_ID):
#   0 = CI only   (--ci)
#   1 = ER only   (--er)
#   2 = CI+ER     (--ci --er)
#
# Fractions: 0.05, 0.10, 0.15, ..., 0.95  (19 levels)
#
# Submit:
#   sbatch submit_delta_p.sh
#
# Quick test (10 reps, 30 days):
#   sbatch --array=0-94 --time=00:30:00 \
#          --export=ALL,NREPS=10,QUICK=1 submit_delta_p.sh
# ============================================================

# --- Paths ---
PROJECT_DIR="/projects/sciences/zoology/geurten_lab/wolbachia_spread_model"
CODE_DIR="/home/geuba03p/PyProjects/WINGS"
SCRIPT="${CODE_DIR}/gpu/gpu_simulation.py"
OUTDIR="${PROJECT_DIR}/abm_delta_p"

# --- Configuration ---
SIMS_PER_GPU=${SIMS_PER_GPU:-6}
NREPS=${NREPS:-50}
DAYS=${DAYS:-365}
if [ "${QUICK}" = "1" ]; then
    DAYS=30
fi

# Phenotype definitions: label, flags
PHENO_LABELS=("CI" "ER" "CI_ER")
PHENO_FLAGS=("--ci" "--er" "--ci --er")
NPHENO=${#PHENO_LABELS[@]}

# Infection fractions: 0.05 to 0.95 in steps of 0.05
FRACTIONS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
NFRAC=${#FRACTIONS[@]}

# Total runs
NCONDITIONS=$((NPHENO * NFRAC))   # 3 × 19 = 57
TOTAL_RUNS=$((NCONDITIONS * NREPS))  # 57 × 50 = 2850

# --- Wait for filesystem mount ---
sleep 5

# --- Activate environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wings-gpu

# --- Which runs does THIS array task own? ---
TASK_ID=${SLURM_ARRAY_TASK_ID}
RUN_START=$((TASK_ID * SIMS_PER_GPU))
RUN_END=$((RUN_START + SIMS_PER_GPU - 1))
if [ ${RUN_END} -ge ${TOTAL_RUNS} ]; then
    RUN_END=$((TOTAL_RUNS - 1))
fi

# --- Diagnostics ---
echo "============================================"
echo "  W.I.N.G.S. — Δp Sweep"
echo "============================================"
echo "  Array task:   ${TASK_ID}"
echo "  Node:         $(hostname)"
echo "  GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Output dir:   ${OUTDIR}"
echo "  Phenotypes:   ${NPHENO}  Fractions: ${NFRAC}  Reps: ${NREPS}"
echo "  Total runs:   ${TOTAL_RUNS}"
echo "  This task:    ${RUN_START}–${RUN_END}"
echo "  Days/run:     ${DAYS}"
echo "============================================"
echo ""

mkdir -p "${OUTDIR}"

# --- Map run index → (phenotype, fraction, replicate) and launch ---
PIDS=()
FNAMES=()
LAUNCHED=0
SKIPPED=0

for RUN_IDX in $(seq ${RUN_START} ${RUN_END}); do
    # Decode: run = condition * NREPS + rep
    CONDITION_ID=$((RUN_IDX / NREPS))
    REP_ID=$((RUN_IDX % NREPS))

    PHENO_ID=$((CONDITION_ID / NFRAC))
    FRAC_ID=$((CONDITION_ID % NFRAC))

    LABEL=${PHENO_LABELS[$PHENO_ID]}
    FLAGS=${PHENO_FLAGS[$PHENO_ID]}
    FRAC=${FRACTIONS[$FRAC_ID]}

    # Format fraction for filename: 0.05 → 005, 0.50 → 050
    FRAC_STR=$(printf "%03d" $(echo "${FRAC} * 100" | bc | cut -d. -f1))

    FNAME="${LABEL}_frac${FRAC_STR}_rep${REP_ID}.csv"

    # Skip if already completed
    if [ -f "${OUTDIR}/${FNAME}" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

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
echo "  Launched: ${LAUNCHED}  |  Skipped: ${SKIPPED}"
echo "  Waiting for ${LAUNCHED} concurrent simulations..."
echo ""

# --- Wait for all and report ---
FAILED=0
T_START=$(date +%s)
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    FNAME=${FNAMES[$i]}
    if wait ${PID}; then
        echo "  [done]   ${FNAME}"
    else
        echo "  [FAILED] ${FNAME} (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
done
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

echo ""
echo "============================================"
echo "  Task ${TASK_ID} complete"
echo "============================================"
echo "  Launched:  ${LAUNCHED}"
echo "  Succeeded: $((LAUNCHED - FAILED))"
echo "  Failed:    ${FAILED}"
echo "  Skipped:   ${SKIPPED}"
echo "  Wall time: $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "============================================"
