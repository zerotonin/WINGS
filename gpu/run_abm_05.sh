#!/usr/bin/env bash
# ============================================================================
#  WINGS — ABM batch runner at 0.5 initial infection fraction
#
#  Runs all 16 Wolbachia phenotype combinations × 200 replicates
#  Output: abm_05_results/<combo_label>/rep_<NNN>.csv
#
#  Usage (sequential, single GPU):
#    bash run_abm_05.sh
#
#  Usage (SLURM array, one job per combo):
#    sbatch --array=0-15 run_abm_05.sh
# ============================================================================

set -euo pipefail

# ── Configuration ──
SCRIPT="gpu_simulation.py"
OUTDIR="abm_05_results"
N_REPS=200
DAYS=365
POP=50
MAX_POP=20000
INFECTED_FRACTION=0.50
DEVICE="cuda"

# ── All 16 flag combinations (CI MK ER IE) ──
COMBOS=(
  "    "       # 0:  None
  "   --ie"    # 1:  IE
  "  --er  "   # 2:  ER
  "  --er --ie" # 3:  ER+IE
  " --mk   "   # 4:  MK
  " --mk  --ie" # 5:  MK+IE
  " --mk --er  " # 6:  MK+ER
  " --mk --er --ie" # 7: MK+ER+IE
  "--ci    "   # 8:  CI
  "--ci   --ie" # 9:  CI+IE
  "--ci  --er  " # 10: CI+ER
  "--ci  --er --ie" # 11: CI+ER+IE
  "--ci --mk   " # 12: CI+MK
  "--ci --mk  --ie" # 13: CI+MK+IE
  "--ci --mk --er  " # 14: CI+MK+ER
  "--ci --mk --er --ie" # 15: CI+MK+ER+IE
)

LABELS=(
  "None" "IE" "ER" "ER+IE"
  "MK" "MK+IE" "MK+ER" "MK+ER+IE"
  "CI" "CI+IE" "CI+ER" "CI+ER+IE"
  "CI+MK" "CI+MK+IE" "CI+MK+ER" "CI+MK+ER+IE"
)

# ── Determine which combos to run ──
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  # SLURM array mode: one combo per job
  START=${SLURM_ARRAY_TASK_ID}
  END=${SLURM_ARRAY_TASK_ID}
  echo "SLURM array task ${SLURM_ARRAY_TASK_ID}: combo=${LABELS[$START]}"
else
  # Sequential mode: all combos
  START=0
  END=15
  echo "Sequential mode: running all 16 combos × ${N_REPS} reps"
fi

mkdir -p "${OUTDIR}"

# ── Main loop ──
for combo_idx in $(seq ${START} ${END}); do
  label="${LABELS[$combo_idx]}"
  flags="${COMBOS[$combo_idx]}"
  combo_dir="${OUTDIR}/${label// /}"  # strip spaces from label

  mkdir -p "${combo_dir}"
  echo ""
  echo "━━━ Combo ${combo_idx}/15: ${label} ━━━"
  echo "    Flags: ${flags}"
  echo "    Output: ${combo_dir}/"

  for rep in $(seq 1 ${N_REPS}); do
    outfile="${combo_dir}/rep_$(printf '%03d' ${rep}).csv"

    # Skip if already completed
    if [[ -f "${outfile}" ]]; then
      continue
    fi

    seed=$((combo_idx * 10000 + rep))

    python "${SCRIPT}" \
      --population ${POP} \
      --max-pop ${MAX_POP} \
      --days ${DAYS} \
      --infected-fraction ${INFECTED_FRACTION} \
      --device ${DEVICE} \
      --seed ${seed} \
      --output "${outfile}" \
      ${flags}

    # Progress indicator (every 10 reps)
    if (( rep % 10 == 0 )); then
      echo "    ✓ ${label}: ${rep}/${N_REPS}"
    fi
  done

  echo "    ✓ ${label}: done (${N_REPS} reps)"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  All ABM 0.5 simulations complete."
echo "  Results in: ${OUTDIR}/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
