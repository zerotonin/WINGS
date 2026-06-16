#!/usr/bin/env python3
# ┌────────────────────────────────────────────────────────────┐
# │ run_dp_batch  « batched low-frequency Δp runner »           │
# └────────────────────────────────────────────────────────────┘
"""Run a contiguous block of low-frequency ABM replicates in one process.

The low-frequency Turelli-threshold sweep seeds infections at p = 0.5–6 %
and only needs the *initial* Δp, so each simulation is short (~72 days).
At that length the per-process torch/CUDA startup (~8 s) dwarfs the
compute (~3.5 s), so launching one process per replicate wastes ~70 % of
the wall-clock.  This wrapper imports the heavy stack **once** and loops
the replicate range, amortising the startup across the whole block.

A ``--condition`` selects the phenotype set under test — CI alone (the
threshold control), ER alone, or CI+ER (the relay) — so the same wrapper
serves the whole low-frequency panel.  Output filenames follow
``{COND}_frac{NNNN}_rep{R}.csv`` where ``NNNN`` is the seed fraction in
per-mille (4 digits), so half-percent steps (0.005 → ``0005``) are
encodable — unlike the 0.01-resolution ``frac{NNN}`` scheme in
:mod:`wings.analysis.ingest_delta_p`.  The CI seed stream and filenames are
unchanged from the original CI-only sweep, so completed CI cells resume
byte-for-byte.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# ┌────────────────────────────────────────────────────────────┐
# │ Phenotype conditions  « effect toggles + seed-stream offset »│
# └────────────────────────────────────────────────────────────┘
#  CI keeps offset 0 so its seed stream and filenames match the already-
#  completed CI-only sweep exactly; ER/CI_ER get disjoint streams.
CONDITION_EFFECTS: dict[str, dict[str, bool]] = {
    "CI":    {"cytoplasmic_incompatibility": True,  "increased_exploration_rate": False},
    "ER":    {"cytoplasmic_incompatibility": False, "increased_exploration_rate": True},
    "CI_ER": {"cytoplasmic_incompatibility": True,  "increased_exploration_rate": True},
}
CONDITION_SEED_OFFSET: dict[str, int] = {"CI": 0, "ER": 1, "CI_ER": 2}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batched CI-only low-frequency Δp replicate runner")
    parser.add_argument("--mu", type=float, required=True,
                        help="Maternal-transmission leakage")
    parser.add_argument("--condition", choices=sorted(CONDITION_EFFECTS),
                        default="CI",
                        help="Phenotype set: CI (control), ER, or CI_ER (relay)")
    parser.add_argument("--frac", type=float, required=True,
                        help="Initial infected fraction (0.005–0.06)")
    parser.add_argument("--rep-start", type=int, required=True)
    parser.add_argument("--rep-end", type=int, required=True,
                        help="Inclusive last replicate index")
    parser.add_argument("--population", type=int, default=2000,
                        help="Initial population (≥2000 to seed sub-1%% cleanly)")
    parser.add_argument("--max-pop", type=int, default=20_000)
    parser.add_argument("--grid-size", type=int, default=500)
    parser.add_argument("--days", type=int, default=72)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Heavy imports happen once per process — the whole point of batching.
    from wings.models.gpu_abm import SimConfig, run_experiment

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    permille = round(args.frac * 1000)          # 0.005 → 5 → "0005"
    mu_id = round(args.mu * 1000)
    cond_effects = CONDITION_EFFECTS[args.condition]
    cond_offset = CONDITION_SEED_OFFSET[args.condition] * 100_000_000
    launched = skipped = 0

    for rep in range(args.rep_start, args.rep_end + 1):
        out_csv = outdir / f"{args.condition}_frac{permille:04d}_rep{rep}.csv"
        if out_csv.exists() and out_csv.stat().st_size > 0:
            skipped += 1
            continue
        seed = cond_offset + args.seed_base + mu_id * 1_000_000 + permille * 1_000 + rep
        cfg = SimConfig(
            initial_population=args.population,
            max_population=args.max_pop,
            grid_size=args.grid_size,
            infected_fraction=args.frac,
            maternal_transmission_leakage=args.mu,
            mortality_mode="cannibalism",
            mating_backend="brute",
            device=args.device,
            seed=seed,
            wolbachia_effects={
                "cytoplasmic_incompatibility": cond_effects["cytoplasmic_incompatibility"],
                "male_killing": False,
                "increased_exploration_rate": cond_effects["increased_exploration_rate"],
                "increased_eggs": False,
                "reduced_eggs": False,
            },
        )
        sim = run_experiment(cfg, n_days=args.days, verbose=False)
        sim.export_history_csv(str(out_csv))
        launched += 1

    print(f"cond={args.condition} mu={args.mu} frac={args.frac} "
          f"reps[{args.rep_start}-{args.rep_end}]: "
          f"launched={launched} skipped={skipped}")


if __name__ == "__main__":
    main()
