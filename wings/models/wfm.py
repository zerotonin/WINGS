#!/usr/bin/env python3
"""
W.I.N.G.S. — Fixed-size discrete-generation Wolbachia spread model.

Wright-Fisher style:  N adults are sampled each generation from the
offspring pool.  This decouples infection-frequency dynamics from
population-size dynamics, providing a clean baseline for measuring
the effect of CI, male-killing, increased exploration, and increased
eggs on Wolbachia invasion.

Biology
-------
Tribolium castaneum generation time ≈ 30 days at 30 °C (Pointer et al.
2021, Heredity).  One year ≈ 12 generations.

Mapping spatial effects to well-mixed
--------------------------------------
The ABM's "increased exploration rate" gives infected ♀ a 1.4× mating
distance.  In a well-mixed cage of 50 beetles, the spatial advantage
is translated as a *mating-probability advantage*: infected ♀ mate
with probability 1.0 while uninfected ♀ mate with probability
1 / exploration_rate_boost ≈ 0.714.  This represents the empirical
observation that more-active females encounter males more frequently
even in small arenas (Pai & Bhatt 1995, J. Stored Prod. Res.).

Expected CI dynamics (sanity check)
------------------------------------
With full CI (strength 1.0) and initial freq p = 0.5:
  Infected ♀ (freq p) mate with any ♂  → offspring all infected.
  Uninfected ♀ (freq 1-p) × uninfected ♂ (freq 1-p) → offspring uninfected.
  Uninfected ♀ × infected ♂ → NO offspring (CI).

  New p ≈ p / (p + (1-p)²)

  Gen 0: p=0.50 → Gen 1: p≈0.67 → Gen 2: p≈0.86 → Gen 3: p≈0.97 → fixation.
  Expect fixation within 4–6 generations with stochastic variation.

Usage
-----
  # Single run
  python fixed_generation_sim.py --ci --seed 42 --output result.csv

  # All 16 combos × 200 reps (takes ~30 seconds on one CPU core)
  python fixed_generation_sim.py --run-all --nreps 200

  # Quick test
  python fixed_generation_sim.py --run-all --nreps 2 --outdir ./test_fixed
"""

import argparse
import csv
import os
import sys
import time
from itertools import product as iterproduct

import numpy as np


# ======================================================================
#  Core simulation
# ======================================================================

def simulate(
    N: int = 50,
    max_generations: int = 12,
    seed: int = 42,
    ci: bool = False,
    mk: bool = False,
    er: bool = False,
    ie: bool = False,
    ci_strength: float = 1.0,
    egg_laying_max: int = 15,
    male_offspring_rate: float = 0.1,
    fecundity_increase_factor: float = 1.35,
    er_mating_advantage: float = 1.4,
    initial_infection_freq: float = 0.5,
):
    """
    Run one fixed-size discrete-generation simulation.

    Parameters
    ----------
    N : int
        Fixed adult population size per generation.
    max_generations : int
        Maximum generations to simulate (≈ 1 year at 12).
    seed : int
        RNG seed for reproducibility.
    ci, mk, er, ie : bool
        Wolbachia effect toggles.
    ci_strength : float
        Probability that each egg from ♂I × ♀U cross is killed (0–1).
    egg_laying_max : int
        Maximum clutch size per mating (uniform draw from 1..max).
    male_offspring_rate : float
        Probability offspring is male when MK active and mother infected.
    fecundity_increase_factor : float
        Clutch multiplier for infected mothers when IE active.
    er_mating_advantage : float
        Mating probability ratio (infected / uninfected) when ER active.
    initial_infection_freq : float
        Starting Wolbachia frequency (0–1).

    Returns
    -------
    list of (population_size: int, infection_rate: float)
        One entry per generation, padded to max_generations + 1.
    """
    rng = np.random.default_rng(seed)

    # --- Initialise population ---
    n_infected_init = int(round(N * initial_infection_freq))
    infected = np.zeros(N, dtype=bool)
    infected[:n_infected_init] = True
    rng.shuffle(infected)

    is_male = np.zeros(N, dtype=bool)
    is_male[: N // 2] = True
    rng.shuffle(is_male)

    history = []  # (pop_size, infection_rate)

    for gen in range(max_generations + 1):
        # --- Record state ---
        n_alive = len(infected)
        n_inf = int(infected.sum())
        inf_rate = n_inf / n_alive if n_alive > 0 else 0.0
        history.append((n_alive, inf_rate))

        # --- Stopping conditions ---
        if inf_rate >= 1.0:
            # Fixation — pad remaining generations
            for _ in range(max_generations - gen):
                history.append((N, 1.0))
            break

        if inf_rate <= 0.0 and gen > 0:
            # Wolbachia lost — pad remaining generations
            for _ in range(max_generations - gen):
                history.append((N, 0.0))
            break

        if gen == max_generations:
            break

        # --- Identify sexes ---
        females = np.where(~is_male)[0]
        males = np.where(is_male)[0]

        if len(females) == 0 or len(males) == 0:
            # All one sex — population cannot reproduce; pad with current state
            for _ in range(max_generations - gen):
                history.append((N, inf_rate))
            break

        # --- Mating probability (ER effect) ---
        if er:
            # Infected ♀ mate with prob 1.0;  uninfected ♀ at reduced rate
            mate_prob = np.where(
                infected[females], 1.0, 1.0 / er_mating_advantage
            )
        else:
            mate_prob = np.ones(len(females))

        do_mate = rng.random(len(females)) < mate_prob
        mating_females = females[do_mate]

        if len(mating_females) == 0:
            for _ in range(max_generations - gen):
                history.append((N, inf_rate))
            break

        # --- Pair each mating ♀ with a random ♂ ---
        chosen_males = rng.choice(males, size=len(mating_females), replace=True)

        # --- Offspring production (vectorised) ---
        f_inf = infected[mating_females]
        m_inf = infected[chosen_males]

        # Base clutch sizes  (Uniform 1..egg_laying_max)
        eggs = rng.integers(1, egg_laying_max + 1, size=len(mating_females))

        # IE: infected mothers produce more eggs
        if ie:
            boost_mask = f_inf
            eggs = np.where(
                boost_mask,
                np.round(eggs * fecundity_increase_factor).astype(int),
                eggs,
            )

        # CI: infected ♂ × uninfected ♀  →  eggs killed
        if ci:
            ci_mask = m_inf & ~f_inf
            if ci_strength >= 1.0:
                eggs[ci_mask] = 0
            else:
                # Partial CI: each egg survives independently
                ci_idx = np.where(ci_mask)[0]
                for idx in ci_idx:
                    eggs[idx] = rng.binomial(eggs[idx], 1.0 - ci_strength)

        # --- Expand eggs into individual offspring ---
        # Mothers with 0 eggs contribute nothing
        valid = eggs > 0
        if not valid.any():
            for _ in range(max_generations - gen):
                history.append((N, inf_rate))
            break

        eggs_v = eggs[valid]
        f_inf_v = f_inf[valid]

        total_offspring = int(eggs_v.sum())

        # Infection: inherited from mother
        offspring_infected = np.repeat(f_inf_v, eggs_v)

        # Sex determination
        if mk:
            # Infected mothers → mostly female offspring
            prob_male = np.where(
                offspring_infected,
                male_offspring_rate,
                0.5,
            )
            offspring_male = rng.random(total_offspring) < prob_male
        else:
            offspring_male = rng.random(total_offspring) < 0.5

        # --- Wright-Fisher sampling: draw N adults for next generation ---
        if total_offspring == 0:
            for _ in range(max_generations - gen):
                history.append((N, inf_rate))
            break

        if total_offspring >= N:
            idx = rng.choice(total_offspring, size=N, replace=False)
        else:
            # Fewer offspring than N — sample with replacement (bottleneck)
            idx = rng.choice(total_offspring, size=N, replace=True)

        infected = offspring_infected[idx]
        is_male = offspring_male[idx]

    return history


# ======================================================================
#  I/O helpers
# ======================================================================

def save_csv(history, path):
    """Save simulation history as CSV matching gpu_simulation.py format."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Population Size", "Infection Rate"])
        for pop, inf in history:
            writer.writerow([pop, f"{inf:.6f}"])


def combo_id(ci, mk, er, ie):
    """Compute the 4-bit combo index consistent with the GPU submit script."""
    return (int(ci) << 3) | (int(mk) << 2) | (int(er) << 1) | int(ie)


def combo_label(ci, mk, er, ie):
    parts = []
    if ci: parts.append("CI")
    if mk: parts.append("MK")
    if er: parts.append("ER")
    if ie: parts.append("IE")
    return "+".join(parts) if parts else "None"


def make_filename(ci, mk, er, ie, rep):
    """Build filename matching the original ABM format for ingest compatibility."""
    return (
        f"cytoplasmic_incompatibility_{ci}"
        f"_male_killing_{mk}"
        f"_increased_exploration_rate_{er}"
        f"_increased_eggs_{ie}"
        f"_{rep}.csv"
    )


# ======================================================================
#  Batch mode
# ======================================================================

def run_all(args):
    """Run all 16 combos × nreps replicates."""
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    combos = list(iterproduct([False, True], repeat=4))  # CI, MK, ER, IE
    total = len(combos) * args.nreps
    done = 0
    skipped = 0
    t0 = time.time()

    print("=" * 56)
    print("  W.I.N.G.S. — Fixed-size discrete-generation model")
    print("=" * 56)
    print(f"  Population:   {args.population} adults / generation")
    print(f"  Generations:  {args.max_generations} (≈ {args.max_generations * 30} days)")
    print(f"  Init. freq:   {args.initial_infection_freq:.0%}")
    print(f"  Combinations: {len(combos)}")
    print(f"  Replicates:   {args.nreps}")
    print(f"  Total runs:   {total}")
    print(f"  Output:       {outdir}")
    print("=" * 56)
    print()

    # --- Track per-combo fixation statistics ---
    fixation_stats = {}

    for ci, mk, er, ie in combos:
        label = combo_label(ci, mk, er, ie)
        fixation_gens = []
        combo_skipped = 0

        for rep in range(args.nreps):
            fname = make_filename(ci, mk, er, ie, rep)
            fpath = os.path.join(outdir, fname)

            if os.path.exists(fpath):
                skipped += 1
                combo_skipped += 1
                done += 1
                continue

            cid = combo_id(ci, mk, er, ie)
            seed = 42 + cid * 1000 + rep

            history = simulate(
                N=args.population,
                max_generations=args.max_generations,
                seed=seed,
                ci=ci, mk=mk, er=er, ie=ie,
                ci_strength=args.ci_strength,
                initial_infection_freq=args.initial_infection_freq,
            )
            save_csv(history, fpath)

            # Track fixation
            for gen_i, (_, inf) in enumerate(history):
                if inf >= 1.0:
                    fixation_gens.append(gen_i)
                    break

            done += 1

        # Report per-combo progress
        n_ran = args.nreps - combo_skipped
        n_fixed = len(fixation_gens)
        if n_ran > 0:
            pct = 100 * n_fixed / n_ran
            avg_gen = f"{np.mean(fixation_gens):.1f}" if n_fixed > 0 else "—"
            print(
                f"  {label:20s}  "
                f"ran={n_ran:3d}  "
                f"fixation={n_fixed:3d}/{n_ran:3d} ({pct:5.1f}%)  "
                f"avg_gen={avg_gen}"
            )
        else:
            print(f"  {label:20s}  (all skipped)")

        fixation_stats[label] = fixation_gens

    elapsed = time.time() - t0
    print()
    print("=" * 56)
    print(f"  Complete:  {done} runs  ({skipped} skipped)")
    print(f"  Wall time: {elapsed:.1f}s")
    print(f"  Output:    {outdir}")
    print("=" * 56)

    return fixation_stats


# ======================================================================
#  CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="W.I.N.G.S. — Fixed-size discrete-generation Wolbachia model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run with CI enabled
  python fixed_generation_sim.py --ci --seed 42 --output result.csv

  # All 3200 runs (16 combos × 200 reps), takes ~30 s on one CPU core
  python fixed_generation_sim.py --run-all --nreps 200

  # Quick sanity check (2 reps per combo, local output)
  python fixed_generation_sim.py --run-all --nreps 2 --outdir ./test_fixed
        """,
    )

    # --- Population parameters ---
    parser.add_argument(
        "--population", type=int, default=50,
        help="Fixed adult population size per generation (default: 50)",
    )
    parser.add_argument(
        "--max-generations", type=int, default=12,
        help="Max generations to simulate; 12 ≈ 1 year (default: 12)",
    )
    parser.add_argument(
        "--initial-infection-freq", type=float, default=0.5,
        help="Starting Wolbachia frequency (default: 0.5)",
    )

    # --- Wolbachia effect toggles ---
    parser.add_argument("--ci", action="store_true",
                        help="Enable cytoplasmic incompatibility")
    parser.add_argument("--mk", action="store_true",
                        help="Enable male-killing")
    parser.add_argument("--er", action="store_true",
                        help="Enable increased exploration rate")
    parser.add_argument("--ie", action="store_true",
                        help="Enable increased eggs (fecundity boost)")

    # --- Effect strengths ---
    parser.add_argument("--ci-strength", type=float, default=1.0,
                        help="CI strength: P(egg killed) for ♂I×♀U (default: 1.0)")
    parser.add_argument("--egg-laying-max", type=int, default=15,
                        help="Max eggs per mating (uniform 1..max, default: 15)")
    parser.add_argument("--male-offspring-rate", type=float, default=0.1,
                        help="P(male) for infected mothers when MK active (default: 0.1)")
    parser.add_argument("--fecundity-increase-factor", type=float, default=1.35,
                        help="Clutch multiplier for IE (default: 1.35)")
    parser.add_argument("--er-mating-advantage", type=float, default=1.4,
                        help="Mating probability ratio infected/uninfected for ER (default: 1.4)")

    # --- Single-run output ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (single-run mode)")

    # --- Batch mode ---
    parser.add_argument("--run-all", action="store_true",
                        help="Run all 16 combos × nreps replicates")
    parser.add_argument("--nreps", type=int, default=200,
                        help="Replicates per combo in batch mode (default: 200)")
    parser.add_argument(
        "--outdir", type=str,
        default="/projects/sciences/zoology/geurten_lab/"
                "wolbachia_spread_model/gpu_results_50beetles",
        help="Output directory for batch mode",
    )

    args = parser.parse_args()

    if args.run_all:
        run_all(args)
    else:
        history = simulate(
            N=args.population,
            max_generations=args.max_generations,
            seed=args.seed,
            ci=args.ci,
            mk=args.mk,
            er=args.er,
            ie=args.ie,
            ci_strength=args.ci_strength,
            egg_laying_max=args.egg_laying_max,
            male_offspring_rate=args.male_offspring_rate,
            fecundity_increase_factor=args.fecundity_increase_factor,
            er_mating_advantage=args.er_mating_advantage,
            initial_infection_freq=args.initial_infection_freq,
        )
        outpath = args.output or "result.csv"
        save_csv(history, outpath)

        # Print summary
        final_pop, final_inf = history[-1]
        n_gens = len(history) - 1
        fixation_gen = None
        for i, (_, inf) in enumerate(history):
            if inf >= 1.0:
                fixation_gen = i
                break

        flags = []
        if args.ci: flags.append("CI")
        if args.mk: flags.append("MK")
        if args.er: flags.append("ER")
        if args.ie: flags.append("IE")
        label = "+".join(flags) if flags else "None"

        print(f"  Effects:     {label}")
        print(f"  Generations: {n_gens}")
        print(f"  Final freq:  {final_inf:.3f}")
        if fixation_gen is not None:
            print(f"  Fixation:    generation {fixation_gen}")
        else:
            print(f"  Fixation:    not reached")
        print(f"  Saved:       {outpath}")


if __name__ == "__main__":
    main()
