import itertools
import os
import multiprocessing
import pandas as pd
from Environment import Environment
from tqdm import tqdm
import numpy as np

# Constants for batch simulations
GRID_SIZE = 500
INITIAL_POPULATION = 10
INFECTED_FRACTION = 0.2
MAX_TIME = 8760  # simulate one year per trial
SAVE_PATH = "./data/compare_spread_features/"
PARALLEL_PROCESSES = 18  # adjust based on available CPU cores

def run_simulation(args):
    """
    Runs a single simulation with given effects and saves daily median results to a CSV.
    args: tuple containing (wolbachia_effects_dict, trial_number).
    """
    effects, trial = args
    # Extract CI strength if provided in effects, otherwise default to 1.0
    ci_strength = effects.pop('ci_strength', 1.0)
    env = Environment(GRID_SIZE, INITIAL_POPULATION, effects, 
                      infected_fraction=INFECTED_FRACTION, 
                      ci_strength=ci_strength, multiple_mating=True, 
                      use_gpu=True)
    # Run the simulation for MAX_TIME hours
    for hour in range(MAX_TIME):
        env.run_simulation_step()
    # Compute daily medians for population size and infection rate
    daily_pop = [np.median(env.population_size[i:i+24]) for i in range(0, len(env.population_size), 24)]
    daily_inf = [np.median(env.infection_history[i:i+24]) for i in range(0, len(env.infection_history), 24)]
    # Construct filename based on effects and CI strength
    effect_tokens = [f"{k}_{v}" for k, v in effects.items()]
    if effects.get('cytoplasmic_incompatibility', False):
        effect_tokens.insert(1, f"ci_strength_{ci_strength}")
    filename = os.path.join(SAVE_PATH, f"{'_'.join(effect_tokens)}_{trial:03d}.csv")
    # Ensure output directory exists and save results to CSV
    os.makedirs(SAVE_PATH, exist_ok=True)
    pd.DataFrame({'Population Size': daily_pop, 'Infection Rate': daily_inf}).to_csv(filename, index=False)

def main():
    # Define all combinations of effect toggles (excluding CI strength which is handled separately)
    keys = ['cytoplasmic_incompatibility', 'male_killing', 'increased_exploration_rate', 'increased_eggs', 'reduced_eggs']
    all_combos = list(itertools.product([True, False], repeat=len(keys)))
    # Filter out invalid combination where both increased_eggs and reduced_eggs are True
    valid_combos = [
        combo for combo in all_combos 
        if not (combo[keys.index('increased_eggs')] and combo[keys.index('reduced_eggs')])
    ]
    # Prepare job list: each job is (effects_dict, trial_number)
    trials = range(50)  # number of trials per scenario
    jobs = []
    for combo in valid_combos:
        effects = dict(zip(keys, combo))
        # If CI is enabled, run separate sets of trials for each CI strength value
        if effects.get('cytoplasmic_incompatibility', False):
            for ci_val in [0.5, 0.75, 1.0]:
                eff_copy = effects.copy()
                eff_copy['ci_strength'] = ci_val
                for t in trials:
                    jobs.append((eff_copy, t))
        else:
            for t in trials:
                jobs.append((effects, t))
    # Run simulations in parallel
    os.makedirs(SAVE_PATH, exist_ok=True)
    with multiprocessing.Pool(PARALLEL_PROCESSES) as pool:
        list(tqdm(pool.imap(run_simulation, jobs), total=len(jobs), desc="Batch Simulations"))

if __name__ == "__main__":
    main()
