import itertools
import os
import multiprocessing
import pandas as pd
from Environment import Environment
from tqdm import tqdm
import numpy as np

# Constants for the simulation
GRID_SIZE = 500
INITIAL_POPULATION = 10
INFECTED_FRACTION = 0.2
MAX_TIME = 8760 # hours to simulate, 8760 is a year
SAVE_PATH = "/home/bgeurten/wolbachia_spread_model/raw_data/compare_spread_features/"  # Update with your desired path
PARALLEL_THREADS = 20 #Number of parallel simulations


def run_simulation(args):
    """
    Runs a single Wolbachia simulation and saves the results to a CSV file.

    Parameters:
        args (tuple): A tuple containing wolbachia_effects and trial_number.
    """
    wolbachia_effects, trial_number = args  
    
    env = Environment(GRID_SIZE, INITIAL_POPULATION, wolbachia_effects, INFECTED_FRACTION)

    for hour in range(MAX_TIME):  # Simulation runs for one year or until all beetles are infected
        env.run_simulation_step()
        if env.infected_fraction >= 1.0:
            break

    # Process and save data to daily median values
    save_simulation_results(env, wolbachia_effects, trial_number)

def save_simulation_results(env, wolbachia_effects, trial_number):
    """
    Processes and saves the simulation results to a CSV file.

    Parameters:
        env (Environment): The simulation environment with data to be saved.
        wolbachia_effects (dict): Dictionary of the Wolbachia effects used in the simulation.
        trial_number (int): The trial number of the simulation.
    """
    daily_population_size = [np.median(env.population_size[i:i+24]) for i in range(0, len(env.population_size), 24)]
    daily_infection_rate = [np.median(env.infection_history[i:i+24]) for i in range(0, len(env.infection_history), 24)]

    filename = os.path.join(SAVE_PATH, f"{'_'.join([k+'_'+str(v) for k,v in wolbachia_effects.items()])}_{trial_number:02d}.csv")
    pd.DataFrame({'Population Size': daily_population_size, 'Infection Rate': daily_infection_rate}).to_csv(filename, index=False)

def main():
    """
    Main function to run multiple simulations across various Wolbachia effect combinations.
    """
    all_combinations = list(itertools.product([True, False], repeat=4))
    trials = list(range(100))
    jobs = [(dict(zip(['cytoplasmic_incompatibility', 'male_killing', 'increased_exploration_rate', 'increased_eggs'], combo)), trial) for combo in all_combinations for trial in trials]

    # Run simulations in parallel using multiprocessing
    with multiprocessing.Pool(PARALLEL_THREADS) as pool:  # Adjust the number of processes as needed
        list(tqdm(pool.imap(run_simulation, jobs), total=len(jobs)))

if __name__ == "__main__":
    main()
