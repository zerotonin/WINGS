from Environment import Environment
from ParameterSet import ParameterSet
import matplotlib.pyplot as plt
from tqdm import tqdm

# Simulation parameters
GRID_SIZE = 500
INITIAL_POPULATION = 10
wolbachia_effects = {
    'cytoplasmic_incompatibility': True,
    'male_killing': False,
    'increased_exploration_rate': True,
    'increased_eggs': True,
    'reduced_eggs': False
}
INFECTED_FRACTION = 0.2  # 20% initially infected
CI_STRENGTH = 1.0        # CI strength (can be 0.5, 0.75, or 1.0)
USE_GPU = True           # Set True to use GPU acceleration (requires PyTorch)

# (Optional) Use ParameterSet for stochastic parameters
# params = ParameterSet()
# env = Environment(GRID_SIZE, INITIAL_POPULATION, wolbachia_effects,
#                   infected_fraction=INFECTED_FRACTION, param_set=params, 
#                   ci_strength=params.ci_strength,
#                   multiple_mating=True, use_gpu=USE_GPU)
# By default, use fixed CI_STRENGTH and no ParameterSet:
env = Environment(GRID_SIZE, INITIAL_POPULATION, wolbachia_effects, 
                  infected_fraction=INFECTED_FRACTION, 
                  ci_strength=CI_STRENGTH, multiple_mating=True, 
                  use_gpu=USE_GPU)

# Run the simulation for one year (8760 hours) or until infection fixes in the population
with tqdm(total=8760, desc="Simulating") as pbar:
    for hour in range(8760):
        env.run_simulation_step()
        # Update progress bar with current infection rate and population size
        pbar.set_description(f"Simulating (Infection Rate: {env.infected_fraction:.2f}, Population: {len(env.population)}; Eggs: {len(env.eggs)})")
        pbar.update(1)
        # Stop early if infection reaches 100%
        if env.infected_fraction >= 1.0:
            print("All beetles are now infected. Ending simulation early.")
            break

# Plot results after simulation
plt.figure(figsize=(10, 5))
# Population size over time
plt.subplot(1, 2, 1)
plt.plot(env.population_size, label='Population Size')
plt.title('Population Size Over Time')
plt.xlabel('Time (hours)')
plt.ylabel('Population Size')
plt.legend()
# Infection rate over time
plt.subplot(1, 2, 2)
plt.plot(env.infection_history, label='Infection Rate', color='orange')
plt.title('Infection Rate Over Time')
plt.xlabel('Time (hours)')
plt.ylabel('Infection Prevalence')
plt.legend()
plt.tight_layout()
plt.show()
