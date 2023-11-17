from Environment import Environment
import matplotlib.pyplot as plt
from tqdm import tqdm
# Simulation parameters
grid_size = 500  # Adjust as needed
initial_population = 100  # Adjust as needed
wolbachia_effects = {'cytoplasmic_incompatibility': True, 'male_killing': True, 'increased_exploration_rate': True,'increased_eggs':True}
infected_fraction = 0.2  # 10% initially infected

# Create the environment
env = Environment(grid_size, initial_population, wolbachia_effects, infected_fraction)

# Run the simulation for one year (8760 hours) or until all beetles are infected
with tqdm(total=8760, desc="Simulating") as pbar:
    for hour in range(8760):
        env.run_simulation_step()
        
        # Update the progress bar description with current infection rate and population size
        pbar.set_description(f"Simulating (Infection Rate: {env.infected_fraction:.2f}, Population Size: {len(env.population)}), Eggs: {len(env.eggs)}")
        pbar.update(1)

        if env.infected_fraction >= 1.0:
            print("All beetles are infected.")
            break

# Plotting results
plt.figure(figsize=(12, 6))

# Population size over time
plt.subplot(1, 2, 1)
plt.plot(env.population_size, label='Population Size')
plt.title('Population Size Over Time')
plt.xlabel('Time (hours)')
plt.ylabel('Population Size')
plt.legend()

# Infection rate over time
plt.subplot(1, 2, 2)
plt.plot(env.infection_history, label='Infection Rate')
plt.title('Infection Rate Over Time')
plt.xlabel('Time (hours)')
plt.ylabel('Infection Rate')
plt.legend()

plt.tight_layout()
plt.show()
