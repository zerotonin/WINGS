from Reproduction import Reproduction
from Beetle import Beetle
import numpy as np
import random

class Environment:
    """
    Represents the simulation environment for the Wolbachia-infected beetle population.
    Manages the beetle population, their interactions, and tracks metrics like population size and infection rates.

    Parameters:
        size (int): Size of the simulation grid.
        initial_population (int): Initial number of beetles.
        wolbachia_effects (dict): Which Wolbachia effects are active (CI, male_killing, etc).
        infected_fraction (float): Initial fraction of the population infected.
        max_population (int): Maximum allowed population size.
        max_eggs (int): Maximum allowed number of eggs (unhatched offspring).
        male_to_female_ratio (float): Desired male:female ratio for initial population.
        param_set (ParameterSet, optional): Provides randomized parameters (fecundity, etc).
        ci_strength (float): CI strength (0.0–1.0, fraction of incompatible eggs that fail).
        multiple_mating (bool): Whether females can mate multiple times per cycle.
        use_gpu (bool): Whether to use GPU acceleration with PyTorch.
    """
    def __init__(self, size, initial_population, wolbachia_effects, 
                 infected_fraction=0.1, max_population=50, max_eggs=40, 
                 male_to_female_ratio=0.5,
                 param_set=None, ci_strength=1.0, multiple_mating=True, 
                 use_gpu=False):
        self.grid_size = size
        self.population = []
        self.wolbachia_effects = wolbachia_effects
        self.infected_fraction = infected_fraction
        self.infection_history = [self.infected_fraction]
        self.population_size = [initial_population]
        self.initial_infected_count = int(np.round(initial_population * infected_fraction))
        self.male_to_female_ratio = male_to_female_ratio

        # Initialize parameter set and reproduction settings
        self.params = param_set
        self.ci_strength = ci_strength
        self.multiple_mating = multiple_mating
        if self.params is not None:
            # If a ParameterSet is provided, override CI strength with its value
            self.ci_strength = self.params.ci_strength

        # GPU acceleration setup
        self.use_gpu = use_gpu
        if self.use_gpu:
            try:
                import torch
                self.torch = torch
                # Use CUDA if available; otherwise fall back to CPU (PyTorch tensor operations)
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            except ImportError:
                print("PyTorch not installed. Running without GPU acceleration.")
                self.use_gpu = False

        # Initialize beetle population and reproduction system
        self.initialize_population(initial_population)
        self.reproduction_system = Reproduction(self)
        self.max_population = max_population
        self.sim_time = 0
        self.eggs = []
        self.max_eggs = max_eggs
        self.mating_distance = 5.0

        # Cache initial population data for GPU processing, if applicable
        if self.use_gpu:
            self.update_population_arrays()

    def initialize_population(self, initial_population):
        """
        Initializes the beetle population with the given size.
        Places beetles randomly in the central area and assigns age, sex, and infection status.
        """
        infected_count = 0
        male_count = 0
        female_count = 0

        for _ in range(initial_population):
            position = self.generate_position_in_central_third()
            # Random initial age in hours (~1–3.5 months)
            age = np.random.randint(889, 2500)

            # Determine sex and infection status for the new beetle
            if infected_count < np.round(self.initial_infected_count / 2):
                # First half of infected_count: infected females
                sex = 'female'
                infected = True
                infected_count += 1
                female_count += 1
            elif infected_count < self.initial_infected_count:
                # Second half of infected_count: infected males
                sex = 'male'
                infected = True
                infected_count += 1
                male_count += 1
            else:
                # Assign remaining beetles to reach the desired male:female ratio
                if male_count / max(female_count, 1) < self.male_to_female_ratio:
                    sex = 'male'
                    male_count += 1
                else:
                    sex = 'female'
                    female_count += 1
                infected = False

            # Determine mating cooldown (females default 48h or from params; males use 10% of this in Beetle.can_mate)
            mating_cd = self.params.female_mating_interval if (
                hasattr(self, 'params') and self.params is not None and hasattr(self.params, 'female_mating_interval')
            ) else 48
            beetle = Beetle(position, infected, sex, self, age=age, mating_cooldown=mating_cd)
            self.population.append(beetle)

    def generate_position_in_central_third(self):
        """Generates a random (x, y) position within the central third region of the grid."""
        third = self.grid_size // 3
        x = random.randint(third, 2 * third)
        y = random.randint(third, 2 * third)
        return (x, y)

    def run_simulation_step(self):
        """
        Executes a single hour of simulation:
        - Moves all beetles (vectorized on GPU if enabled)
        - Ages and hatches eggs
        - Handles mating events and enforces population limits
        - Updates infection statistics
        """
        # 1. Move all beetles (Lévy flight step for each)
        if self.use_gpu and len(self.population) > 0:
            # Parallel movement update using PyTorch for all beetles
            positions_t = self.torch.tensor([b.position for b in self.population], 
                                            dtype=self.torch.float32, device=self.device)
            # Lévy flight step: sample step sizes and directions
            U = self.torch.rand(len(self.population), device=self.device)
            step_sizes = U.pow(-1 / 1.5)
            angles = 2 * np.pi * self.torch.rand(len(self.population), device=self.device)
            dx = step_sizes * self.torch.cos(angles)
            dy = step_sizes * self.torch.sin(angles)
            # Compute new positions with toroidal wrapping
            new_positions = positions_t.clone()
            new_positions[:, 0] = (positions_t[:, 0] + dx) % self.grid_size
            new_positions[:, 1] = (positions_t[:, 1] + dy) % self.grid_size
            # Transfer new positions to CPU and update each beetle object
            new_positions_cpu = new_positions.cpu().numpy()
            for i, beetle in enumerate(self.population):
                beetle.position = (new_positions_cpu[i, 0], new_positions_cpu[i, 1])
                beetle.age += 1
        else:
            # Sequential movement (CPU) for each beetle
            for beetle in self.population:
                beetle.move()
                beetle.age += 1

        # 2. Age and possibly hatch eggs
        for egg in list(self.eggs):  # make a copy of list to allow removal
            egg.age += 1
            # If egg has matured (~23 days or 552 hours), hatch it into the population
            if egg.age > 552:
                self.population.append(egg)
                self.eggs.remove(egg)

        # 3. Increment time
        self.sim_time += 1
        # Remove beetles that exceeded their life expectancy
        self.retire_old_beetles()
        # Handle mating events (produce offspring eggs)
        self.check_for_mating()
        # Enforce population limits (cull excess beetles/eggs randomly)
        self.population = self.grim_reaper(self.population, self.max_population)
        self.eggs = self.grim_reaper(self.eggs, self.max_eggs)
        # Update infection statistics after this time step
        self.check_infection_status()
        self.population_size.append(len(self.population))

        # 4. Update cached population data for GPU computations (if applicable)
        if self.use_gpu:
            self.update_population_arrays()

    def grim_reaper(self, target_list, max_size):
        """
        Ensures the list does not exceed max_size by randomly removing surplus elements.
        Returns a pruned list (or the original list if within limit).
        """
        excess = len(target_list) - max_size
        if excess > 0:
            # Randomly sample the survivors (keep max_size items)
            return random.sample(target_list, len(target_list) - excess)
        return target_list

    def retire_old_beetles(self):
        """Removes beetles that have exceeded their life expectancy from the population."""
        self.population = [b for b in self.population if b.age <= b.max_life_expectancy]

    def check_infection_status(self):
        """Calculates the current infected fraction of the population and logs it."""
        if len(self.population) == 0:
            self.infected_fraction = 0.0
        else:
            infected_count = sum(beetle.infected for beetle in self.population)
            self.infected_fraction = infected_count / len(self.population)
        self.infection_history.append(self.infected_fraction)

    def check_for_mating(self):
        """
        Checks each female for mating opportunities with nearby males.
        Allows multiple matings if enabled. Produces offspring eggs for each successful mating.
        """
        current_time = self.sim_time
        for female in filter(lambda b: b.sex == 'female' and b.can_mate(current_time), self.population):
            mates_found = 0
            for male in filter(lambda b: b.sex == 'male' and b.can_mate(current_time), self.population):
                if self.is_within_mating_distance(female, male):
                    # Perform mating and produce offspring (as eggs)
                    female.update_last_mating_time(current_time)
                    male.update_last_mating_time(current_time)
                    offspring_eggs = self.reproduction_system.mate(female, male)
                    # Add offspring (eggs) to the egg list
                    self.eggs.extend(offspring_eggs)
                    mates_found += 1
                    # If multiple mating is disallowed or the female has mated twice, stop checking further males
                    if not self.multiple_mating or mates_found >= 2:
                        break

    def is_within_mating_distance(self, female, male):
        """
        Determines if two beetles are within mating distance.
        If 'increased_exploration_rate' is in effect and the female is infected, expands mating range by 40%.
        """
        distance = np.linalg.norm(np.array(female.position) - np.array(male.position))
        if female.infected and self.wolbachia_effects.get('increased_exploration_rate', False):
            return distance <= self.mating_distance * 1.4
        return distance <= self.mating_distance

    def update_population_arrays(self):
        """
        Updates cached tensors of population positions and infection statuses for GPU-based reproduction.
        Called after any change in the population when GPU is in use.
        """
        if not hasattr(self, 'torch'):
            return  # No torch available (should not happen if use_gpu is True)
        # Tensor of all beetle positions (shape: [1, N, 2] for compatibility with batch ops)
        positions = self.torch.tensor([b.position for b in self.population],
                                      dtype=self.torch.float32, device=self.device)
        # Tensor of infection status for each beetle (True/False)
        infected = self.torch.tensor([b.infected for b in self.population],
                                     dtype=self.torch.bool, device=self.device)
        # Store with a simulation batch dimension (sim index 0 for this single simulation)
        self.positions = positions.unsqueeze(0)
        self.infected = infected.unsqueeze(0)
