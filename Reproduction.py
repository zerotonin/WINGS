import random
from Beetle import Beetle

class Reproduction:
    """
    Handles the reproduction process for beetles, including Wolbachia effects like CI and male-killing.
    Attributes:
        grid_size (int): Size of the environment grid.
        wolbachia_effects (dict): Active Wolbachia effect toggles.
        environment (Environment): Reference to the simulation environment.
        egg_laying_max (int): Max number of eggs a female can lay at once.
    """
    def __init__(self, environment):
        self.grid_size = environment.grid_size
        self.wolbachia_effects = environment.wolbachia_effects
        self.environment = environment
        self.egg_laying_max = 15

    def mate(self, female, male):
        """
        Facilitates mating between a female and male beetle, returning offspring (eggs).
        Applies cytoplasmic incompatibility (CI) if active.
        Returns a list of offspring (Beetle objects with age 0) or an empty list if no eggs survive.
        """
        # Only allow female-male pairings
        if female.sex != 'female' or male.sex != 'male':
            return []
        # Cytoplasmic Incompatibility: infected male with uninfected female
        if (self.wolbachia_effects.get('cytoplasmic_incompatibility', False) and 
                male.infected and not female.infected):
            # CI is active for this pair
            if self.environment.ci_strength < 1.0:
                # Partial CI: some eggs may survive
                offspring = self.generate_offspring(female)
                survivors = []
                for egg in offspring:
                    # Each egg survives with probability (1 - ci_strength)
                    if random.random() >= self.environment.ci_strength:
                        survivors.append(egg)
                return survivors
            else:
                # Full CI: no offspring from this mating
                return []
        # If no CI conditions block reproduction, generate offspring normally
        return self.generate_offspring(female)

    def generate_offspring(self, female):
        """
        Generates offspring (as Beetle objects) for a female beetle after a mating event (CPU mode).
        Applies male-killing effect and maternal transmission of infection.
        """
        offspring_count = self.determine_offspring_count(female)
        offspring_list = []
        for _ in range(offspring_count):
            # Determine offspring sex
            if female.infected and self.wolbachia_effects.get('male_killing', False):
                # Male-killing active: heavily skew sex ratio towards female
                if hasattr(self.environment, 'params') and self.environment.params is not None and hasattr(self.environment.params, 'male_offspring_rate'):
                    male_probability = self.environment.params.male_offspring_rate
                else:
                    male_probability = 0.1  # default 10% chance offspring is male
                sex = 'male' if random.random() < male_probability else 'female'
            else:
                sex = random.choice(['male', 'female'])
            # Position offspring near the mother
            offspring_position = self.get_nearby_position(female.position)
            # Offspring inherits mother's infection status (Wolbachia is maternally transmitted)
            offspring_infected = female.infected
            # Create the Beetle (age 0 by default in Beetle.__init__)
            offspring_list.append(Beetle(offspring_position, offspring_infected, sex, self.environment))
        return offspring_list

    def determine_offspring_count(self, female):
        """
        Determines how many eggs a female will lay from a mating event.
        If Wolbachia infection affects fecundity (increase or reduction), adjust the base egg count.
        """
        egg_num = random.randint(1, self.egg_laying_max)
        if female.infected:
            inc_effect = self.wolbachia_effects.get('increased_eggs', False)
            red_effect = self.wolbachia_effects.get('reduced_eggs', False)
            # If both effects are toggled, no net change
            if inc_effect and red_effect:
                pass
            elif inc_effect and not red_effect:
                # Increased fecundity: raise egg count
                factor = (self.environment.params.fecundity_increase_factor 
                          if hasattr(self.environment, 'params') and self.environment.params 
                          else 1.35)
                egg_num = int(round(egg_num * factor))
            elif red_effect and not inc_effect:
                # Reduced fecundity: lower egg count
                factor = (self.environment.params.fecundity_decrease_factor 
                          if hasattr(self.environment, 'params') and self.environment.params 
                          else 0.8)
                egg_num = int(round(egg_num * factor))
        return egg_num

    def get_nearby_position(self, position):
        """
        Generates a new position (within 1 unit in x and y) near the given position.
        Uses toroidal wrapping if the position goes out of bounds.
        """
        new_x = (position[0] + random.randint(-1, 1)) % self.grid_size
        new_y = (position[1] + random.randint(-1, 1)) % self.grid_size
        return (new_x, new_y)

    def batch_mating_events(self, sim, female_indices, male_indices):
        """
        Vectorized offspring generation for multiple mating pairs (GPU mode).
        sim (int): Index of the simulation batch.
        female_indices (List[int]): Population indices of mothers.
        male_indices (List[int]): Population indices of fathers.
        Returns a dict with offspring attributes (positions, infected, sex, age, life) for all offspring.
        """
        torch = self.environment.torch  # use the same torch module (device context) as the environment
        device = self.environment.device
        num_pairs = len(female_indices)
        if num_pairs == 0:
            return {'count': 0}
        # Convert indices to tensors on the target device
        female_idx_t = torch.tensor(female_indices, device=device, dtype=torch.long)
        male_idx_t   = torch.tensor(male_indices,   device=device, dtype=torch.long)
        # Random base number of eggs per mating (between 1 and egg_laying_max, inclusive)
        eggs_per_pair = torch.randint(1, self.egg_laying_max + 1, (num_pairs,), device=device, dtype=torch.long)
        # Adjust fecundity based on Wolbachia effects (if any)
        if self.wolbachia_effects.get('increased_eggs', False) or self.wolbachia_effects.get('reduced_eggs', False):
            mothers_infected = self.environment.infected[sim, female_idx_t]  # boolean mask for infected mothers
            if self.wolbachia_effects.get('increased_eggs', False) and not self.wolbachia_effects.get('reduced_eggs', False):
                factor = (self.environment.params.fecundity_increase_factor 
                          if hasattr(self.environment, 'params') and self.environment.params 
                          else 1.35)
                eggs_per_pair[mothers_infected] = torch.round(
                    eggs_per_pair[mothers_infected].float() * factor
                ).to(torch.long)
            elif self.wolbachia_effects.get('reduced_eggs', False) and not self.wolbachia_effects.get('increased_eggs', False):
                factor = (self.environment.params.fecundity_decrease_factor 
                          if hasattr(self.environment, 'params') and self.environment.params 
                          else 0.8)
                eggs_per_pair[mothers_infected] = torch.round(
                    eggs_per_pair[mothers_infected].float() * factor
                ).to(torch.long)
            # If both increased_eggs and reduced_eggs are True, we skip any adjustment.
        # Apply cytoplasmic incompatibility (CI) effect
        if self.wolbachia_effects.get('cytoplasmic_incompatibility', False):
            male_infected = self.environment.infected[sim, male_idx_t]
            female_infected = self.environment.infected[sim, female_idx_t]
            ci_pairs_mask = male_infected & ~female_infected  # mask of pairs subject to CI
            if ci_pairs_mask.any().item():
                if self.environment.ci_strength >= 1.0:
                    # Full CI: no offspring for these pairs
                    eggs_per_pair[ci_pairs_mask] = 0
                elif self.environment.ci_strength > 0.0:
                    # Partial CI: randomly determine survival of eggs for affected pairs
                    affected_idx = ci_pairs_mask.nonzero(as_tuple=False).squeeze()
                    if affected_idx.numel() > 0:
                        max_eggs_ci = int(eggs_per_pair[affected_idx].max().item())
                        if max_eggs_ci > 0:
                            # Random matrix to decide survival of each potential egg
                            rand_matrix = torch.rand((affected_idx.shape[0], max_eggs_ci), device=device)
                            # Current egg counts for each affected pair (column vector)
                            lengths = eggs_per_pair[affected_idx].unsqueeze(1)
                            # Mask for positions that represent actual eggs (within lengths) 
                            valid = torch.arange(max_eggs_ci, device=device).expand(affected_idx.shape[0], max_eggs_ci) < lengths
                            # An egg survives if random >= ci_strength (and position is valid)
                            survive_mask = (rand_matrix >= self.environment.ci_strength) & valid
                            survivors_count = survive_mask.sum(dim=1).to(torch.long)
                            eggs_per_pair[affected_idx] = survivors_count
        # Determine total offspring to generate
        total_offspring = int(eggs_per_pair.sum().item())
        if total_offspring == 0:
            return {'count': 0}
        # Repeat each mother index according to how many offspring she produces
        mother_indices_for_offspring = female_idx_t.repeat_interleave(eggs_per_pair)
        # Retrieve mother attributes for each offspring
        mother_positions = self.environment.positions[sim, mother_indices_for_offspring, :]  # shape [total_offspring, 2]
        mother_infected = self.environment.infected[sim, mother_indices_for_offspring]       # shape [total_offspring]
        # Assign random offsets (Δx, Δy in {-1,0,1}) for each offspring and apply toroidal wrap
        offsets = torch.randint(-1, 2, (total_offspring, 2), device=device, dtype=torch.long)
        new_positions_x = (mother_positions[:, 0] + offsets[:, 0].float()) % self.grid_size
        new_positions_y = (mother_positions[:, 1] + offsets[:, 1].float()) % self.grid_size
        new_positions = torch.stack((new_positions_x, new_positions_y), dim=1)
        # Inherited infection status for offspring (True if mother is infected)
        new_infected = mother_infected.clone()
        # Determine sex for each offspring (male=1, female=0)
        male_killing = self.wolbachia_effects.get('male_killing', False)
        if male_killing:
            base_prob = 0.5
            male_prob = (self.environment.params.male_offspring_rate if 
                         (hasattr(self.environment, 'params') and self.environment.params and 
                          hasattr(self.environment.params, 'male_offspring_rate')) else 0.1)
            probs = base_prob * torch.ones(total_offspring, device=device)
            probs[mother_infected] = male_prob  # infected mothers have mostly female offspring
        else:
            probs = 0.5 * torch.ones(total_offspring, device=device)
        rand_vals = torch.rand(total_offspring, device=device)
        male_mask = rand_vals < probs
        new_sex = male_mask.to(torch.long)
        # Assign life expectancy and starting age for each offspring
        new_life = torch.randint(280*24, 450*24, (total_offspring,), device=device, dtype=torch.long)
        new_age = torch.zeros(total_offspring, device=device, dtype=torch.long)
        # Return all offspring attributes for integration into the environment
        return {
            'count': total_offspring,
            'positions': new_positions,
            'infected': new_infected,
            'sex': new_sex,
            'age': new_age,
            'life': new_life
        }
