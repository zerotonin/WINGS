from Reproduction import Reproduction
from Beetle import Beetle
import numpy as np
import random

class Environment:
    """
    Represents the simulation environment for the Wolbachia-infected beetle population.

    This class manages the beetle population, their interactions, and tracks various metrics like population size and infection rates.

    Attributes:
        grid_size (int): The size of the simulation environment.
        population (list): The list of Beetle objects representing the beetle population.
        wolbachia_effects (dict): Dictionary indicating the presence and effects of Wolbachia.
        infected_fraction (float): The initial fraction of the population that is infected.
        infection_history (list): Historical record of the infection rate in the population.
        population_size (list): Historical record of the population size.
        reproduction_system (Reproduction): The system handling the reproduction of beetles.
        max_population (int): Maximum allowed population size in the environment.
        sim_time (int): The current time in the simulation (in hours).
        eggs (list): List of Beetle objects representing unhatched eggs.
        max_eggs (int): Maximum allowed number of eggs in the environment.

    Parameters:
        size (int): The size of the simulation grid.
        initial_population (int): The initial number of beetles in the population.
        wolbachia_effects (dict): Specifies the effects of Wolbachia on the population.
        infected_fraction (float): The initial fraction of the population that is infected.
        max_population (int): The maximum number of beetles allowed in the environment.
        max_eggs (int): The maximum number of eggs allowed in the environment.
    """

    def __init__(self, size, initial_population, wolbachia_effects, infected_fraction=0.1,max_population=100, max_eggs=50):
        self.grid_size = size
        self.population = []
        self.wolbachia_effects = wolbachia_effects
        self.infected_fraction = infected_fraction
        self.infection_history = [self.infected_fraction]
        self.population_size = [initial_population]
        self.initialize_population(initial_population)
        self.reproduction_system = Reproduction(self)
        self.max_population = max_population
        self.sim_time = 0
        self.eggs = list()
        self.max_eggs = max_eggs
        self.mating_distance = 5.0 

    def initialize_population(self, initial_population):
        """
        Initializes the beetle population with the given initial population size.

        Beetles are randomly placed in the central third of the environment and assigned random age, sex, and infection status.

        Parameters:
            initial_population (int): The number of beetles to initialize in the population.
        """
        for _ in range(initial_population):
            position = self.generate_position_in_central_third()
            infected = random.random() < self.infected_fraction
            sex = 'male' if random.random() < 0.5 else 'female'
            age = 889 #random.randint(888, 2000)  # Age in hours
            beetle = Beetle(position, infected, sex, self, age)
            self.population.append(beetle)

    def generate_position_in_central_third(self):
        """
        Generates a random position within the central third of the environment.

        Returns:
            tuple: A tuple representing the (x, y) coordinates of the position.
        """
        third = self.grid_size // 3
        x = random.randint(third, 2 * third)
        y = random.randint(third, 2 * third)
        return (x, y)

    def run_simulation_step(self):
        """
        Executes a single step of the simulation.

        This method updates the position and age of each beetle, processes egg hatching, and handles mating and population management.
        """

        # Move all beetles
        for beetle in self.population:
            beetle.move()
            beetle.age += 1

        #Age and Hatch eggs
        for egg in self.eggs:
            egg.age +=1
            if egg.age> 744 + 144: # 744 hours till hatching plus 144 hours till sexual development
                self.population.append(egg)
                self.eggs.remove(egg)
        
        # Update simulation time
        self.sim_time +=1
        # Remove beetles that have exceeded their life expectancy
        self.retire_old_beetles()
        # Check for potential mating pairs
        self.check_for_mating()
        # Invoke the Grim Reaper to maintain population size
        self.population = self.grim_reaper(self.population, self.max_population)
        self.eggs = self.grim_reaper(self.eggs, self.max_eggs)  # Assuming self.max_eggs is defined
        # Check the infection status after movement and other actions
        self.check_infection_status()
        self.population_size.append(len(self.population))
        

    def grim_reaper(self, target_list, max_size):
        """
        Reduces the size of a given list to a specified maximum, randomly removing elements if necessary.

        Parameters:
            target_list (list): The list from which elements are to be removed.
            max_size (int): The maximum allowed size of the list.

        Returns:
            list: The modified list after applying the grim reaper process.
        """
        # Calculate the number of items to remove, with some random fluctuation
        excess = len(target_list) - max_size
        if excess > 0:
            items_to_remove = excess + random.randint(-10, 10)
            items_to_remove = max(0, min(items_to_remove, len(target_list)))  # Ensure valid range

            # Randomly remove the calculated number of items
            return random.sample(target_list, len(target_list) - items_to_remove)
        else:
            return target_list

    def retire_old_beetles(self):
        """
        Removes beetles from the population that have exceeded their maximum life expectancy.
        """
        # "All those moments will be lost in time, like tears in rain." 
        #                        – Roy Batty, portrayed by Rutger Hauer
        
        self.population = [beetle for beetle in self.population if beetle.age <= beetle.max_life_expectancy]
        

    def check_infection_status(self):
        """
        Updates the infection status of the beetle population.

        Calculates the current fraction of infected beetles and appends this data to the infection history.
        """
        infected_count = sum(beetle.infected for beetle in self.population)
        self.infected_fraction = infected_count/len(self.population)
        self.infection_history.append(self.infected_fraction)


    def check_for_mating(self):
        """
        Checks each beetle in the population for potential mating opportunities.

        Mating is considered based on proximity, age, sex, and mating cooldown. Successful mating results in the production of offspring.
        """
        current_time = self.sim_time
        for female in filter(lambda b: b.sex == 'female' and  b.can_mate(current_time), self.population): 
            for male in filter(lambda b: b.sex == 'male' and  b.can_mate(current_time), self.population):
                if self.is_within_mating_distance(female, male):
                    female.update_last_mating_time(current_time)
                    male.update_last_mating_time(current_time)
                    offspring = self.reproduction_system.mate(female, male)
                    # Add offspring to the population if any
                    self.eggs.extend(offspring)
                    break  # Break to prevent the female from mating again in this cycle


    def is_within_mating_distance(self, female, male):
        """
        Determines if two beetles are within a certain distance to allow for mating.

        The mating distance is increased if the female is infected with Wolbachia.

        Parameters:
            female (Beetle): The female beetle.
            male (Beetle): The male beetle.

        Returns:
            bool: True if the beetles are within mating distance, False otherwise.
        """
        distance = np.linalg.norm(np.array(female.position) - np.array(male.position))
         # Default mating distance
        if female.infected and self.wolbachia_effects['increased_exploration_rate']:
            return distance <= self.mating_distance * 1.4  # Increase distance if either beetle is infected
        else:
            return distance <= self.mating_distance