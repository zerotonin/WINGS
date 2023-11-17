import random
from Beetle import Beetle

class Reproduction:
    """
    Handles the reproduction process for beetles in the Wolbachia simulation environment.

    This class is responsible for mating beetles and generating offspring, taking into account the effects of Wolbachia.

    Attributes:
        grid_size (int): The size of the grid in the environment.
        wolbachia_effects (dict): A dictionary containing the status of various Wolbachia effects.
        environment (Environment): A reference to the environment the reproduction system is part of.
        egg_laying_max (int): The maximum number of eggs a female beetle can lay at once.
    """

    def __init__(self, environment):
        """
        Initializes the Reproduction system with a reference to the environment.

        Parameters:
            environment (Environment): The environment to which the reproduction system belongs.
        """
        self.grid_size = environment.grid_size
        self.wolbachia_effects = environment.wolbachia_effects
        self.environment = environment
        self.egg_laying_max = 15

    def mate(self, female, male):
        """
        Facilitates the mating of a female and male beetle, producing offspring.

        Considers Wolbachia effects such as cytoplasmic incompatibility. Offspring generation is deferred to generate_offspring method.

        Parameters:
            female (Beetle): The female beetle participating in mating.
            male (Beetle): The male beetle participating in mating.

        Returns:
            list: A list of offspring (Beetle objects), empty if mating is unsuccessful.
        """
        if female.age < 888 or male.age < 888:
            return []  # Both beetles must be older than 888 hours (37 days) to mate
        
        if female.sex != 'female' or male.sex != 'male':
            return []  # Ensure correct sex pairing
        
        if self.wolbachia_effects['cytoplasmic_incompatibility'] and male.infected and not female.infected:
            return []  # Prevent offspring in case of cytoplasmic incompatibility

        return self.generate_offspring(female)

    def generate_offspring(self, female):
        """
        Generates offspring for a female beetle.

        Applies male-killing effect if enabled and the female is infected. The number of offspring is determined by determine_offspring_count method.

        Parameters:
            female (Beetle): The female beetle generating offspring.

        Returns:
            list: A list of offspring (Beetle objects).
        """
        offspring_count = self.determine_offspring_count(female)
        offspring = []

        for _ in range(offspring_count):
            if female.infected and self.wolbachia_effects['male_killing']:
                # Apply male killing effect with a 1 in 10 chance of being male
                sex = 'male' if random.randint(1, 10) == 1 else 'female'
            else:
                sex = random.choice(['male', 'female']) 
                
            offspring_position = self.get_nearby_position(female.position)
            offspring_infected = female.infected  # Offspring inherit Wolbachia infection from the mother
            offspring.append(Beetle(offspring_position, offspring_infected, sex, self.environment))

        return offspring

    def determine_offspring_count(self, female):
        """
        Determines the number of offspring a female beetle will produce.

        The number is influenced by the beetle's infection status and the 'increased_eggs' Wolbachia effect.

        Parameters:
            female (Beetle): The female beetle being considered.

        Returns:
            int: The number of offspring to be produced.
        """
        egg_num = random.randint(1, self.egg_laying_max)
        if female.infected and self.wolbachia_effects['increased_eggs']:
            egg_num = int(round(egg_num * 1.35))  # Increase egg number by 35% if infected
        return egg_num
    
    def get_nearby_position(self, position):
        """
        Generates a nearby position for offspring based on the parent's position.

        The new position is randomly chosen within one unit distance from the parent, respecting the environment boundaries.

        Parameters:
            position (tuple): The position of the parent beetle.

        Returns:
            tuple: The generated nearby position for the offspring.
        """
        new_x = (position[0] + random.randint(-1, 1)) % self.grid_size
        new_y = (position[1] + random.randint(-1, 1)) % self.grid_size
        return new_x, new_y
