import numpy as np

class Beetle:
    """
    A class representing a beetle in the Wolbachia simulation environment.

    Attributes:
        position (tuple): The (x, y) coordinates of the beetle in the environment.
        infected (bool): True if the beetle is infected with Wolbachia, False otherwise.
        sex (str): The sex of the beetle, either 'male' or 'female'.
        age (int): The age of the beetle in hours.
        environment (Environment): A reference to the environment the beetle is in.
        max_life_expectancy (float): The maximum life expectancy of the beetle in hours.
        mating_cooldown (int): The cooldown period in hours before the beetle can mate again.
        last_mating_time (int): The last time the beetle mated (in hours since start).
        grid_size (int): The size of the simulation environment.
    """
    def __init__(self, position, infected, sex, environment, age=0, mating_cooldown=48):
        self.position = position
        self.infected = infected
        self.sex = sex
        self.age = age
        self.environment = environment
        self.max_life_expectancy = self.generate_life_expectancy()
        self.mating_cooldown = mating_cooldown
        self.last_mating_time = -1 * mating_cooldown  # allow immediate mating at t=0
        self.grid_size = self.environment.grid_size

    def generate_life_expectancy(self):
        """
        Generates the beetle's life expectancy (in hours) based on a uniform distribution.
        Roughly between 9 and 15 months.
        """
        return np.random.randint(280*24, 450*24)

    def levy_flight_step(self):
        """
        Performs a movement step based on a Lévy flight pattern.
        The step size follows a Pareto distribution, and direction is random (0 to 2π).
        """
        step_size = np.random.pareto(a=1.5) + 1  # heavy-tailed step length
        angle = np.random.uniform(0, 2 * np.pi)
        new_x = (self.position[0] + step_size * np.cos(angle)) % self.grid_size
        new_y = (self.position[1] + step_size * np.sin(angle)) % self.grid_size
        self.position = (new_x, new_y)

    def move(self):
        """
        Updates the beetle's position by performing a Lévy flight step.
        (Movement occurs regardless of age for simplicity in this model.)
        """
        self.levy_flight_step()

    def update_last_mating_time(self, current_time):
        """Updates the last mating time of the beetle to the current time (hours)."""
        self.last_mating_time = current_time

    def can_mate(self, current_time):
        """
        Determines whether the beetle can mate based on current time and its mating cooldown.
        Males have a shorter cooldown (10% of the female cooldown period).
        """
        cooldown = self.mating_cooldown if self.sex == 'female' else self.mating_cooldown / 10
        return current_time - self.last_mating_time >= cooldown
