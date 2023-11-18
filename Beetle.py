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
        last_mating_time (int): The last time the beetle mated, in hours since the start of the simulation.
        grid_size (int): The size of the simulation environment.

    Parameters:
        position (tuple): The initial position of the beetle in the environment.
        infected (bool): Initial infection status of the beetle.
        sex (str): The sex of the beetle.
        environment (Environment): The environment to which the beetle belongs.
        age (int, optional): The initial age of the beetle. Defaults to 0.
        mating_cooldown (int, optional): The mating cooldown period in hours. Defaults to 240.
    """

    def __init__(self, position, infected, sex, environment, age=0, mating_cooldown=240):
        self.position = position
        self.infected = infected
        self.sex = sex
        self.age = age
        self.environment = environment
        self.max_life_expectancy = self.generate_life_expectancy()
        self.mating_cooldown = mating_cooldown
        self.last_mating_time = -1 * mating_cooldown
        self.grid_size = self.environment.grid_size

    def generate_life_expectancy(self):
        """
        Generates the beetle's life expectancy based on a normal distribution.

        The life expectancy is centered around 9 months with a standard deviation of about 15 days.

        Returns:
            float: The generated life expectancy of the beetle in hours.
        """
        mu = 12 * 30 * 24  # Mean: 12 months in hours
        sigma = 3 * 30 * 24  # Standard deviation: ~15 days in hours
        life_expectancy = np.random.normal(mu, sigma)
        return max(0, min(life_expectancy, 13 * 30 * 24))

    def levy_flight_step(self):
        """
        Performs a movement step based on the Levy flight pattern.

        The step size is determined by a Pareto distribution, and the direction is random.
        The beetle's position is updated with toroidal wrapping around the environment boundaries.
        """
        step_size = np.random.pareto(a=1.5) + 1
        angle = np.random.uniform(0, 2 * np.pi)

        new_x = (self.position[0] + step_size * np.cos(angle)) % self.grid_size
        new_y = (self.position[1] + step_size * np.sin(angle)) % self.grid_size
        self.position = (new_x, new_y)

    def move(self):
        """
        Updates the beetle's position by performing a Levy flight step.

        Movement occurs only if the beetle has hatched (age is at least one month).
        """
        if self.age >= 31 * 24:  # Hatches after one month
            self.levy_flight_step()

    def update_last_mating_time(self, current_time):
        """
        Updates the last mating time of the beetle.

        Parameters:
            current_time (int): The current time in the simulation, in hours.
        """
        self.last_mating_time = current_time

    def can_mate(self, current_time):
        """
        Determines whether the beetle can mate based on the current time and mating cooldown.
        The male cooldown is always 10% of the female cooldown periode.

        Parameters:
            current_time (int): The current time in the simulation, in hours.

        Returns:
            bool: True if the beetle can mate, False otherwise.
        """
        cooldown = self.mating_cooldown if self.sex == 'female' else self.mating_cooldown / 10
        return current_time - self.last_mating_time >= cooldown
