import numpy as np

class Beetle:
    """A single beetle agent in the CPU-based ABM.

    Represents an individual *Tribolium* beetle with spatial position,
    infection status, sex, age, and mating state. Movement follows a
    Lévy flight distribution, with infected beetles optionally moving
    further (increased exploration rate).

    Attributes:
        position (tuple): ``(x, y)`` coordinates on the toroidal grid.
        infected (bool): Whether the beetle carries *Wolbachia*.
        sex (str): ``'male'`` or ``'female'``.
        age (int): Current age in hours.
        environment (Environment): Reference to the simulation environment.
        max_life_expectancy (float): Maximum lifespan in hours.
        mating_cooldown (int): Hours between successive matings.
        last_mating_time (int): Hour of the most recent mating event.
        grid_size (int): Side length of the simulation grid.
    """
    def __init__(self, position, infected, sex, environment, age=0, mating_cooldown=48):
        """Initialise a beetle agent.

        Args:
            position (tuple): Initial ``(x, y)`` grid coordinates.
            infected (bool): *Wolbachia* infection status.
            sex (str): ``'male'`` or ``'female'``.
            environment (Environment): The simulation environment.
            age (int): Starting age in hours. Defaults to ``0``.
            mating_cooldown (int): Minimum hours between matings.
                Defaults to ``48`` (2 days).
        """
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
        """Move the beetle using a Lévy flight step.

        Step length is drawn from a power-law distribution.  Infected
        beetles with the *increased_exploration_rate* phenotype receive
        a 1.4× movement multiplier, increasing their effective mating
        radius.  The grid wraps toroidally.
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
