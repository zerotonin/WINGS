import random

class ParameterSet:
    """Stochastic physiological and behavioural parameters.

    Each attribute is sampled from a biologically plausible range
    on instantiation, introducing inter-individual variability
    into the simulation.

    Attributes:
        fecundity_increase_factor (float): Multiplier (>1) for
            *Wolbachia*-induced fecundity increase. Range: 1.1–1.3.
        fecundity_decrease_factor (float): Multiplier (<1) for
            *Wolbachia*-induced fecundity cost. Range: 0.8–0.9.
        ci_strength (float): Cytoplasmic incompatibility strength.
            Sampled from {0.5, 0.75, 1.0}.
        infected_male_advantage (float): Sperm competition advantage
            factor for infected males. Range: 0.7–0.9.
        male_offspring_rate (float): Fraction of offspring that are
            male under male killing. Range: 0.1–0.2.
        female_mating_interval (int): Base mating cooldown in hours.
            Fixed at 48 (2 days).
    """
    def __init__(self):
        # Sample fecundity effect factors within plausible ranges (10–30% increase, 10–20% decrease)
        self.fecundity_increase_factor = random.uniform(1.1, 1.3)
        self.fecundity_decrease_factor = random.uniform(0.8, 0.9)
        # Choose CI strength from common values
        self.ci_strength = random.choice([0.5, 0.75, 1.0])
        # Infected male advantage (not explicitly used in simulation logic)
        self.infected_male_advantage = random.uniform(0.7, 0.9)
        # Male offspring rate under male-killing (e.g., ~10% males)
        self.male_offspring_rate = random.uniform(0.1, 0.2)
        # Female mating interval (hours) – typically ~48 hours (2 days)
        self.female_mating_interval = 48
