import random

class ParameterSet:
    """
    Stores physiological and behavioral parameters with predefined ranges, sampling them stochastically.
    These introduce variability into simulations.
    Attributes sampled on initialization:
        fecundity_increase_factor (float): Multiplier >1.0 for increased fecundity (if Wolbachia provides fertility benefit).
        fecundity_decrease_factor (float): Multiplier <1.0 for reduced fecundity (if Wolbachia imposes a fertility cost).
        ci_strength (float): Cytoplasmic incompatibility strength (e.g., 0.5, 0.75, or 1.0).
        infected_male_advantage (float): Advantage factor for infected male sperm in competition (not used directly in current logic).
        male_offspring_rate (float): Probability an offspring is male under male-killing (typically around 0.1).
        female_mating_interval (int): Base mating cooldown for females, in hours.
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
