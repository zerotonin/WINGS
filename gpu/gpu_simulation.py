"""
W.I.N.G.S. GPU-Accelerated Simulation Engine
=============================================
Replaces the per-beetle Python loop with fully vectorized PyTorch tensor
operations.  Designed for NVIDIA L40S (48 GB VRAM) but will run on any
CUDA device (or CPU as fallback).

Scaling strategy
----------------
The original ABM is O(F·M) per time-step for the mating search because every
female is checked against every male.  At N = 20 000 that means ~100 M distance
evaluations per step – trivial on an L40S but wasteful in memory for N > 50 000.

This module provides **two mating backends**:

1. ``brute``  – compute the full female × male distance matrix on GPU.
   Memory: O(F·M) floats ≈ 400 MB at N = 20 000.  Fast and simple.

2. ``cell_list`` – partition the toroidal grid into cells of side length
   ≥ mating_distance, then only check the 3×3 neighbourhood of each
   female's cell.  Memory: O(N · k) where k is the mean number of
   neighbours.  Scales to N > 100 000.

Usage
-----
>>> from gpu_simulation import GPUSimulation, SimConfig
>>> cfg = SimConfig(initial_population=20_000, max_population=25_000,
...                 grid_size=500, wolbachia_effects={'cytoplasmic_incompatibility': True,
...                 'male_killing': False, 'increased_exploration_rate': True,
...                 'increased_eggs': True, 'reduced_eggs': False})
>>> sim = GPUSimulation(cfg)
>>> for day in range(365):
...     sim.step_one_day()          # 24 hourly sub-steps
...     print(sim.get_infection_rate())

Author: Adapted from WINGS ABM (Geurten et al.)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import time


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    """All tuneable knobs in one place."""

    # --- Population ---
    initial_population: int = 50
    max_population: int = 20_000       # carrying capacity for adults
    max_eggs: int = 800_000            # egg buffer cap (must be large: eggs take 23 days to
                                       # hatch, so the pipeline holds ~daily_eggs × 23 days)
    infected_fraction: float = 0.10
    male_to_female_ratio: float = 0.50 # fraction male among initial uninfected

    # --- Spatial ---
    grid_size: int = 500               # side length of the toroidal arena
    mating_distance: float = 5.0

    # --- Wolbachia effects (bool flags) ---
    wolbachia_effects: Dict[str, bool] = field(default_factory=lambda: {
        'cytoplasmic_incompatibility': True,
        'male_killing': False,
        'increased_exploration_rate': False,
        'increased_eggs': False,
        'reduced_eggs': False,
    })

    # --- Reproduction parameters ---
    egg_laying_max: int = 15
    ci_strength: float = 1.0           # 0.0–1.0
    fecundity_increase_factor: float = 1.2
    fecundity_decrease_factor: float = 0.85
    male_offspring_rate: float = 0.10  # under male-killing
    exploration_rate_boost: float = 1.4 # radius multiplier for infected females
    mating_cooldown_female: int = 48   # hours
    mating_cooldown_male: int = 5      # hours (~48/10)
    multiple_mating: bool = True       # allow up to 2 mates per female per step
    egg_hatching_age: int = 552        # hours (~23 days)

    # --- Life expectancy (hours) ---
    life_expectancy_min: int = 280 * 24  # ~280 days
    life_expectancy_max: int = 450 * 24  # ~450 days
    initial_age_min: int = 889           # hours (~37 days)
    initial_age_max: int = 2500          # hours (~104 days)

    # --- Levy flight ---
    levy_alpha: float = 1.5             # Pareto shape parameter

    # --- Density-dependent mortality mode ---
    # Controls how population regulation occurs beyond logistic birth suppression.
    #   'none'        : Only natural death (age > max_life) + hard cap. Original ABM behavior.
    #   'logistic'    : Per-capita adult death rate increases linearly with N/K.
    #   'cannibalism' : Beetle-specific: adults destroy eggs proportional to density.
    #                   Based on Tribolium literature (Daly & Ryan 1983, Park 1934).
    #   'contest'     : Above K, excess adults die each hour with probability ∝ (N-K)/N.
    mortality_mode: str = 'cannibalism'
    # Exponent for density-dependent effects (higher = sharper response near K)
    mortality_beta: float = 2.0
    # Egg cannibalism rate: fraction of eggs consumed per adult per hour at density = K
    cannibalism_rate: float = 0.0001

    # --- Backend ---
    mating_backend: str = 'cell_list'  # 'brute' or 'cell_list'
    device: str = 'cuda'               # 'cuda' or 'cpu'
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Tensor-based beetle state  (Structure-of-Arrays)
# ---------------------------------------------------------------------------
class PopulationState:
    """
    Holds the entire population (adults + eggs) as flat GPU tensors.

    We use a Structure-of-Arrays layout so that every operation
    (move, age, filter, mate) is a single vectorized kernel call
    instead of a Python loop over beetles.

    Attributes (all tensors on ``device``):
        x, y         : float32 [N]  – positions
        infected     : bool    [N]  – Wolbachia status
        is_male      : bool    [N]  – sex (True = male)
        age          : int32   [N]  – current age in hours
        max_life     : int32   [N]  – sampled life expectancy (hours)
        last_mate    : int32   [N]  – sim-time of last mating event
        is_egg       : bool    [N]  – True while in egg stage (not yet hatched)
    """

    def __init__(self, device: torch.device):
        self.device = device
        # Start empty – filled by GPUSimulation.initialize_population()
        self.x         = torch.empty(0, device=device, dtype=torch.float32)
        self.y         = torch.empty(0, device=device, dtype=torch.float32)
        self.infected  = torch.empty(0, device=device, dtype=torch.bool)
        self.is_male   = torch.empty(0, device=device, dtype=torch.bool)
        self.age       = torch.empty(0, device=device, dtype=torch.int32)
        self.max_life  = torch.empty(0, device=device, dtype=torch.int32)
        self.last_mate = torch.empty(0, device=device, dtype=torch.int32)
        self.is_egg    = torch.empty(0, device=device, dtype=torch.bool)

    @property
    def n(self) -> int:
        return self.x.shape[0]

    @property
    def n_adults(self) -> int:
        return int((~self.is_egg).sum().item())

    @property
    def n_eggs(self) -> int:
        return int(self.is_egg.sum().item())

    # --- Helpers for adding / removing individuals -------------------------

    def append(self, **kwargs):
        """Concatenate new individuals (given as keyword tensors) onto the state."""
        for attr in ('x', 'y', 'infected', 'is_male', 'age', 'max_life',
                     'last_mate', 'is_egg'):
            current = getattr(self, attr)
            new_vals = kwargs[attr]
            setattr(self, attr, torch.cat([current, new_vals], dim=0))

    def keep(self, mask: torch.Tensor):
        """Keep only individuals where ``mask`` is True (boolean tensor [N])."""
        for attr in ('x', 'y', 'infected', 'is_male', 'age', 'max_life',
                     'last_mate', 'is_egg'):
            setattr(self, attr, getattr(self, attr)[mask])

    def subsample(self, max_n: int):
        """If n > max_n, randomly keep max_n individuals (GPU grim_reaper)."""
        if self.n <= max_n:
            return
        perm = torch.randperm(self.n, device=self.device)[:max_n]
        for attr in ('x', 'y', 'infected', 'is_male', 'age', 'max_life',
                     'last_mate', 'is_egg'):
            setattr(self, attr, getattr(self, attr)[perm])


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
class GPUSimulation:
    """
    Fully GPU-vectorized WINGS simulation.

    Every time-step is one simulated **hour**.  Call ``step()`` for a single
    hour or ``step_one_day()`` for 24 hours with aggregated recording.
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

        # Device selection
        if cfg.device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            if cfg.device == 'cuda':
                print("CUDA not available – falling back to CPU.")
            self.device = torch.device('cpu')

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)

        # State
        self.pop = PopulationState(self.device)
        self.sim_time: int = 0

        # History (recorded per call to step / step_one_day)
        self.infection_history: List[float] = []
        self.population_history: List[int] = []

        # Pre-compute cell-list grid (if using that backend)
        self._cell_size: float = max(cfg.mating_distance,
                                     cfg.mating_distance * cfg.exploration_rate_boost)
        self._n_cells: int = max(1, int(cfg.grid_size / self._cell_size))
        self._cell_size = cfg.grid_size / self._n_cells  # exact fit

        # Initialize
        self._initialize_population()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize_population(self):
        N = self.cfg.initial_population
        n_infected = int(round(N * self.cfg.infected_fraction))

        # --- Positions: central third of the grid ---
        lo = self.cfg.grid_size / 3.0
        hi = 2.0 * self.cfg.grid_size / 3.0
        x = torch.empty(N, device=self.device).uniform_(lo, hi)
        y = torch.empty(N, device=self.device).uniform_(lo, hi)

        # --- Infection status ---
        infected = torch.zeros(N, device=self.device, dtype=torch.bool)
        infected[:n_infected] = True
        # Shuffle so infected aren't all in the first slots
        perm = torch.randperm(N, device=self.device)
        infected = infected[perm]

        # --- Sex assignment ---
        # Among infected: 50/50 male/female
        # Among uninfected: follow male_to_female_ratio
        is_male = torch.zeros(N, device=self.device, dtype=torch.bool)

        inf_mask = infected
        uninf_mask = ~infected
        n_inf = int(inf_mask.sum().item())
        n_uninf = int(uninf_mask.sum().item())

        # Infected: half male
        inf_indices = inf_mask.nonzero(as_tuple=False).squeeze(-1)
        n_inf_male = n_inf // 2
        inf_male_sel = inf_indices[torch.randperm(n_inf, device=self.device)[:n_inf_male]]
        is_male[inf_male_sel] = True

        # Uninfected: male_to_female_ratio fraction male
        uninf_indices = uninf_mask.nonzero(as_tuple=False).squeeze(-1)
        n_uninf_male = int(round(n_uninf * self.cfg.male_to_female_ratio))
        uninf_male_sel = uninf_indices[torch.randperm(n_uninf, device=self.device)[:n_uninf_male]]
        is_male[uninf_male_sel] = True

        # --- Ages & life expectancy ---
        age = torch.randint(self.cfg.initial_age_min, self.cfg.initial_age_max + 1,
                            (N,), device=self.device, dtype=torch.int32)
        max_life = torch.randint(self.cfg.life_expectancy_min, self.cfg.life_expectancy_max + 1,
                                 (N,), device=self.device, dtype=torch.int32)

        # --- Mating cooldown: allow immediate mating at t=0 ---
        last_mate = torch.full((N,), -self.cfg.mating_cooldown_female,
                               device=self.device, dtype=torch.int32)

        # --- All start as adults (not eggs) ---
        is_egg = torch.zeros(N, device=self.device, dtype=torch.bool)

        # Write into state
        self.pop.x = x
        self.pop.y = y
        self.pop.infected = infected
        self.pop.is_male = is_male
        self.pop.age = age
        self.pop.max_life = max_life
        self.pop.last_mate = last_mate
        self.pop.is_egg = is_egg

    # ------------------------------------------------------------------
    # Single simulation step  (1 hour)
    # ------------------------------------------------------------------
    def step(self):
        """Advance the simulation by one hour."""
        self._move()
        self._age()
        self._hatch_eggs()
        self._retire_dead()
        self._mate()
        self._enforce_capacity()
        self._record_stats()
        self.sim_time += 1

    def step_one_day(self):
        """Run 24 hourly steps, recording stats only at the end of the day."""
        for _ in range(24):
            self._move()
            self._age()
            self._hatch_eggs()
            self._retire_dead()
            self._mate()
            self._enforce_capacity()
            self.sim_time += 1
        self._record_stats()

    # ------------------------------------------------------------------
    # Movement  (vectorized Lévy flight)
    # ------------------------------------------------------------------
    def _move(self):
        """
        Lévy flight for all adults (eggs don't move).

        Step size ~ Pareto(α) + 1  implemented via inverse-CDF:
            U ~ Uniform(0,1)  →  step = U^(-1/α)
        Direction ~ Uniform(0, 2π).
        Positions wrap toroidally.
        """
        adults = ~self.pop.is_egg
        n_adults = int(adults.sum().item())
        if n_adults == 0:
            return

        U = torch.rand(n_adults, device=self.device).clamp(min=1e-7)
        step_sizes = U.pow(-1.0 / self.cfg.levy_alpha)
        angles = 2.0 * np.pi * torch.rand(n_adults, device=self.device)

        dx = step_sizes * torch.cos(angles)
        dy = step_sizes * torch.sin(angles)

        self.pop.x[adults] = (self.pop.x[adults] + dx) % self.cfg.grid_size
        self.pop.y[adults] = (self.pop.y[adults] + dy) % self.cfg.grid_size

    # ------------------------------------------------------------------
    # Aging
    # ------------------------------------------------------------------
    def _age(self):
        """Increment age of every individual (adults and eggs) by 1 hour."""
        self.pop.age += 1

    # ------------------------------------------------------------------
    # Egg hatching
    # ------------------------------------------------------------------
    def _hatch_eggs(self):
        """Eggs whose age exceeds the hatching threshold become adults."""
        ready = self.pop.is_egg & (self.pop.age > self.cfg.egg_hatching_age)
        self.pop.is_egg[ready] = False  # promote to adult

    # ------------------------------------------------------------------
    # Death / retirement  (with density-dependent mortality options)
    # ------------------------------------------------------------------
    def _retire_dead(self):
        """
        Remove dead individuals from the population.
        Always applies natural death (age > max_life).
        Then applies the selected density-dependent mortality mode.
        """
        # 1. Natural death: remove adults that exceeded their life expectancy
        alive = self.pop.is_egg | (self.pop.age <= self.pop.max_life)
        self.pop.keep(alive)

        mode = self.cfg.mortality_mode
        if mode == 'none':
            return

        adult_mask = ~self.pop.is_egg
        n_adults = int(adult_mask.sum().item())
        K = self.cfg.max_population

        if n_adults == 0 or K <= 0:
            return

        density_ratio = n_adults / K  # N/K

        if mode == 'logistic':
            # ── Logistic density-dependent adult mortality ──
            # Per-capita hourly death probability increases with (N/K)^β.
            # At N=K, extra mortality ≈ 1/mean_lifespan per hour (doubles natural rate).
            # At N<<K, extra mortality ≈ 0.
            if density_ratio <= 0.5:
                return  # negligible at low density
            base_hourly_mu = 1.0 / ((self.cfg.life_expectancy_min + self.cfg.life_expectancy_max) / 2)
            extra_mu = base_hourly_mu * (density_ratio ** self.cfg.mortality_beta)
            extra_mu = min(extra_mu, 0.1)  # cap at 10% per hour to avoid instabilities
            # Each adult dies with probability extra_mu this hour
            adult_idx = adult_mask.nonzero(as_tuple=False).squeeze(-1)
            death_roll = torch.rand(n_adults, device=self.device)
            survivors = death_roll >= extra_mu
            kill_mask = torch.ones(self.pop.n, device=self.device, dtype=torch.bool)
            kill_mask[adult_idx[~survivors]] = False
            self.pop.keep(kill_mask)

        elif mode == 'cannibalism':
            # ── Egg cannibalism (Tribolium-style) ──
            # Adults consume eggs at a rate that scales with adult density.
            # This is the primary population regulation mechanism in flour beetles
            # (Daly & Ryan 1983, Park 1934, Sonleitner & Gutherie 1991).
            # Probability each egg is eaten = cannibalism_rate × N_adults × (N/K)^β
            egg_mask = self.pop.is_egg
            n_eggs = int(egg_mask.sum().item())
            if n_eggs == 0:
                return
            p_eaten = self.cfg.cannibalism_rate * n_adults * (density_ratio ** self.cfg.mortality_beta)
            p_eaten = min(p_eaten, 0.95)  # cap so some eggs always survive
            egg_idx = egg_mask.nonzero(as_tuple=False).squeeze(-1)
            death_roll = torch.rand(n_eggs, device=self.device)
            kill_mask = torch.ones(self.pop.n, device=self.device, dtype=torch.bool)
            kill_mask[egg_idx[death_roll < p_eaten]] = False
            self.pop.keep(kill_mask)

        elif mode == 'contest':
            # ── Contest competition ──
            # When N > K, each excess individual has a probability of dying
            # each hour.  Below K, no extra mortality.
            if n_adults <= K:
                return
            excess = n_adults - K
            # Kill probability per adult = (excess / N) per hour
            p_die = excess / n_adults
            adult_idx = adult_mask.nonzero(as_tuple=False).squeeze(-1)
            death_roll = torch.rand(n_adults, device=self.device)
            kill_mask = torch.ones(self.pop.n, device=self.device, dtype=torch.bool)
            kill_mask[adult_idx[death_roll < p_die]] = False
            self.pop.keep(kill_mask)

    # ------------------------------------------------------------------
    # Carrying-capacity enforcement
    # ------------------------------------------------------------------
    def _enforce_capacity(self):
        """
        Enforce population limits:
        - Adults: random cull if above max_population (hard carrying capacity).
        - Eggs: age-priority cull — remove the YOUNGEST eggs first so that
          eggs close to hatching survive.  Without this, random culling
          kills eggs long before they reach the 552-hour hatching age.
        """
        adult_mask = ~self.pop.is_egg
        egg_mask = self.pop.is_egg
        n_adults = int(adult_mask.sum().item())
        n_eggs = int(egg_mask.sum().item())

        excess_adults = n_adults - self.cfg.max_population
        excess_eggs = n_eggs - self.cfg.max_eggs

        if excess_adults > 0 or excess_eggs > 0:
            keep = torch.ones(self.pop.n, device=self.device, dtype=torch.bool)
            if excess_adults > 0:
                adult_idx = adult_mask.nonzero(as_tuple=False).squeeze(-1)
                remove_idx = adult_idx[torch.randperm(n_adults, device=self.device)[:excess_adults]]
                keep[remove_idx] = False
            if excess_eggs > 0:
                # Age-priority: remove the youngest eggs (smallest age → furthest from hatching)
                egg_idx = egg_mask.nonzero(as_tuple=False).squeeze(-1)
                egg_ages = self.pop.age[egg_idx]
                # Sort eggs by age ascending — first entries are youngest
                _, age_order = torch.sort(egg_ages)
                youngest_idx = egg_idx[age_order[:excess_eggs]]
                keep[youngest_idx] = False
            self.pop.keep(keep)

    # ------------------------------------------------------------------
    # Mating  (dispatcher)
    # ------------------------------------------------------------------
    def _mate(self):
        if self.cfg.mating_backend == 'brute':
            self._mate_brute()
        else:
            self._mate_cell_list()

    # ------------------------------------------------------------------
    # Mating backend 1:  brute-force distance matrix
    # ------------------------------------------------------------------
    def _mate_brute(self):
        """
        Compute the full female × male toroidal distance matrix on GPU.
        For N = 20 000  (10 K × 10 K) this is ~400 MB float32 — fits
        comfortably in the L40S's 48 GB.

        Steps:
        1.  Identify eligible females and males (adult, off cooldown).
        2.  Compute pairwise toroidal distances.
        3.  Build a boolean "within range" mask.
        4.  For each female, randomly pick one (or two) eligible males.
        5.  Generate offspring and add as eggs.
        """
        t = self.sim_time
        pop = self.pop

        # --- Eligibility masks ---
        adult = ~pop.is_egg
        female_mask = adult & ~pop.is_male
        male_mask   = adult & pop.is_male

        # Cooldown check
        cd_female = self.cfg.mating_cooldown_female
        cd_male   = self.cfg.mating_cooldown_male
        female_eligible = female_mask & ((t - pop.last_mate) >= cd_female)
        male_eligible   = male_mask   & ((t - pop.last_mate) >= cd_male)

        fem_idx = female_eligible.nonzero(as_tuple=False).squeeze(-1)
        mal_idx = male_eligible.nonzero(as_tuple=False).squeeze(-1)
        nf = fem_idx.shape[0]
        nm = mal_idx.shape[0]
        if nf == 0 or nm == 0:
            return

        # --- Toroidal distances  [nf, nm] ---
        fx = pop.x[fem_idx].unsqueeze(1)  # [nf, 1]
        fy = pop.y[fem_idx].unsqueeze(1)
        mx = pop.x[mal_idx].unsqueeze(0)  # [1, nm]
        my = pop.y[mal_idx].unsqueeze(0)

        gs = self.cfg.grid_size
        dx = torch.abs(fx - mx)
        dx = torch.min(dx, gs - dx)
        dy = torch.abs(fy - my)
        dy = torch.min(dy, gs - dy)
        dist = torch.sqrt(dx * dx + dy * dy)   # [nf, nm]

        # --- Per-female mating distance (infected females may have expanded range) ---
        md = self.cfg.mating_distance
        if self.cfg.wolbachia_effects.get('increased_exploration_rate', False):
            fem_infected = pop.infected[fem_idx]  # [nf]
            per_fem_dist = torch.where(fem_infected,
                                       torch.tensor(md * self.cfg.exploration_rate_boost,
                                                    device=self.device),
                                       torch.tensor(md, device=self.device))  # [nf]
            in_range = dist <= per_fem_dist.unsqueeze(1)  # [nf, nm]
        else:
            in_range = dist <= md

        # --- Pair assignment: for each female, pick 1 (or 2) random males ---
        self._assign_mates_and_reproduce(fem_idx, mal_idx, in_range)

    # ------------------------------------------------------------------
    # Mating backend 2:  cell-list spatial hashing  (for N > 50 000)
    # ------------------------------------------------------------------
    def _mate_cell_list(self):
        """
        Partition beetles into grid cells of side ≥ mating_distance.
        Each female only checks males in her own cell and the 8 neighbours
        (with toroidal wrapping).  This reduces the distance evaluations
        from O(F·M) to O(F · k) where k is the mean per-cell male count
        × 9.  At uniform density with 20 000 beetles and cell_size = 5
        on a 500×500 grid, each cell has ~2 beetles on average, so
        k ≈ 18.  This scales linearly with N.

        Implementation: we process one cell-neighbourhood at a time in a
        vectorized batch.  No Python loop over individual beetles.
        """
        t = self.sim_time
        pop = self.pop
        gs = self.cfg.grid_size
        cs = self._cell_size
        nc = self._n_cells  # number of cells per dimension

        # --- Eligibility ---
        adult = ~pop.is_egg
        cd_female = self.cfg.mating_cooldown_female
        cd_male   = self.cfg.mating_cooldown_male
        female_eligible = adult & ~pop.is_male & ((t - pop.last_mate) >= cd_female)
        male_eligible   = adult &  pop.is_male & ((t - pop.last_mate) >= cd_male)

        fem_idx = female_eligible.nonzero(as_tuple=False).squeeze(-1)
        mal_idx = male_eligible.nonzero(as_tuple=False).squeeze(-1)
        nf = fem_idx.shape[0]
        nm = mal_idx.shape[0]
        if nf == 0 or nm == 0:
            return

        # --- Assign cell indices ---
        fem_cx = (pop.x[fem_idx] / cs).long() % nc
        fem_cy = (pop.y[fem_idx] / cs).long() % nc
        fem_cell = fem_cx * nc + fem_cy  # flat cell index

        mal_cx = (pop.x[mal_idx] / cs).long() % nc
        mal_cy = (pop.y[mal_idx] / cs).long() % nc
        mal_cell = mal_cx * nc + mal_cy

        # --- Sort males by cell for fast lookup ---
        sort_order = torch.argsort(mal_cell)
        mal_idx_sorted = mal_idx[sort_order]
        mal_cell_sorted = mal_cell[sort_order]

        # Build cell → male index ranges via searchsorted
        all_cells = torch.arange(nc * nc, device=self.device, dtype=torch.long)
        cell_starts = torch.searchsorted(mal_cell_sorted, all_cells)
        cell_ends = torch.searchsorted(mal_cell_sorted, all_cells + 1)

        # --- For each female, gather candidate males from 3×3 neighbourhood ---
        # Offsets for the 9 neighbours (including self)
        offsets_x = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1],
                                 device=self.device, dtype=torch.long)
        offsets_y = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1],
                                 device=self.device, dtype=torch.long)

        # For efficiency, we process females in batches grouped by cell.
        # But a simpler approach that works well for moderate N:
        # iterate over the 9 neighbour offsets, gather all candidate males
        # for ALL females at once, compute distances, then merge.

        # We'll build a sparse list of (female_local_idx, male_global_idx) pairs
        # that are within mating distance.
        md = self.cfg.mating_distance
        boost = self.cfg.exploration_rate_boost if self.cfg.wolbachia_effects.get(
            'increased_exploration_rate', False) else 1.0

        pair_fem = []  # local indices into fem_idx
        pair_mal = []  # global indices into pop arrays (via mal_idx_sorted)

        for ox, oy in zip(offsets_x.tolist(), offsets_y.tolist()):
            # Neighbour cell for each female
            nb_cx = (fem_cx + ox) % nc
            nb_cy = (fem_cy + oy) % nc
            nb_cell = nb_cx * nc + nb_cy  # [nf]

            # Gather start/end for each female's neighbour cell
            starts = cell_starts[nb_cell]  # [nf]
            ends   = cell_ends[nb_cell]    # [nf]
            lengths = ends - starts        # [nf] – males in this neighbour cell

            max_len = int(lengths.max().item()) if nf > 0 else 0
            if max_len == 0:
                continue

            # Build a [nf, max_len] index matrix of candidate males
            arange = torch.arange(max_len, device=self.device).unsqueeze(0)  # [1, max_len]
            offsets_mat = starts.unsqueeze(1) + arange                       # [nf, max_len]
            valid = arange < lengths.unsqueeze(1)                            # [nf, max_len]

            # Clamp to avoid out-of-bounds (invalid slots will be masked out)
            offsets_mat = offsets_mat.clamp(max=nm - 1)
            cand_global = mal_idx_sorted[offsets_mat]  # [nf, max_len] – global pop indices

            # Compute toroidal distances
            f_x = pop.x[fem_idx].unsqueeze(1).expand_as(offsets_mat.float())
            f_y = pop.y[fem_idx].unsqueeze(1).expand_as(offsets_mat.float())
            c_x = pop.x[cand_global]
            c_y = pop.y[cand_global]

            ddx = torch.abs(f_x - c_x)
            ddx = torch.min(ddx, gs - ddx)
            ddy = torch.abs(f_y - c_y)
            ddy = torch.min(ddy, gs - ddy)
            d = torch.sqrt(ddx * ddx + ddy * ddy)

            # Per-female mating distance
            if boost > 1.0:
                fem_inf = pop.infected[fem_idx].unsqueeze(1).expand_as(d)
                threshold = torch.where(fem_inf,
                                        torch.tensor(md * boost, device=self.device),
                                        torch.tensor(md, device=self.device))
            else:
                threshold = md

            close = (d <= threshold) & valid

            # Extract pairs
            fi_local, mj_local = close.nonzero(as_tuple=True)
            if fi_local.shape[0] > 0:
                pair_fem.append(fi_local)
                pair_mal.append(cand_global[fi_local, mj_local])

        if len(pair_fem) == 0:
            return

        all_pair_fem = torch.cat(pair_fem)       # local indices into fem_idx
        all_pair_mal_global = torch.cat(pair_mal) # global pop indices of males

        # Deduplicate: same (female, male) pair may appear from overlapping cells
        pair_key = all_pair_fem.long() * pop.n + all_pair_mal_global.long()
        unique_keys, unique_inv = torch.unique(pair_key, return_inverse=True)
        # Keep first occurrence
        first_occ = torch.zeros(unique_keys.shape[0], device=self.device, dtype=torch.long)
        first_occ.scatter_(0, unique_inv, torch.arange(all_pair_fem.shape[0],
                           device=self.device, dtype=torch.long))
        all_pair_fem = all_pair_fem[first_occ]
        all_pair_mal_global = all_pair_mal_global[first_occ]

        # Build the in_range boolean matrix as a sparse representation,
        # then delegate to the shared assignment routine.
        # For the assignment function, we need a [nf, nm] mask.
        # But that defeats the purpose of the cell list for very large N.
        # Instead, we call a sparse-pair version of the assignment.
        self._assign_mates_sparse(fem_idx, mal_idx, all_pair_fem, all_pair_mal_global)

    # ------------------------------------------------------------------
    # Mate assignment + offspring generation (dense matrix version)
    # ------------------------------------------------------------------
    def _assign_mates_and_reproduce(self, fem_idx, mal_idx, in_range):
        """
        Given the boolean [nf, nm] in_range matrix, assign mates randomly
        and generate offspring.

        Strategy:
          - For each female, pick one random in-range male.
          - If multiple_mating, allow a second mate for females that had a first.
          - A male can only be claimed once per step (enforced sequentially on
            unique males, but parallelised over the candidate selection).

        For 10K females this takes a few ms on GPU.
        """
        nf = fem_idx.shape[0]
        nm = mal_idx.shape[0]
        pop = self.pop
        t = self.sim_time
        max_mates = 2 if self.cfg.multiple_mating else 1

        # Random scores for each candidate pair (used to pick a random in-range male)
        rand_scores = torch.rand(nf, nm, device=self.device)
        rand_scores[~in_range] = -1.0  # disqualify out-of-range

        all_offspring = []

        male_claimed = torch.zeros(nm, device=self.device, dtype=torch.bool)

        for _ in range(max_mates):
            # Mask out already-claimed males
            current_scores = rand_scores.clone()
            current_scores[:, male_claimed] = -1.0

            # For each female, find the male with the highest random score
            best_scores, best_local_mal = current_scores.max(dim=1)  # [nf]
            has_mate = best_scores > 0  # females that found at least one eligible male

            if not has_mate.any():
                break

            # Indices of matched females (local) and males (local into mal_idx)
            matched_fem_local = has_mate.nonzero(as_tuple=False).squeeze(-1)
            matched_mal_local = best_local_mal[matched_fem_local]

            # Mark these males as claimed
            male_claimed[matched_mal_local] = True

            # Global indices
            gf = fem_idx[matched_fem_local]
            gm = mal_idx[matched_mal_local]

            # Update mating times
            pop.last_mate[gf] = t
            pop.last_mate[gm] = t

            # Disable these females from getting another mate in this pass
            rand_scores[matched_fem_local, :] = -1.0

            # Generate offspring
            offspring = self._generate_offspring_batch(gf, gm)
            if offspring is not None:
                all_offspring.append(offspring)

        # Add all offspring to population
        self._add_offspring(all_offspring)

    # ------------------------------------------------------------------
    # Mate assignment + offspring generation (sparse pairs version)
    # ------------------------------------------------------------------
    def _assign_mates_sparse(self, fem_idx, mal_idx, pair_fem_local, pair_mal_global):
        """
        Sparse-pair version for the cell-list backend.
        ``pair_fem_local``: indices into fem_idx for each candidate pair.
        ``pair_mal_global``: global pop indices for the male in each pair.
        """
        pop = self.pop
        t = self.sim_time
        max_mates = 2 if self.cfg.multiple_mating else 1

        nf = fem_idx.shape[0]

        # Random scores for tie-breaking
        rand_scores = torch.rand(pair_fem_local.shape[0], device=self.device)

        all_offspring = []
        female_mated_count = torch.zeros(nf, device=self.device, dtype=torch.int32)
        male_claimed = set()  # track globally which males are claimed

        # Sort pairs by female, then iterate in a vectorized-ish way
        # For each mating round, we pick one male per female.
        for _ in range(max_mates):
            if pair_fem_local.shape[0] == 0:
                break

            # For each female, pick the pair with the highest random score
            # Use scatter_max-like logic
            best_score = torch.full((nf,), -1.0, device=self.device)
            best_pair_idx = torch.full((nf,), -1, device=self.device, dtype=torch.long)

            # Simple approach: sort by (female, -score) and take first per female
            sort_key = pair_fem_local.float() - rand_scores / (rand_scores.max() + 1)
            order = torch.argsort(sort_key)
            sorted_fem = pair_fem_local[order]
            sorted_pair = order

            # Find first occurrence of each female
            change = torch.ones(sorted_fem.shape[0], device=self.device, dtype=torch.bool)
            change[1:] = sorted_fem[1:] != sorted_fem[:-1]

            first_indices = change.nonzero(as_tuple=False).squeeze(-1)
            selected_fem_local = sorted_fem[first_indices]
            selected_pair_idx = sorted_pair[first_indices]
            selected_mal_global = pair_mal_global[selected_pair_idx]

            # Filter out females that already reached max mates
            still_eligible = female_mated_count[selected_fem_local] < max_mates
            selected_fem_local = selected_fem_local[still_eligible]
            selected_mal_global = selected_mal_global[still_eligible]

            if selected_fem_local.shape[0] == 0:
                break

            # Deduplicate males (each male only once per round)
            _, unique_male_inv = torch.unique(selected_mal_global, return_inverse=True)
            unique_male_first = torch.zeros(selected_mal_global.max().item() + 1,
                                            device=self.device, dtype=torch.bool)
            keep_pair = torch.zeros(selected_fem_local.shape[0], device=self.device, dtype=torch.bool)
            for i in range(selected_fem_local.shape[0]):
                mg = selected_mal_global[i].item()
                if mg not in male_claimed:
                    keep_pair[i] = True
                    male_claimed.add(mg)

            selected_fem_local = selected_fem_local[keep_pair]
            selected_mal_global = selected_mal_global[keep_pair]

            if selected_fem_local.shape[0] == 0:
                break

            gf = fem_idx[selected_fem_local]
            gm = selected_mal_global

            pop.last_mate[gf] = t
            pop.last_mate[gm] = t
            female_mated_count[selected_fem_local] += 1

            offspring = self._generate_offspring_batch(gf, gm)
            if offspring is not None:
                all_offspring.append(offspring)

            # Remove used pairs
            used_fem_set = set(selected_fem_local.tolist())
            used_mal_set = male_claimed
            mask = torch.tensor([(pf.item() not in used_fem_set) and
                                 (pm.item() not in used_mal_set)
                                 for pf, pm in zip(pair_fem_local, pair_mal_global)],
                                device=self.device, dtype=torch.bool)
            pair_fem_local = pair_fem_local[mask]
            pair_mal_global = pair_mal_global[mask]
            rand_scores = rand_scores[mask]

        self._add_offspring(all_offspring)

    # ------------------------------------------------------------------
    # Vectorized offspring generation
    # ------------------------------------------------------------------
    def _generate_offspring_batch(self, mother_global_idx, father_global_idx):
        """
        Generate offspring for all mating pairs in a single vectorized call.

        Parameters
        ----------
        mother_global_idx : Tensor [P]  – global population indices of mothers
        father_global_idx : Tensor [P]  – global population indices of fathers

        Returns
        -------
        dict with tensors for each offspring attribute, or None if no offspring.
        """
        P = mother_global_idx.shape[0]
        if P == 0:
            return None

        pop = self.pop
        cfg = self.cfg
        effects = cfg.wolbachia_effects

        # --- Eggs per pair  (Uniform(1, egg_laying_max)) ---
        eggs = torch.randint(1, cfg.egg_laying_max + 1, (P,),
                             device=self.device, dtype=torch.int32)

        # --- Logistic birth suppression: reduce clutch size as N → K ---
        # This prevents runaway egg production at carrying capacity.
        # L(N) = max(0, 1 - N_adults/K).  Each egg count is multiplied by L.
        n_adults = int((~pop.is_egg).sum().item())
        logistic_factor = max(0.0, 1.0 - n_adults / cfg.max_population)
        if logistic_factor < 1.0:
            eggs = torch.round(eggs.float() * logistic_factor).to(torch.int32)
            eggs = eggs.clamp(min=0)

        # --- Fecundity modifiers (only for infected mothers) ---
        mom_infected = pop.infected[mother_global_idx]  # [P]
        inc_eggs = effects.get('increased_eggs', False)
        red_eggs = effects.get('reduced_eggs', False)

        if inc_eggs and not red_eggs:
            eggs[mom_infected] = torch.round(
                eggs[mom_infected].float() * cfg.fecundity_increase_factor
            ).to(torch.int32)
        elif red_eggs and not inc_eggs:
            eggs[mom_infected] = torch.round(
                eggs[mom_infected].float() * cfg.fecundity_decrease_factor
            ).to(torch.int32)
        # If both or neither, no change.

        # --- Cytoplasmic incompatibility ---
        if effects.get('cytoplasmic_incompatibility', False):
            dad_infected = pop.infected[father_global_idx]
            ci_mask = dad_infected & ~mom_infected  # infected ♂ × uninfected ♀
            if ci_mask.any():
                if cfg.ci_strength >= 1.0:
                    eggs[ci_mask] = 0
                else:
                    # Each egg survives independently with prob (1 - ci_strength)
                    ci_idx = ci_mask.nonzero(as_tuple=False).squeeze(-1)
                    max_e = int(eggs[ci_idx].max().item())
                    if max_e > 0:
                        rand_mat = torch.rand(ci_idx.shape[0], max_e, device=self.device)
                        lengths = eggs[ci_idx].unsqueeze(1)
                        valid = torch.arange(max_e, device=self.device).unsqueeze(0) < lengths
                        survived = ((rand_mat >= cfg.ci_strength) & valid).sum(dim=1).to(torch.int32)
                        eggs[ci_idx] = survived

        # --- Expand: repeat mother attributes per egg ---
        total_eggs = int(eggs.sum().item())
        if total_eggs == 0:
            return None

        # Mother index for each offspring
        mom_for_egg = mother_global_idx.repeat_interleave(eggs.long())  # [total_eggs]

        # --- Position near mother (offset ∈ {-1, 0, 1}) ---
        ox = torch.randint(-1, 2, (total_eggs,), device=self.device, dtype=torch.float32)
        oy = torch.randint(-1, 2, (total_eggs,), device=self.device, dtype=torch.float32)
        new_x = (pop.x[mom_for_egg] + ox) % cfg.grid_size
        new_y = (pop.y[mom_for_egg] + oy) % cfg.grid_size

        # --- Infection: inherited from mother ---
        new_infected = pop.infected[mom_for_egg]

        # --- Sex determination ---
        if effects.get('male_killing', False):
            # Infected mothers → mostly female offspring
            prob_male = torch.where(
                new_infected,
                torch.tensor(cfg.male_offspring_rate, device=self.device),
                torch.tensor(0.5, device=self.device)
            )
        else:
            prob_male = torch.full((total_eggs,), 0.5, device=self.device)
        new_is_male = torch.rand(total_eggs, device=self.device) < prob_male

        # --- Age & life expectancy ---
        new_age = torch.zeros(total_eggs, device=self.device, dtype=torch.int32)
        new_max_life = torch.randint(cfg.life_expectancy_min, cfg.life_expectancy_max + 1,
                                     (total_eggs,), device=self.device, dtype=torch.int32)
        new_last_mate = torch.full((total_eggs,), -cfg.mating_cooldown_female,
                                   device=self.device, dtype=torch.int32)
        new_is_egg = torch.ones(total_eggs, device=self.device, dtype=torch.bool)

        return {
            'x': new_x, 'y': new_y, 'infected': new_infected,
            'is_male': new_is_male, 'age': new_age, 'max_life': new_max_life,
            'last_mate': new_last_mate, 'is_egg': new_is_egg,
        }

    def _add_offspring(self, offspring_list):
        """Concatenate all offspring dicts into the population state."""
        if not offspring_list:
            return
        combined = {}
        for key in offspring_list[0]:
            combined[key] = torch.cat([o[key] for o in offspring_list], dim=0)
        self.pop.append(**combined)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _record_stats(self):
        adult = ~self.pop.is_egg
        n_adults = int(adult.sum().item())
        if n_adults == 0:
            self.infection_history.append(0.0)
        else:
            n_inf = int((self.pop.infected & adult).sum().item())
            self.infection_history.append(n_inf / n_adults)
        self.population_history.append(n_adults)

    def get_infection_rate(self) -> float:
        return self.infection_history[-1] if self.infection_history else 0.0

    def get_population_size(self) -> int:
        return self.population_history[-1] if self.population_history else 0

    def get_sex_ratio(self) -> dict:
        """Returns counts of adult males and females (infected and uninfected)."""
        adult = ~self.pop.is_egg
        return {
            'F_U': int((adult & ~self.pop.is_male & ~self.pop.infected).sum().item()),
            'F_I': int((adult & ~self.pop.is_male &  self.pop.infected).sum().item()),
            'M_U': int((adult &  self.pop.is_male & ~self.pop.infected).sum().item()),
            'M_I': int((adult &  self.pop.is_male &  self.pop.infected).sum().item()),
        }

    # ------------------------------------------------------------------
    # CSV export (compatible with existing analysis pipeline)
    # ------------------------------------------------------------------
    def export_history_csv(self, path: str):
        """Write the recorded time-series to a CSV matching the existing format."""
        import csv
        n = min(len(self.infection_history), len(self.population_history))
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Population Size', 'Infection Rate'])
            for i in range(n):
                writer.writerow([self.population_history[i],
                                 f"{self.infection_history[i]:.6f}"])


# ---------------------------------------------------------------------------
# Convenience: run a full experiment
# ---------------------------------------------------------------------------
def run_experiment(cfg: SimConfig, n_days: int = 365, verbose: bool = True) -> GPUSimulation:
    """
    Run a complete simulation for ``n_days`` days and return the simulation object.
    """
    sim = GPUSimulation(cfg)
    start = time.time()
    for day in range(1, n_days + 1):
        sim.step_one_day()
        if verbose and day % 30 == 0:
            elapsed = time.time() - start
            print(f"  Day {day:4d} | Pop {sim.get_population_size():6d} | "
                  f"Inf {sim.get_infection_rate():.3f} | "
                  f"Eggs {sim.pop.n_eggs:6d} | "
                  f"Total {sim.pop.n:7d} | "
                  f"{elapsed:.1f}s elapsed")
    if verbose:
        print(f"Completed {n_days} days in {time.time()-start:.1f}s")
    return sim


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse, json

    parser = argparse.ArgumentParser(description="WINGS GPU Simulation")
    parser.add_argument('--population', type=int, default=50)
    parser.add_argument('--max-pop', type=int, default=20_000)
    parser.add_argument('--max-eggs', type=int, default=800_000,
                        help='Egg buffer cap (must be large for 23-day pipeline)')
    parser.add_argument('--grid-size', type=int, default=500)
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--ci', action='store_true', help='Enable cytoplasmic incompatibility')
    parser.add_argument('--mk', action='store_true', help='Enable male killing')
    parser.add_argument('--er', action='store_true', help='Enable increased exploration rate')
    parser.add_argument('--ie', action='store_true', help='Enable increased eggs')
    parser.add_argument('--re', action='store_true', help='Enable reduced eggs')
    parser.add_argument('--ci-strength', type=float, default=1.0)
    parser.add_argument('--mortality', choices=['none', 'logistic', 'cannibalism', 'contest'],
                        default='cannibalism',
                        help='Density-dependent mortality mode')
    parser.add_argument('--mortality-beta', type=float, default=2.0,
                        help='Exponent for density-dependent effects')
    parser.add_argument('--cannibalism-rate', type=float, default=0.0001,
                        help='Egg cannibalism rate per adult per hour at N=K')
    parser.add_argument('--backend', choices=['brute', 'cell_list'], default='cell_list')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    args = parser.parse_args()

    cfg = SimConfig(
        initial_population=args.population,
        max_population=args.max_pop,
        max_eggs=args.max_eggs,
        grid_size=args.grid_size,
        ci_strength=args.ci_strength,
        mortality_mode=args.mortality,
        mortality_beta=args.mortality_beta,
        cannibalism_rate=args.cannibalism_rate,
        mating_backend=args.backend,
        device=args.device,
        seed=args.seed,
        wolbachia_effects={
            'cytoplasmic_incompatibility': args.ci,
            'male_killing': args.mk,
            'increased_exploration_rate': args.er,
            'increased_eggs': args.ie,
            'reduced_eggs': args.re,
        },
    )

    print(f"WINGS GPU Simulation")
    print(f"  Device:     {args.device}")
    print(f"  Backend:    {args.backend}")
    print(f"  Population: {args.population}")
    print(f"  Max pop:    {args.max_pop}  |  Max eggs: {cfg.max_eggs}")
    print(f"  Mortality:  {args.mortality} (beta={args.mortality_beta})")
    print(f"  Grid:       {args.grid_size}×{args.grid_size}")
    print(f"  Days:       {args.days}")
    print(f"  Effects:    {json.dumps(cfg.wolbachia_effects)}")
    print()

    sim = run_experiment(cfg, n_days=args.days)

    if args.output:
        sim.export_history_csv(args.output)
        print(f"Results saved to {args.output}")
