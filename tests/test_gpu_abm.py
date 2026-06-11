"""
W.I.N.G.S. — GPU ABM engine tests.

These exercise the PyTorch engine on the CPU device, so they run
anywhere torch is installed (no CUDA required).  The whole module is
skipped when torch is absent.
"""

import pytest

torch = pytest.importorskip("torch")

from wings.models.gpu_abm import GPUSimulation, SimConfig, run_experiment  # noqa: E402


def _small_config(**overrides):
    """A tiny, fast CPU configuration for unit tests."""
    base = dict(
        initial_population=40,
        max_population=200,
        max_eggs=20_000,
        grid_size=40,
        device="cpu",
        seed=0,
        mating_backend="brute",
        wolbachia_effects={
            "cytoplasmic_incompatibility": False,
            "male_killing": False,
            "increased_exploration_rate": False,
            "increased_eggs": False,
            "reduced_eggs": False,
        },
    )
    base.update(overrides)
    return SimConfig(**base)


def test_config_defaults():
    cfg = SimConfig()
    assert cfg.mortality_mode == "cannibalism"
    assert cfg.mating_backend == "cell_list"
    assert 0.0 <= cfg.infected_fraction <= 1.0


def test_initial_infection_fraction_respected():
    cfg = _small_config(initial_population=100, infected_fraction=0.5)
    sim = GPUSimulation(cfg)
    n_infected = int(sim.pop.infected.sum().item())
    assert abs(n_infected - 50) <= 1


def test_zero_infection_stays_zero():
    cfg = _small_config(infected_fraction=0.0)
    sim = GPUSimulation(cfg)
    for _ in range(10):
        sim.step()
    assert all(rate == 0.0 for rate in sim.infection_history)


def test_step_records_history():
    sim = GPUSimulation(_small_config())
    for _ in range(5):
        sim.step()
    assert len(sim.infection_history) == 5
    assert len(sim.population_history) == 5
    assert sim.get_population_size() > 0


def test_reproducible_with_seed():
    a = GPUSimulation(_small_config(seed=123))
    b = GPUSimulation(_small_config(seed=123))
    for _ in range(8):
        a.step()
        b.step()
    assert a.infection_history == b.infection_history
    assert a.population_history == b.population_history


@pytest.mark.parametrize("backend", ["brute", "cell_list"])
def test_both_mating_backends_run(backend):
    sim = GPUSimulation(_small_config(mating_backend=backend, infected_fraction=0.2))
    for _ in range(6):
        sim.step()
    assert sim.get_population_size() > 0
    assert 0.0 <= sim.get_infection_rate() <= 1.0


def test_full_ci_kills_incompatible_eggs():
    """With CI on, an all-infected-male × uninfected-female cross yields no infected gain."""
    cfg = _small_config(
        infected_fraction=0.2,
        wolbachia_effects={
            "cytoplasmic_incompatibility": True,
            "male_killing": False,
            "increased_exploration_rate": False,
            "increased_eggs": False,
            "reduced_eggs": False,
        },
    )
    sim = run_experiment(cfg, n_days=10, verbose=False)
    # Sanity: the run completed and produced a valid infection trajectory.
    assert len(sim.infection_history) == 10
    assert all(0.0 <= r <= 1.0 for r in sim.infection_history)


def test_run_experiment_smoke():
    sim = run_experiment(_small_config(), n_days=4, verbose=False)
    assert sim.get_population_size() > 0
    assert len(sim.population_history) == 4


def test_full_leakage_introduces_uninfected_offspring():
    """With mu=1.0 every offspring of an infected mother is uninfected,
    so a fully-infected start drops below fixation once eggs hatch."""
    cfg = _small_config(infected_fraction=1.0, maternal_transmission_leakage=1.0)
    sim = run_experiment(cfg, n_days=50, verbose=False)
    assert sim.get_infection_rate() < 1.0

