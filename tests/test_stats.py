"""
W.I.N.G.S. — statistics helper tests.

Covers the bootstrap median CI and the effect-combination helpers in
analysis.stats.
"""

import numpy as np

from wings.analysis import stats


def test_bootstrap_ci_brackets_the_median():
    rng = np.random.default_rng(0)
    np.random.seed(0)  # bootstrap_ci uses the legacy global RNG
    data = rng.normal(10.0, 2.0, size=500)
    lo, hi = stats.bootstrap_ci(data, n_bootstrap=400)
    assert lo <= hi
    assert data.min() <= lo <= hi <= data.max()
    assert lo <= np.median(data) <= hi


def test_calculate_statistics_keys():
    import pandas as pd
    s = stats.calculate_statistics(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert {"mean", "median", "sem", "ci_median"} <= set(s)
    assert s["mean"] == 2.5
    assert s["median"] == 2.5


def test_get_wolbachia_effects_has_five_effects():
    effects = stats.get_wolbachia_effects()
    assert set(effects) == {
        "cytoplasmic_incompatibility", "male_killing",
        "increased_exploration_rate", "increased_eggs", "reduced_eggs",
    }


def test_combination_to_string():
    effects = stats.get_wolbachia_effects()
    assert stats.combination_to_string(effects, (True, False, False, False, False)) == "ci"
    assert stats.combination_to_string(effects, (False,) * 5) == "no_effects"
    assert stats.combination_to_string(
        effects, (True, False, True, False, False)) == "cier"
