"""
W.I.N.G.S. — Turelli-threshold analysis tests.

Pins the deterministic skeleton: the µ=0 special case has no interior
equilibrium (infection invades from rare → drift, not a threshold), µ>0
produces a CI-only unstable threshold at ≈µ/s_h matching the closed form,
and ER removes the threshold (invasion from rare).
"""

import numpy as np
import pytest

from wings.analysis import threshold as th


def test_closed_form_threshold():
    assert th.ci_threshold_closed_form(0.0) == 0.0
    p = th.ci_threshold_closed_form(0.03)
    assert p == pytest.approx((1 - np.sqrt(1 - 4 * 0.03)) / 2)
    assert th.ci_threshold_closed_form(0.01) == pytest.approx(0.01, abs=2e-3)


def test_no_interior_equilibrium_at_mu0():
    s_h, m_I, m_U, f = th.condition_params({"ci": True, "er": False, "ie": False})
    assert th.interior_equilibria(0.0, s_h, m_I, m_U, f) == []


def test_ci_has_unstable_and_stable_at_mu():
    s_h, m_I, m_U, f = th.condition_params({"ci": True, "er": False, "ie": False})
    eqs = th.interior_equilibria(0.03, s_h, m_I, m_U, f)
    kinds = {k for _, k in eqs}
    assert "unstable" in kinds and "stable" in kinds
    unstable = next(p for p, k in eqs if k == "unstable")
    assert unstable == pytest.approx(th.ci_threshold_closed_form(0.03), abs=1e-3)


@pytest.mark.parametrize("mu", [0.01, 0.02, 0.05, 0.08])
def test_numeric_matches_closed_form(mu):
    s_h, m_I, m_U, f = th.condition_params({"ci": True, "er": False, "ie": False})
    num = th.unstable_threshold(mu, s_h, m_I, m_U, f)
    assert num == pytest.approx(th.ci_threshold_closed_form(mu, s_h), abs=2e-3)


def test_relay_er_removes_threshold():
    _, m_I, m_U, f = th.condition_params({"ci": True, "er": False, "ie": False})
    assert th.invasion_ratio(0.03, m_I, m_U, f) < 1.0          # CI alone: no invasion from rare
    s2, mI2, mU2, f2 = th.condition_params({"ci": True, "er": True, "ie": False})
    assert th.invasion_ratio(0.03, mI2, mU2, f2) > 1.0          # CI+ER: invades from rare
    assert np.isnan(th.unstable_threshold(0.03, s2, mI2, mU2, f2))  # → no threshold


def test_expected_p_next_mu0_matches_manual():
    p = np.array([0.1, 0.5, 0.9])
    out = th.expected_p_next(p, 0.0, 1.0, 1.0, 1.0, 1.0)
    manual = p / (p + (1 - p) * (1 - p))
    np.testing.assert_allclose(out, manual)


def test_figure_and_results(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    mu_grid = np.round(np.arange(0.0, 0.051, 0.01), 4)
    th.plot_threshold_vs_mu(mu_grid, tmp_path)
    assert (tmp_path / "fig_threshold_vs_mu.png").exists()
    assert (tmp_path / "fig_threshold_vs_mu.csv").exists()
    out = th.build_results_md(mu_grid, tmp_path, None)
    assert out.exists() and "Turelli" in out.read_text(encoding="utf-8")


def test_stochastic_straddle_above_beats_below():
    recs = th.stochastic_straddle([0.05], offset=0.04, n_reps=60, max_generations=20)
    below = next(r for r in recs if r["side"] == "below")
    above = next(r for r in recs if r["side"] == "above")
    assert above["frac_fixed"] >= below["frac_fixed"]


# --- ABM crossover aggregator -----------------------------------------

def test_crossover_from_params_known_point():
    # CI Δp = p (A=1,α=1,γ=0); ER Δp = 0.05 (s0=0.05,β=0) → cross at p=0.05
    params = {"A": 1.0, "alpha": 1.0, "gamma": 0.0, "s_0": 0.05, "beta": 0.0}
    assert th.crossover_from_params(params) == pytest.approx(0.05, abs=2e-3)


def test_crossover_from_params_missing_keys():
    assert np.isnan(th.crossover_from_params({"A": 1.0}))


def test_parse_mu_from_name():
    assert th._parse_mu_from_name("data/dp_mu0.03.csv") == 0.03
    assert th._parse_mu_from_name("data/dp_mu0.0.csv") == 0.0
    assert th._parse_mu_from_name("nope.csv") is None


def test_plot_pstar_vs_threshold(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    rows = [
        {"mu": 0.01, "p_star": 0.012, "p_hat": th.ci_threshold_closed_form(0.01)},
        {"mu": 0.03, "p_star": 0.035, "p_hat": th.ci_threshold_closed_form(0.03)},
        {"mu": 0.05, "p_star": float("nan"), "p_hat": th.ci_threshold_closed_form(0.05)},
    ]
    th.plot_pstar_vs_threshold(rows, tmp_path)
    assert (tmp_path / "fig_pstar_vs_threshold.png").exists()
    assert (tmp_path / "fig_pstar_vs_threshold.csv").exists()
    md = tmp_path / "results.md"
    md.write_text("seed\n", encoding="utf-8")
    th.append_crossover_table(md, rows)
    assert "ABM crossover" in md.read_text(encoding="utf-8")
