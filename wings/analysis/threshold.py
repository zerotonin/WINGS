#!/usr/bin/env python3
"""
W.I.N.G.S. — Wright-Fisher Turelli-threshold analysis.

Deterministic skeleton of the discrete-generation recursion with
maternal-transmission leakage ``mu``:

    E[p'] = p m_I f (1-mu) / [ p m_I f + (1-p) m_U (1 - s_h p) ]

where m_I, m_U are the infected / uninfected female mating probabilities
(the ER effect), f is the IE fecundity multiplier, and s_h is the CI
strength.  Setting mu = 0 reproduces the perfect-transmission recursion.

This module provides:
  * the deterministic map and its interior equilibria (with stability),
  * the CI-only closed-form unstable threshold
        p_hat = [1 - sqrt(1 - 4 mu / s_h)] / 2  (≈ mu / s_h for small mu),
  * the low-frequency invasion ratio m_I f (1-mu) / m_U (the relay check),
  * a stochastic straddle experiment around p_hat (calls the WFM), and
  * the fig_threshold_vs_mu figure + a results.md table.

The deterministic analysis and the figure are pure NumPy/SciPy and run in
seconds on a CPU.  The ``--stochastic`` straddle runs the finite-N WFM and
is still cheap (seconds).

Usage:
    python -m wings.analysis.threshold --outdir results/threshold
    python -m wings.analysis.threshold --outdir results/threshold --stochastic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from wings.models.wfm import simulate

# ======================================================================
#  Style  (Paul Tol qualitative — consistent with plot_wings / plot_delta_p)
# ======================================================================

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
})

_TOL_BLUE = "#4477AA"
_TOL_TEAL = "#009988"
_TOL_GREY = "#BBBBBB"
_TOL_RED = "#EE6677"

# Phenotype conditions analysed (CI, ER, IE flags).  MK is omitted (absent
# in the wTcon strain).  ER and IE values are calibration yardsticks.
CONDITIONS: list[tuple[str, dict[str, bool]]] = [
    ("None", {"ci": False, "er": False, "ie": False}),
    ("CI", {"ci": True, "er": False, "ie": False}),
    ("ER", {"ci": False, "er": True, "ie": False}),
    ("IE", {"ci": False, "er": False, "ie": True}),
    ("CI+ER", {"ci": True, "er": True, "ie": False}),
    ("CI+ER+IE", {"ci": True, "er": True, "ie": True}),
]


# ======================================================================
#  Deterministic skeleton
# ======================================================================

def condition_params(flags: dict[str, bool], s_h: float = 1.0,
                     er_advantage: float = 1.4,
                     ie_factor: float = 1.2) -> tuple[float, float, float, float]:
    """Map a phenotype condition to ``(s_h, m_I, m_U, f)`` recursion terms."""
    s_h_eff = s_h if flags["ci"] else 0.0
    m_I = 1.0
    m_U = 1.0 / er_advantage if flags["er"] else 1.0
    f = ie_factor if flags["ie"] else 1.0
    return s_h_eff, m_I, m_U, f


def expected_p_next(p, mu: float, s_h: float, m_I: float, m_U: float,
                    f: float):
    """E[p'] from the deterministic skeleton (scalar or array ``p``)."""
    p = np.asarray(p, dtype=float)
    num = p * m_I * f * (1.0 - mu)
    den = p * m_I * f + (1.0 - p) * m_U * (1.0 - s_h * p)
    out = np.divide(num, den, out=np.zeros_like(den), where=den > 0)
    return out


def _delta(p: float, mu: float, s_h: float, m_I: float, m_U: float,
           f: float) -> float:
    return float(expected_p_next(p, mu, s_h, m_I, m_U, f)) - p


def interior_equilibria(mu: float, s_h: float, m_I: float, m_U: float,
                        f: float, n_grid: int = 4000
                        ) -> list[tuple[float, str]]:
    """Interior equilibria of Δp in (0, 1), each tagged stable/unstable."""
    ps = np.linspace(1e-4, 1.0 - 1e-4, n_grid)
    g = expected_p_next(ps, mu, s_h, m_I, m_U, f) - ps
    roots: list[tuple[float, str]] = []
    sign_change = np.where(np.diff(np.sign(g)) != 0)[0]
    for i in sign_change:
        a, b = ps[i], ps[i + 1]
        try:
            r = brentq(_delta, a, b, args=(mu, s_h, m_I, m_U, f))
        except ValueError:
            continue
        h = 1e-6
        slope = (_delta(r + h, mu, s_h, m_I, m_U, f)
                 - _delta(r - h, mu, s_h, m_I, m_U, f)) / (2 * h)
        roots.append((r, "stable" if slope < 0 else "unstable"))
    return roots


def ci_threshold_closed_form(mu: float, s_h: float = 1.0) -> float:
    """CI-only unstable threshold p_hat = [1 - sqrt(1 - 4 mu / s_h)] / 2."""
    disc = 1.0 - 4.0 * mu / s_h
    if disc < 0:
        return float("nan")  # no real threshold (cost exceeds CI capacity)
    return (1.0 - np.sqrt(disc)) / 2.0


def invasion_ratio(mu: float, m_I: float, m_U: float, f: float) -> float:
    """Low-frequency (p→0) per-generation growth ratio of infection."""
    return m_I * f * (1.0 - mu) / m_U


def unstable_threshold(mu: float, s_h: float, m_I: float, m_U: float,
                       f: float) -> float:
    """The lowest interior *unstable* equilibrium, or NaN if none."""
    for p, kind in interior_equilibria(mu, s_h, m_I, m_U, f):
        if kind == "unstable":
            return p
    return float("nan")


# ======================================================================
#  Stochastic straddle (finite-N WFM)
# ======================================================================

def stochastic_straddle(mu_values: list[float], offset: float = 0.02,
                        n_pop: int = 50, n_reps: int = 200,
                        max_generations: int = 15
                        ) -> list[dict[str, float]]:
    """Run the finite-N WFM just below / above p_hat for CI-only.

    Returns one record per (mu, side) with the fraction fixed/lost and the
    median final frequency with a 95% bootstrap-free percentile interval.
    """
    records: list[dict[str, float]] = []
    for mu in mu_values:
        p_hat = ci_threshold_closed_form(mu)
        for side, p0 in (("below", p_hat - offset), ("above", p_hat + offset)):
            p0 = float(np.clip(p0, 0.0, 1.0))
            finals = np.array([
                simulate(N=n_pop, max_generations=max_generations,
                         seed=1000 + rep, ci=True, mu=mu,
                         initial_infection_freq=p0)[-1][1]
                for rep in range(n_reps)
            ])
            records.append({
                "mu": mu, "p_hat": p_hat, "side": side, "p0": p0,
                "frac_fixed": float(np.mean(finals >= 0.99)),
                "frac_lost": float(np.mean(finals <= 0.01)),
                "median_final": float(np.median(finals)),
                "ci_lo": float(np.percentile(finals, 2.5)),
                "ci_hi": float(np.percentile(finals, 97.5)),
            })
    return records


# ======================================================================
#  Figure + report
# ======================================================================

def save_fig(fig, stem: Path) -> None:
    """Export PNG + SVG (editable text)."""
    fig.savefig(f"{stem}.png")
    fig.savefig(f"{stem}.svg")
    plt.close(fig)


def plot_threshold_vs_mu(mu_grid: np.ndarray, outdir: Path) -> Path:
    """fig_threshold_vs_mu: numeric CI-only threshold + closed-form line."""
    s_h, m_I, m_U, f = condition_params({"ci": True, "er": False, "ie": False})
    numeric = np.array([unstable_threshold(mu, s_h, m_I, m_U, f)
                        for mu in mu_grid])
    closed = np.array([ci_threshold_closed_form(mu, s_h) for mu in mu_grid])

    # CI+ER threshold (should be absent — ER lets it invade from rare)
    s_h2, mI2, mU2, f2 = condition_params({"ci": True, "er": True, "ie": False})
    ci_er = np.array([unstable_threshold(mu, s_h2, mI2, mU2, f2)
                      for mu in mu_grid])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(mu_grid, closed, color=_TOL_BLUE, lw=2.0, zorder=2,
            label=r"CI closed form $\hat p=[1-\sqrt{1-4\mu/s_h}]/2$")
    ax.scatter(mu_grid, numeric, s=28, color=_TOL_BLUE, zorder=3,
               edgecolors="white", linewidths=0.6, label="CI numeric root")
    ax.plot(mu_grid, mu_grid, color=_TOL_GREY, ls=":", lw=1.0, zorder=1,
            label=r"$\hat p=\mu$ (small-$\mu$ limit)")
    # CI+ER: no interior threshold → plot at 0 to make the relay visible
    ax.scatter(mu_grid, np.nan_to_num(ci_er, nan=0.0), s=18, marker="x",
               color=_TOL_TEAL, zorder=3, label="CI+ER (no threshold)")

    ax.set_xlabel(r"maternal-transmission leakage $\mu$")
    ax.set_ylabel(r"unstable infection threshold $\hat p$")
    ax.set_xlim(0, mu_grid.max())
    ax.set_ylim(bottom=-0.005)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("WFM Turelli threshold vs transmission leakage",
                 fontweight="bold", pad=10)
    fig.tight_layout()
    stem = outdir / "fig_threshold_vs_mu"
    save_fig(fig, stem)

    # CSV companion
    import csv
    with open(f"{stem}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mu", "p_hat_numeric_CI", "p_hat_closedform_CI",
                    "p_hat_CI_ER"])
        for i, mu in enumerate(mu_grid):
            w.writerow([f"{mu:.4f}", f"{numeric[i]:.6f}",
                        f"{closed[i]:.6f}", f"{ci_er[i]:.6f}"])
    return stem


def build_results_md(mu_grid: np.ndarray, outdir: Path,
                     straddle: list[dict[str, float]] | None) -> Path:
    """Write results.md: thresholds per condition, invasion conditions, summary."""
    lines: list[str] = []
    lines.append("# WFM Turelli-threshold analysis\n")
    lines.append("Deterministic skeleton: "
                 "`E[p'] = p·m_I·f·(1−µ) / [p·m_I·f + (1−p)·m_U·(1 − s_h·p)]`, "
                 "s_h = 1.0 (CI), ER → m_U = 1/1.4, IE → f = 1.2.\n")

    # Per-condition summary at a reference mu
    mu_ref = 0.03
    lines.append(f"## Equilibria & invasion at µ = {mu_ref}\n")
    lines.append("| Condition | invasion ratio (p→0) | invades from rare? "
                 "| unstable threshold p̂ | stable equilibrium |")
    lines.append("|---|---|---|---|---|")
    for name, flags in CONDITIONS:
        s_h, m_I, m_U, f = condition_params(flags)
        ratio = invasion_ratio(mu_ref, m_I, m_U, f)
        eqs = interior_equilibria(mu_ref, s_h, m_I, m_U, f)
        unstable = next((f"{p:.3f}" for p, k in eqs if k == "unstable"), "—")
        stable = next((f"{p:.3f}" for p, k in eqs if k == "stable"), "—")
        invades = "yes" if ratio > 1.0 else "no"
        lines.append(f"| {name} | {ratio:.3f} | {invades} | {unstable} "
                     f"| {stable} |")
    lines.append("")

    # CI-only threshold vs mu (numeric vs closed form)
    lines.append("## CI-only threshold p̂ vs µ\n")
    lines.append("| µ | p̂ numeric | p̂ closed form | µ/s_h |")
    lines.append("|---|---|---|---|")
    s_h, m_I, m_U, f = condition_params({"ci": True, "er": False, "ie": False})
    for mu in mu_grid:
        num = unstable_threshold(mu, s_h, m_I, m_U, f)
        cf = ci_threshold_closed_form(mu, s_h)
        num_s = "none" if np.isnan(num) else f"{num:.4f}"
        lines.append(f"| {mu:.2f} | {num_s} | {cf:.4f} | {mu:.4f} |")
    lines.append("")

    if straddle is not None:
        lines.append("## Stochastic straddle (finite-N WFM, CI-only)\n")
        lines.append("Median final frequency [95% percentile interval], "
                     "fraction fixed / lost across replicates.\n")
        lines.append("| µ | p̂ | side | p0 | median final | 95% CI "
                     "| frac fixed | frac lost |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in straddle:
            lines.append(
                f"| {r['mu']:.2f} | {r['p_hat']:.3f} | {r['side']} "
                f"| {r['p0']:.3f} | {r['median_final']:.3f} "
                f"| [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] "
                f"| {r['frac_fixed']:.2f} | {r['frac_lost']:.2f} |")
        lines.append("")

    lines.append("## Summary\n")
    lines.append(
        "With perfect transmission (µ = 0) the CI-only map has no interior "
        "equilibrium — infection invades from arbitrarily low frequency, so "
        "the low-frequency failure of CI in the finite-N simulations is "
        "drift, not a deterministic threshold. Introducing leakage µ creates "
        "a genuine unstable Turelli threshold at p̂ ≈ µ/s_h (numeric roots "
        "match the closed form). ER removes the threshold: its low-frequency "
        "invasion ratio (1−µ)/0.714 exceeds 1 for µ < ≈0.29, so CI+ER invades "
        "from rare while CI alone cannot. The ABM Δp crossover p* should be "
        "compared against this p̂ once the GPU sweep at µ = 0.03 completes.")
    lines.append("")

    out = outdir / "results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ======================================================================
#  CLI
# ======================================================================

def main() -> None:
    """Run the deterministic threshold analysis and write outputs."""
    parser = argparse.ArgumentParser(
        description="W.I.N.G.S. — WFM Turelli-threshold analysis")
    parser.add_argument("--outdir", default="results/threshold",
                        help="Output directory for figure, CSV and results.md")
    parser.add_argument("--mu-max", type=float, default=0.10,
                        help="Maximum µ for the curve (default 0.10)")
    parser.add_argument("--mu-step", type=float, default=0.01,
                        help="µ grid step (default 0.01)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Also run the finite-N WFM straddle (cheap)")
    parser.add_argument("--nreps", type=int, default=200,
                        help="Replicates for the stochastic straddle")
    args = parser.parse_args()

    matplotlib.use("Agg")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mu_grid = np.round(np.arange(0.0, args.mu_max + args.mu_step / 2,
                                 args.mu_step), 4)

    print("=" * 56)
    print("  W.I.N.G.S. — WFM Turelli-threshold analysis")
    print("=" * 56)
    print(f"  µ grid:   0.00 → {args.mu_max} step {args.mu_step}")
    print(f"  Output:   {outdir}/")

    stem = plot_threshold_vs_mu(mu_grid, outdir)
    print(f"  ✓ {stem.name}.{{png,svg,csv}}")

    straddle = None
    if args.stochastic:
        straddle = stochastic_straddle([0.01, 0.03, 0.05], n_reps=args.nreps)
        print(f"  ✓ stochastic straddle ({args.nreps} reps × 6 cells)")

    out = build_results_md(mu_grid, outdir, straddle)
    print(f"  ✓ {out.name}")
    print("=" * 56)


if __name__ == "__main__":
    main()
