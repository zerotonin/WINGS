#!/usr/bin/env python3
# ┌────────────────────────────────────────────────────────────┐
# │ threshold_lowfreq  « does the ABM show p̂ ≈ µ? »            │
# └────────────────────────────────────────────────────────────┘
"""Low-frequency CI-only Turelli-threshold analysis.

Each seed fraction p0 ∈ {0.5 … 5 %} is a *controlled* Δp probe: a CI-only
trajectory seeded exactly at p0.  For every (µ, p0) we average the net
change Δp = p(end) − p(0) over replicates.  Below the unstable equilibrium
the infection declines (Δp < 0); above it, it grows (Δp > 0).  The p0 at
which the mean Δp changes sign is the **empirical threshold p\\***, which we
compare against the WFM closed form p̂ ≈ µ.

Two observables, for robustness against runaway-trajectory outliers:
  * ``p*_mean`` — sign change of the mean Δp(p0)
  * ``p*_up``   — where the fraction of replicates that grew crosses 0.5
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wings.analysis.threshold import (  # noqa: E402
    _TOL_BLUE,
    _TOL_GREY,
    _TOL_RED,
    ci_threshold_closed_form,
    save_fig,
)


def _read_endpoints(path: str) -> tuple[float | None, float | None]:
    """First and last infection rate in a trajectory CSV."""
    first = last = None
    with open(path, newline="", encoding="utf-8") as fh:
        rdr = csv.reader(fh)
        next(rdr, None)
        for row in rdr:
            if len(row) < 2:
                continue
            try:
                v = float(row[1])
            except ValueError:
                continue
            if first is None:
                first = v
            last = v
    return first, last


def _frac_from_name(name: str) -> float | None:
    m = re.search(r"frac(\d{4})", name)
    return int(m.group(1)) / 1000 if m else None


def _mu_from_path(path: str) -> float | None:
    m = re.search(r"lowfreq_mu([0-9]+(?:\.[0-9]+)?)", path)
    return float(m.group(1)) if m else None


def collect(root: str) -> dict[float, dict[float, list[float]]]:
    """{µ: {p0: [Δp per replicate]}} from the low-freq sweep tree."""
    data: dict[float, dict[float, list[float]]] = {}
    pattern = f"{root}/abm_dp_lowfreq_mu*/CI_frac*_rep*.csv"
    for f in glob.glob(pattern):
        mu = _mu_from_path(f)
        p0 = _frac_from_name(Path(f).name)
        if mu is None or p0 is None:
            continue
        a, b = _read_endpoints(f)
        if a is None or b is None:
            continue
        data.setdefault(mu, {}).setdefault(p0, []).append(b - a)
    return data


def _zero_crossing(xs: list[float], ys: list[float], target: float = 0.0) -> float:
    """First x where y crosses target (rising), linear-interpolated."""
    for i in range(1, len(xs)):
        a, b = ys[i - 1] - target, ys[i] - target
        if a <= 0 < b or a < 0 <= b:
            return xs[i - 1] + (xs[i] - xs[i - 1]) * (-a) / (b - a)
    return float("nan")


def summarise(data: dict[float, dict[float, list[float]]]) -> dict:
    """Per-µ curves + p* estimates."""
    out = {}
    for mu in sorted(data):
        p0s = sorted(data[mu])
        mean_dp = [float(np.mean(data[mu][p])) for p in p0s]
        med_dp = [float(np.median(data[mu][p])) for p in p0s]
        frac_up = [float(np.mean([d > 0 for d in data[mu][p]])) for p in p0s]
        n = [len(data[mu][p]) for p in p0s]
        out[mu] = {
            "p0": p0s, "mean_dp": mean_dp, "median_dp": med_dp,
            "frac_up": frac_up, "n": n,
            "p_star_mean": _zero_crossing(p0s, mean_dp, 0.0),
            "p_star_up": _zero_crossing(p0s, frac_up, 0.5),
            "p_hat": ci_threshold_closed_form(mu),
        }
    return out


def plot_dp_curves(summ: dict, outdir: Path) -> Path:
    """Mean Δp(p0) per µ, with the WFM p̂ marked."""
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    cmap = plt.get_cmap("viridis")
    mus = sorted(summ)
    for i, mu in enumerate(mus):
        s = summ[mu]
        c = cmap(i / max(1, len(mus) - 1))
        ax.plot(s["p0"], s["mean_dp"], "-o", color=c, ms=4, lw=1.4,
                label=f"µ={mu:.2f} (p̂={s['p_hat']:.3f})")
        if not np.isnan(s["p_hat"]):
            ax.axvline(s["p_hat"], color=c, ls=":", lw=0.8, alpha=0.6)
    ax.axhline(0.0, color=_TOL_GREY, lw=1.0, zorder=0)
    ax.set_xlabel("seed infection frequency $p_0$")
    ax.set_ylabel(r"mean net $\Delta p$ over 72 days")
    ax.set_title("CI-only: net change vs seed frequency\n"
                 "(sign change = empirical threshold; dotted = WFM p̂)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    stem = outdir / "fig_lowfreq_dp_curves"
    save_fig(fig, stem)
    return stem


def plot_pstar(summ: dict, outdir: Path) -> Path:
    """p* (ABM) vs p̂(µ) = WFM threshold."""
    mus = np.array(sorted(summ))
    ps_mean = np.array([summ[m]["p_star_mean"] for m in mus])
    ps_up = np.array([summ[m]["p_star_up"] for m in mus])
    fine = np.linspace(0, max(float(mus.max()), 0.05), 200)
    closed = np.array([ci_threshold_closed_form(m) for m in fine])

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot(fine, closed, color=_TOL_BLUE, lw=2.0, zorder=2,
            label=r"WFM threshold $\hat p(\mu)$")
    ax.plot(fine, fine, color=_TOL_GREY, ls=":", lw=1.0, zorder=1,
            label=r"$\hat p=\mu$")
    ax.scatter(mus, ps_mean, s=52, color=_TOL_RED, zorder=4, edgecolors="white",
               linewidths=0.7, label=r"ABM $p^*$ (mean $\Delta p=0$)")
    ax.scatter(mus, ps_up, s=42, marker="^", color="#228833", zorder=4,
               edgecolors="white", linewidths=0.7,
               label=r"ABM $p^*$ (50% grow)")
    ax.set_xlabel(r"maternal-transmission leakage $\mu$")
    ax.set_ylabel("infection frequency")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Does the ABM threshold track p̂ ≈ µ?",
                 fontweight="bold", pad=10)
    fig.tight_layout()
    stem = outdir / "fig_lowfreq_pstar_vs_threshold"
    save_fig(fig, stem)
    return stem


def write_tables(summ: dict, outdir: Path) -> None:
    with open(outdir / "lowfreq_dp_by_seed.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mu", "p0", "n", "mean_dp", "median_dp", "frac_up"])
        for mu in sorted(summ):
            s = summ[mu]
            for i, p0 in enumerate(s["p0"]):
                w.writerow([f"{mu:.3f}", f"{p0:.4f}", s["n"][i],
                            f"{s['mean_dp'][i]:.6f}", f"{s['median_dp'][i]:.6f}",
                            f"{s['frac_up'][i]:.4f}"])
    lines = ["# Low-frequency CI-only threshold test", "",
             "| µ | p* (mean Δp=0) | p* (50% grow) | WFM p̂ | p*−p̂ |",
             "|---|---|---|---|---|"]
    for mu in sorted(summ):
        s = summ[mu]
        pm = "nan" if np.isnan(s["p_star_mean"]) else f"{s['p_star_mean']:.4f}"
        pu = "nan" if np.isnan(s["p_star_up"]) else f"{s['p_star_up']:.4f}"
        diff = ("nan" if np.isnan(s["p_star_mean"])
                else f"{s['p_star_mean'] - s['p_hat']:+.4f}")
        lines.append(f"| {mu:.2f} | {pm} | {pu} | {s['p_hat']:.4f} | {diff} |")
    (outdir / "lowfreq_results.md").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Low-frequency CI-only Turelli-threshold analysis")
    parser.add_argument("--data-root", required=True,
                        help="Dir containing abm_dp_lowfreq_mu*/ subdirs")
    parser.add_argument("--outdir", default="results/threshold_lowfreq")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = collect(args.data_root)
    if not data:
        raise SystemExit(f"No low-freq CSVs found under {args.data_root}")
    summ = summarise(data)

    for mu in sorted(summ):
        s = summ[mu]
        print(f"µ={mu:.2f}  p̂={s['p_hat']:.4f}  "
              f"p*_mean={s['p_star_mean']:.4f}  p*_up={s['p_star_up']:.4f}  "
              f"(seeds={len(s['p0'])}, reps≈{int(np.median(s['n']))})")

    plot_dp_curves(summ, outdir)
    plot_pstar(summ, outdir)
    write_tables(summ, outdir)
    print(f"\nFigures + tables written to {outdir}")


if __name__ == "__main__":
    main()
