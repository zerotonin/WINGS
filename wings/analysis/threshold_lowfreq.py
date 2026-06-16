#!/usr/bin/env python3
# ┌────────────────────────────────────────────────────────────┐
# │ threshold_lowfreq  « does the ABM show p̂ ≈ µ? »            │
# └────────────────────────────────────────────────────────────┘
"""Low-frequency Turelli-threshold analysis across phenotype conditions.

Each seed fraction p0 ∈ {0.5 … 6 %} is a *controlled* Δp probe: a trajectory
seeded exactly at p0 under one phenotype condition (CI, ER, or CI+ER).  For
every (condition, µ, p0) we average the net change Δp = p(end) − p(0) over
replicates.  Below the unstable equilibrium the infection declines (Δp < 0);
above it, it grows (Δp > 0).  The p0 at which the mean Δp changes sign is the
**empirical threshold p\\***, which we compare against the WFM closed form
p̂ ≈ µ.

For CI alone the threshold exists and tracks p̂ ≈ µ.  The relay hypothesis is
that adding exploration removes it: ER and CI+ER should give Δp > 0 even at
the lowest seeds, so their mean-Δp curves never cross zero (p\\* = nan, i.e.
no threshold).

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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from wings.analysis.threshold import (  # noqa: E402
    _TOL_BLUE,
    _TOL_GREY,
    _TOL_RED,
    ci_threshold_closed_form,
    save_fig,
)
from wings.analysis.plot_wings import get_style  # noqa: E402  (canonical Tol palette)


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


CONDITIONS = ("CI", "ER", "CI_ER")          # display order; CI_ER before CI in regex
_NAME_RE = re.compile(r"^(CI_ER|CI|ER)_frac(\d{4})_rep\d+\.csv$")


def _parse_name(name: str) -> tuple[str | None, float | None]:
    """Condition prefix and seed fraction from a low-freq CSV filename."""
    m = _NAME_RE.match(name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2)) / 1000


def _mu_from_path(path: str) -> float | None:
    m = re.search(r"lowfreq_mu([0-9]+(?:\.[0-9]+)?)", path)
    return float(m.group(1)) if m else None


def collect(root: str) -> dict[str, dict[float, dict[float, list[float]]]]:
    """{condition: {µ: {p0: [Δp per replicate]}}} from the low-freq sweep tree."""
    data: dict[str, dict[float, dict[float, list[float]]]] = {}
    pattern = f"{root}/abm_dp_lowfreq_mu*/*_frac*_rep*.csv"
    for f in glob.glob(pattern):
        cond, p0 = _parse_name(Path(f).name)
        mu = _mu_from_path(f)
        if cond is None or p0 is None or mu is None:
            continue
        a, b = _read_endpoints(f)
        if a is None or b is None:
            continue
        data.setdefault(cond, {}).setdefault(mu, {}).setdefault(p0, []).append(b - a)
    return data


def _zero_crossing(xs: list[float], ys: list[float], target: float = 0.0) -> float:
    """First x where y crosses target (rising), linear-interpolated."""
    for i in range(1, len(xs)):
        a, b = ys[i - 1] - target, ys[i] - target
        if a <= 0 < b or a < 0 <= b:
            return xs[i - 1] + (xs[i] - xs[i - 1]) * (-a) / (b - a)
    return float("nan")


def _regime(p_star: float, mean_dp: list[float]) -> str:
    """Classify a (condition, µ) curve.

    A nan ``p_star`` is ambiguous on its own — it means "no rising
    zero-crossing", which happens both when Δp > 0 everywhere (the
    infection invades from the lowest seed: the threshold is *removed*)
    and when Δp < 0 everywhere (it cannot establish at any tested seed).
    These are opposite outcomes, so classify explicitly.
    """
    if not np.isnan(p_star):
        return "threshold"           # rising crossing at p* (Turelli-like)
    if min(mean_dp) > 0:
        return "spreads"             # Δp>0 throughout → threshold removed
    if max(mean_dp) < 0:
        return "declines"            # Δp<0 throughout → no establishment
    return "mixed"                   # non-monotone / only a falling crossing


def summarise(data: dict[float, dict[float, list[float]]]) -> dict:
    """Per-µ curves + p* estimates + regime classification."""
    out = {}
    for mu in sorted(data):
        p0s = sorted(data[mu])
        mean_dp = [float(np.mean(data[mu][p])) for p in p0s]
        med_dp = [float(np.median(data[mu][p])) for p in p0s]
        q25_dp = [float(np.percentile(data[mu][p], 25)) for p in p0s]
        q75_dp = [float(np.percentile(data[mu][p], 75)) for p in p0s]
        frac_up = [float(np.mean([d > 0 for d in data[mu][p]])) for p in p0s]
        n = [len(data[mu][p]) for p in p0s]
        p_star_mean = _zero_crossing(p0s, mean_dp, 0.0)
        out[mu] = {
            "p0": p0s, "mean_dp": mean_dp, "median_dp": med_dp,
            "q25_dp": q25_dp, "q75_dp": q75_dp,
            "frac_up": frac_up, "n": n,
            "p_star_mean": p_star_mean,
            "p_star_up": _zero_crossing(p0s, frac_up, 0.5),
            "p_hat": ci_threshold_closed_form(mu),
            "regime": _regime(p_star_mean, mean_dp),
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


def plot_pstar(summ: dict, outdir: Path, boot_ci: dict | None = None) -> Path:
    """p* (ABM) vs p̂(µ) = WFM threshold, with 95% bootstrap CI errorbars."""
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
    if boot_ci is not None:
        lo = np.array([boot_ci[m]["pstar_ci_lo"] for m in mus])
        hi = np.array([boot_ci[m]["pstar_ci_hi"] for m in mus])
        yerr = np.clip(np.vstack([ps_mean - lo, hi - ps_mean]), 0, None)
        ax.errorbar(mus, ps_mean, yerr=yerr, fmt="o", color=_TOL_RED, ms=7,
                    capsize=3, elinewidth=1.2, zorder=4, mec="white", mew=0.7,
                    label=r"ABM $p^*$ (mean $\Delta p=0$, 95% CI)")
    else:
        ax.scatter(mus, ps_mean, s=52, color=_TOL_RED, zorder=4,
                   edgecolors="white", linewidths=0.7,
                   label=r"ABM $p^*$ (mean $\Delta p=0$)")
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


# Condition → canonical project combo label (Paul Tol palette in plot_wings)
# and a distinct marker so the small panels stay legible.
_COND_COMBO: dict[str, str] = {"CI": "CI", "ER": "ER", "CI_ER": "CI+ER"}
_COND_MARKER: dict[str, str] = {"CI": "o", "ER": "s", "CI_ER": "^"}
_COND_LABEL: dict[str, str] = {"CI": "CI", "ER": "ER", "CI_ER": "CI+ER"}


def plot_relay(summ_by_cond: dict, outdir: Path) -> Path:
    """Per-µ panels of mean Δp(p0) for CI / ER / CI+ER, with replicate IQR."""
    conds = [c for c in CONDITIONS if c in summ_by_cond]
    mus = sorted({m for c in conds for m in summ_by_cond[c]})
    ncols = 3
    nrows = int(np.ceil(len(mus) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for idx, mu in enumerate(mus):
        ax = axes.flat[idx]
        ax.set_visible(True)
        for cond in conds:
            s = summ_by_cond[cond].get(mu)
            if not s:
                continue
            color, dash, _ = get_style(_COND_COMBO[cond])
            ax.fill_between(s["p0"], s["q25_dp"], s["q75_dp"],
                            color=color, alpha=0.12, lw=0, zorder=1)
            ax.plot(s["p0"], s["mean_dp"], color=color, ls=dash,
                    marker=_COND_MARKER[cond], ms=3.5, lw=1.5, zorder=3,
                    label=_COND_LABEL[cond])
        ax.axhline(0.0, color=_TOL_GREY, lw=0.9, zorder=0)
        p_hat = ci_threshold_closed_form(mu)
        if not np.isnan(p_hat):
            ax.axvline(p_hat, color=_TOL_GREY, ls=":", lw=0.9)
        ax.set_title(f"µ = {mu:.2f}", fontsize=9, fontweight="bold")
        if idx % ncols == 0:
            ax.set_ylabel(r"mean $\Delta p$ (72 d)")
        ax.set_xlabel("seed freq $p_0$")
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Frequency-dependent relay (dotted = CI threshold p̂; bands = replicate IQR):\n"
                 "CI+ER pushes the threshold far below p̂ or removes it; "
                 "ER alone is insufficient at high µ",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    stem = outdir / "fig_lowfreq_relay"
    save_fig(fig, stem)
    return stem


# ┌────────────────────────────────────────────────────────────┐
# │ Bootstrap  « replicate-resampled CIs + threshold test »      │
# └────────────────────────────────────────────────────────────┘
_COND_SEED = {"CI": 1, "ER": 2, "CI_ER": 3}


def _pstar_numeric(p0s: list[float], mean_dp: list[float]) -> float:
    """Threshold as a single number: crossing, 0 if spreads, nan if declines."""
    pc = _zero_crossing(p0s, mean_dp, 0.0)
    if not np.isnan(pc):
        return pc
    if min(mean_dp) > 0:
        return 0.0                    # Δp>0 throughout → threshold removed
    return float("nan")               # Δp<0 throughout → no establishment


def bootstrap_cell(rep_by_p0: dict[float, list[float]], n_boot: int,
                   seed: int) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    """Resample replicates → bootstrap mean/median curves and p* draws."""
    p0s = sorted(rep_by_p0)
    rng = np.random.default_rng(seed)
    bmean = np.empty((n_boot, len(p0s)))
    bmed = np.empty((n_boot, len(p0s)))
    for j, p in enumerate(p0s):
        a = np.asarray(rep_by_p0[p], float)
        samp = a[rng.integers(0, a.size, size=(n_boot, a.size))]
        bmean[:, j] = samp.mean(axis=1)
        bmed[:, j] = np.median(samp, axis=1)
    bpstar = np.array([_pstar_numeric(p0s, bmean[b].tolist()) for b in range(n_boot)])
    return p0s, bmean, bmed, bpstar


def build_bootstrap(data: dict, n_boot: int = 2000) -> dict:
    """Per (condition, µ): central curves + 95% CIs + bootstrap p* draws."""
    boot: dict = {}
    for cond in data:
        boot[cond] = {}
        for mu in sorted(data[cond]):
            seed = _COND_SEED.get(cond, 0) * 1_000_000 + int(round(mu * 1000))
            p0s, bmean, bmed, bpstar = bootstrap_cell(data[cond][mu], n_boot, seed)
            lo_m, hi_m = np.percentile(bmean, [2.5, 97.5], axis=0)
            lo_md, hi_md = np.percentile(bmed, [2.5, 97.5], axis=0)
            valid = bpstar[~np.isnan(bpstar)]
            ps_lo, ps_hi, ps_med = (
                (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)),
                 float(np.median(valid))) if valid.size else (np.nan, np.nan, np.nan))
            boot[cond][mu] = {
                "p0": p0s,
                "mean": [float(np.mean(data[cond][mu][p])) for p in p0s],
                "mean_ci_lo": list(lo_m), "mean_ci_hi": list(hi_m),
                "median": [float(np.median(data[cond][mu][p])) for p in p0s],
                "median_ci_lo": list(lo_md), "median_ci_hi": list(hi_md),
                "pstar_boot": bpstar,
                "pstar_med": ps_med, "pstar_ci_lo": ps_lo, "pstar_ci_hi": ps_hi,
            }
    return boot


def _bh_fdr(pvals: list[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    adj = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def threshold_tests(boot: dict, n_boot: int,
                    conds: tuple[str, str] = ("CI", "CI_ER")) -> list[dict]:
    """Per-µ bootstrap difference test of p*(CI) vs p*(CI+ER), BH-corrected.

    The threshold is a derived statistic, so reRandomStats' raw two-sample
    resampling does not apply; instead we use the bootstrap difference
    distribution (a non-parametric test) and BH-correct across µ.
    """
    a, b = conds
    mus = sorted(set(boot.get(a, {})) & set(boot.get(b, {})))
    rows = []
    for mu in mus:
        d = boot[a][mu]["pstar_boot"] - boot[b][mu]["pstar_boot"]
        d = d[~np.isnan(d)]
        p = 1.0 if d.size == 0 else min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
        p = max(p, 1.0 / n_boot)              # bootstrap p-value floor
        rows.append({"mu": mu,
                     "pstar_CI": boot[a][mu]["pstar_med"],
                     "pstar_CI_ER": boot[b][mu]["pstar_med"],
                     "delta_med": float(np.median(d)) if d.size else np.nan,
                     # fraction of CI resamples that themselves spread (p*=0):
                     # a high value flags a threshold sitting at the noise floor,
                     # which is why the difference can be ns at the lowest µ.
                     "ci_spread_frac": float(np.mean(boot[a][mu]["pstar_boot"] == 0)),
                     "p_raw": p})
    for row, q in zip(rows, _bh_fdr([r["p_raw"] for r in rows])):
        row["p_bh"] = float(q)
    return rows


def _stars(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def plot_dp_curves_ci(boot: dict, outdir: Path) -> Path:
    """CI-only mean Δp(p0) per µ with a 95% CI band of the mean."""
    b = boot["CI"]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    cmap = plt.get_cmap("viridis")
    mus = sorted(b)
    for i, mu in enumerate(mus):
        s = b[mu]
        c = cmap(i / max(1, len(mus) - 1))
        ax.fill_between(s["p0"], s["mean_ci_lo"], s["mean_ci_hi"],
                        color=c, alpha=0.15, lw=0)
        ax.plot(s["p0"], s["mean"], "-o", color=c, ms=4, lw=1.4,
                label=f"µ={mu:.2f} (p̂={ci_threshold_closed_form(mu):.3f})")
        ph = ci_threshold_closed_form(mu)
        if not np.isnan(ph):
            ax.axvline(ph, color=c, ls=":", lw=0.8, alpha=0.6)
    ax.axhline(0.0, color=_TOL_GREY, lw=1.0, zorder=0)
    ax.set_xlabel("seed infection frequency $p_0$")
    ax.set_ylabel(r"mean net $\Delta p$ over 72 days")
    ax.set_title("CI-only: net change vs seed frequency\n"
                 "(band = 95% CI of the mean; dotted = WFM p̂)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    stem = outdir / "fig_lowfreq_dp_curves_ci"
    save_fig(fig, stem)
    return stem


def plot_relay_ci(boot: dict, outdir: Path) -> Path:
    """Relay panels using the median Δp with a 95% CI-of-median band."""
    conds = [c for c in CONDITIONS if c in boot]
    mus = sorted({m for c in conds for m in boot[c]})
    ncols = 3
    nrows = int(np.ceil(len(mus) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for idx, mu in enumerate(mus):
        ax = axes.flat[idx]
        ax.set_visible(True)
        for cond in conds:
            s = boot[cond].get(mu)
            if not s:
                continue
            color, dash, _ = get_style(_COND_COMBO[cond])
            ax.fill_between(s["p0"], s["median_ci_lo"], s["median_ci_hi"],
                            color=color, alpha=0.18, lw=0, zorder=1)
            ax.plot(s["p0"], s["median"], color=color, ls=dash,
                    marker=_COND_MARKER[cond], ms=3.5, lw=1.5, zorder=3,
                    label=_COND_LABEL[cond])
        ax.axhline(0.0, color=_TOL_GREY, lw=0.9, zorder=0)
        ph = ci_threshold_closed_form(mu)
        if not np.isnan(ph):
            ax.axvline(ph, color=_TOL_GREY, ls=":", lw=0.9)
        ax.set_title(f"µ = {mu:.2f}", fontsize=9, fontweight="bold")
        if idx % ncols == 0:
            ax.set_ylabel(r"median $\Delta p$ (72 d)")
        ax.set_xlabel("seed freq $p_0$")
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Frequency-dependent relay (median Δp; bands = 95% CI of the median):\n"
                 "CI+ER pushes the threshold far below p̂ or removes it; "
                 "ER alone is insufficient at high µ",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    stem = outdir / "fig_lowfreq_relay_ci"
    save_fig(fig, stem)
    return stem


def plot_threshold_box(boot: dict, rows: list[dict], outdir: Path) -> Path:
    """Equal-aspect boxplots of bootstrapped p* for CI vs CI+ER, p* against µ.

    µ is on the x-axis (numeric), p* on the y-axis at the same scale, so the
    p̂ ≈ µ relationship reads as the diagonal.  Whiskers are the standard
    Tukey 1.5×IQR; outliers are drawn (coloured dots) so the mass at p*=0
    from resamples that spread (CI at µ=0.01) stays visible — the source of
    any ns result.
    """
    mus = [r["mu"] for r in rows]
    box_w, off = 0.0016, 0.0021
    col_ci = get_style("CI")[0]
    col_cier = get_style("CI+ER")[0]
    hi = max(mus) + 0.013
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.plot([0, hi], [0, hi], color=_TOL_GREY, ls=":", lw=1.0, zorder=0,
            label=r"$\hat p=\mu$")
    for r in rows:
        mu = r["mu"]
        for cond, col, sign in (("CI", col_ci, -1), ("CI_ER", col_cier, +1)):
            arr = boot[cond][mu]["pstar_boot"]
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                continue
            bp = ax.boxplot(arr, positions=[mu + sign * off], widths=box_w,
                            patch_artist=True, showfliers=True,
                            flierprops=dict(marker="o", markersize=2.5,
                                            markerfacecolor=col, alpha=0.25,
                                            markeredgecolor=col, linestyle="none"),
                            manage_ticks=False)
            for box in bp["boxes"]:
                box.set(facecolor=col, alpha=0.6, edgecolor=col)
            for med in bp["medians"]:
                med.set(color="black", lw=1.4)
            for w in bp["whiskers"] + bp["caps"]:
                w.set(color=col)
        ax.plot(mu, ci_threshold_closed_form(mu), "D", color=_TOL_GREY,
                ms=6, zorder=5)
        top = max(np.nanpercentile(boot["CI"][mu]["pstar_boot"], 97.5),
                  np.nanpercentile(boot["CI_ER"][mu]["pstar_boot"], 97.5))
        y = top + 0.0035
        ax.plot([mu - off, mu + off], [y, y], color="black", lw=0.9)
        ax.text(mu, y, _stars(r["p_bh"]), ha="center", va="bottom", fontsize=9)
    handles = [Rectangle((0, 0), 1, 1, fc=col_ci, alpha=0.6),
               Rectangle((0, 0), 1, 1, fc=col_cier, alpha=0.6),
               Line2D([0], [0], marker="D", color=_TOL_GREY, ls=""),
               Line2D([0], [0], color=_TOL_GREY, ls=":")]
    ax.legend(handles, ["CI", "CI+ER", "WFM p̂", r"$\hat p=\mu$"],
              loc="upper left", fontsize=8)
    ax.set_xticks(mus)
    ax.set_xticklabels([f"{m:.2f}" for m in mus])
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"maternal-transmission leakage $\mu$")
    ax.set_ylabel(r"Turelli threshold $p^{*}$ (bootstrap)")
    ax.set_title("CI vs CI+ER threshold (bootstrap p*; BH-corrected test)",
                 fontweight="bold", pad=10)
    fig.tight_layout()
    stem = outdir / "fig_lowfreq_threshold_box"
    save_fig(fig, stem)
    return stem


def write_threshold_tests(rows: list[dict], outdir: Path) -> None:
    with open(outdir / "lowfreq_threshold_tests.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mu", "pstar_CI", "pstar_CI_ER", "delta_median",
                    "ci_spread_frac", "p_raw", "p_bh"])
        for r in rows:
            w.writerow([f"{r['mu']:.3f}", f"{r['pstar_CI']:.4f}",
                        f"{r['pstar_CI_ER']:.4f}", f"{r['delta_med']:.4f}",
                        f"{r['ci_spread_frac']:.3f}",
                        f"{r['p_raw']:.4g}", f"{r['p_bh']:.4g}"])
    lines = ["# CI vs CI+ER threshold test (bootstrap difference, BH-FDR)", "",
             "Two-sided bootstrap p that p\\*(CI) = p\\*(CI+ER), per µ; "
             "BH-corrected across µ.  `CI spread%` is the fraction of CI "
             "bootstrap resamples that themselves spread (p\\*=0): when high, "
             "the CI threshold sits at the noise floor and the difference is "
             "not resolvable (e.g. µ=0.01).", "",
             "| µ | p* CI | p* CI+ER | Δ (CI−CI+ER) | CI spread% | p (raw) | p (BH) | sig |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['mu']:.2f} | {r['pstar_CI']:.4f} | "
                     f"{r['pstar_CI_ER']:.4f} | {r['delta_med']:+.4f} | "
                     f"{100*r['ci_spread_frac']:.1f}% | "
                     f"{r['p_raw']:.4g} | {r['p_bh']:.4g} | {_stars(r['p_bh'])} |")
    (outdir / "lowfreq_threshold_tests.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def write_tables(summ_by_cond: dict, outdir: Path) -> None:
    conds = [c for c in CONDITIONS if c in summ_by_cond]
    with open(outdir / "lowfreq_dp_by_seed.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "mu", "p0", "n", "mean_dp", "median_dp", "frac_up"])
        for cond in conds:
            for mu in sorted(summ_by_cond[cond]):
                s = summ_by_cond[cond][mu]
                for i, p0 in enumerate(s["p0"]):
                    w.writerow([cond, f"{mu:.3f}", f"{p0:.4f}", s["n"][i],
                                f"{s['mean_dp'][i]:.6f}", f"{s['median_dp'][i]:.6f}",
                                f"{s['frac_up'][i]:.4f}"])
    lines = ["# Low-frequency threshold test (CI / ER / CI+ER)", "",
             "p\\* = seed fraction where mean Δp crosses zero (the unstable "
             "equilibrium). **regime** disambiguates a missing crossing:",
             "",
             "- `threshold` — rising crossing at p\\* (Turelli-like)",
             "- `spreads` — Δp > 0 at every seed → threshold removed, "
             "invades from the rarest seed",
             "- `declines` — Δp < 0 at every seed → cannot establish in the "
             "tested range",
             "- `mixed` — non-monotone / only a falling crossing",
             ""]
    for cond in conds:
        summ = summ_by_cond[cond]
        lines += [f"## {_COND_LABEL[cond]}", "",
                  "| µ | regime | p* (mean Δp=0) | p* (50% grow) | WFM p̂ | p*−p̂ |",
                  "|---|---|---|---|---|---|"]
        for mu in sorted(summ):
            s = summ[mu]
            pm = "—" if np.isnan(s["p_star_mean"]) else f"{s['p_star_mean']:.4f}"
            pu = "—" if np.isnan(s["p_star_up"]) else f"{s['p_star_up']:.4f}"
            diff = ("—" if np.isnan(s["p_star_mean"])
                    else f"{s['p_star_mean'] - s['p_hat']:+.4f}")
            lines.append(f"| {mu:.2f} | {s['regime']} | {pm} | {pu} | "
                         f"{s['p_hat']:.4f} | {diff} |")
        lines.append("")
    (outdir / "lowfreq_results.md").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Low-frequency CI-only Turelli-threshold analysis")
    parser.add_argument("--data-root", required=True,
                        help="Dir containing abm_dp_lowfreq_mu*/ subdirs")
    parser.add_argument("--outdir", default="results/threshold_lowfreq")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="Bootstrap resamples for CIs and the threshold test")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = collect(args.data_root)
    if not data:
        raise SystemExit(f"No low-freq CSVs found under {args.data_root}")
    summ_by_cond = {cond: summarise(data[cond]) for cond in data}

    for cond in [c for c in CONDITIONS if c in summ_by_cond]:
        print(f"=== {_COND_LABEL[cond]} ===")
        for mu in sorted(summ_by_cond[cond]):
            s = summ_by_cond[cond][mu]
            print(f"  µ={mu:.2f}  p̂={s['p_hat']:.4f}  "
                  f"p*_mean={s['p_star_mean']:.4f}  p*_up={s['p_star_up']:.4f}  "
                  f"[{s['regime']}]  "
                  f"(seeds={len(s['p0'])}, reps≈{int(np.median(s['n']))})")

    boot = build_bootstrap(data, n_boot=args.n_boot)

    if "CI" in summ_by_cond:                  # threshold-validation figures (CI)
        plot_dp_curves(summ_by_cond["CI"], outdir)
        plot_dp_curves_ci(boot, outdir)
        plot_pstar(summ_by_cond["CI"], outdir, boot_ci=boot["CI"])
    plot_relay(summ_by_cond, outdir)          # relay (IQR bands)
    plot_relay_ci(boot, outdir)               # relay (95% CI of median)

    if "CI" in boot and "CI_ER" in boot:      # CI vs CI+ER threshold comparison
        rows = threshold_tests(boot, args.n_boot)
        plot_threshold_box(boot, rows, outdir)
        write_threshold_tests(rows, outdir)
        print("\n=== CI vs CI+ER threshold (bootstrap, BH-FDR) ===")
        for r in rows:
            print(f"  µ={r['mu']:.2f}  p*_CI={r['pstar_CI']:.4f}  "
                  f"p*_CI+ER={r['pstar_CI_ER']:.4f}  "
                  f"Δ={r['delta_med']:+.4f}  p_BH={r['p_bh']:.4g} {_stars(r['p_bh'])}")

    write_tables(summ_by_cond, outdir)
    print(f"\nFigures + tables written to {outdir}")


if __name__ == "__main__":
    main()
