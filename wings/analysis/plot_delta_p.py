#!/usr/bin/env python3
"""
W.I.N.G.S. — Δp vs p figure (complementary strategies).

Extracts per-interval infection frequency change (Δp) from ABM time
series and overlays analytical models.

Key design choices
------------------
- **First-passage extraction**: for each replicate, only uses the FIRST
  time the trajectory passes through each p-bin (ascending), avoiding
  contamination from equilibrium wobble or stalled runs.
- **Asymmetric CI model**: A·p^α·(1-p)^γ allows the peak to shift right,
  capturing spatial/growth effects the mean-field Turelli model misses.
- **Diminishing-returns ER model**: s_0·(1-p)^β captures mate-finding
  saturation.

Outputs (PNG 300dpi + SVG with editable text):
    delta_p_vs_p.{png,svg}      — 3-panel (CI, ER, CI+ER)
    delta_p_overlay.{png,svg}   — single-panel with crossing point

Usage:
    python plot_delta_p.py --input data/combined_delta_p.csv
    python plot_delta_p.py --input data/combined_delta_p.csv --dt 14 --ascending-only
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=FutureWarning)

# ======================================================================
#  Style
# ======================================================================

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.8",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "svg.fonttype": "none",  # editable text in SVG
})

# Paul Tol colours
COL_CI   = "#4477AA"
COL_ER   = "#228833"
COL_CIER = "#009988"
COL_GREY = "#BBBBBB"

PHENO_STYLE = {
    "CI":    {"color": COL_CI,   "label": "CI only"},
    "ER":    {"color": COL_ER,   "label": "ER only"},
    "CI_ER": {"color": COL_CIER, "label": "CI + ER"},
}

EGG_HATCH_DAY = 23


# Module-level flag, set from CLI
USE_SYMLOG_Y = False


def apply_symlog_y(ax, binned_df=None, dp_col='dp_median'):
    """
    Apply symmetric-log y-axis if USE_SYMLOG_Y is True.
    Auto-computes linthresh from the data: the linear region spans
    the smallest non-trivial median Δp, so the ER signal (small but
    positive) stays visible while the CI peak gets compressed.
    """
    if not USE_SYMLOG_Y:
        return

    # Auto-compute linthresh from the smallest positive median
    if binned_df is not None and dp_col in binned_df.columns:
        abs_vals = binned_df[dp_col].abs()
        pos_vals = abs_vals[abs_vals > 0]
        if len(pos_vals) > 0:
            linthresh = float(pos_vals.quantile(0.1))
        else:
            linthresh = 1e-4
    else:
        linthresh = 1e-4

    # Ensure linthresh is sensible
    linthresh = max(linthresh, 1e-6)

    ax.set_yscale('symlog', linthresh=linthresh)

    # Nice tick formatting — avoid scientific notation clutter
    from matplotlib.ticker import FuncFormatter
    def fmt(x, _):
        if abs(x) < linthresh:
            return f'{x:.1e}'
        elif abs(x) < 0.01:
            return f'{x:.4f}'
        elif abs(x) < 0.1:
            return f'{x:.3f}'
        else:
            return f'{x:.2f}'
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))


# ======================================================================
#  Column detection
# ======================================================================

def detect_columns(df):
    """Auto-detect time and infection rate column names."""
    cols_lower = {c.lower().strip(): c for c in df.columns}

    time_col = None
    for cand in ['day', 'days', 'time', 'hour', 'hours', 'generation', 'step']:
        if cand in cols_lower:
            time_col = cols_lower[cand]
            break
    if time_col is None:
        for c in df.columns:
            if any(k in c.lower() for k in ['day', 'time', 'generation']):
                time_col = c
                break
    if time_col is None:
        print(f"  ERROR: No time column found. Columns: {list(df.columns)}")
        sys.exit(1)

    inf_col = None
    for cand in ['infection rate', 'infection_rate', 'infectionrate']:
        if cand in cols_lower:
            inf_col = cols_lower[cand]
            break
    if inf_col is None:
        for c in df.columns:
            if 'infection' in c.lower():
                inf_col = c
                break
    if inf_col is None:
        print(f"  ERROR: No infection rate column found. Columns: {list(df.columns)}")
        sys.exit(1)

    return time_col, inf_col


# ======================================================================
#  Analytical models
# ======================================================================

def model_ci_asymmetric(p, A, alpha, gamma):
    """
    Asymmetric CI selection: A · p^α · (1-p)^γ.
    Generalises the Turelli p(1-p) form (which is α=γ=1).
    When α > γ the peak shifts right (higher p), matching ABM
    where spatial clustering and population growth sustain CI
    advantage beyond p = 0.5.
    """
    return A * np.power(p, alpha) * np.power(1 - p, gamma)


def model_er(p, s_0, beta):
    """
    ER diminishing returns: s_0 · (1-p)^β.
    Mate-finding advantage strongest when infected females are
    rare, saturates as they become common.
    """
    return s_0 * np.power(1 - p, beta)


def model_ci_er(p, A, alpha, gamma, s_0, beta):
    """Combined CI + ER (additive)."""
    return model_ci_asymmetric(p, A, alpha, gamma) + model_er(p, s_0, beta)


# ======================================================================
#  Δp extraction
# ======================================================================

def extract_delta_p(df, time_col, inf_col, dt=7, ascending_only=False):
    """
    Extract (p, Δp) pairs from each replicate time series.

    Parameters
    ----------
    ascending_only : bool
        If True, only use intervals where p is increasing (Δp > 0)
        and only the FIRST passage through each p-range. This isolates
        the active invasion phase from equilibrium noise.
    """
    records = []
    groups = df.groupby(['Phenotype', 'Infected Fraction', 'Replicate ID'])
    print(f"    Processing {len(groups)} time series...")

    # Detect population size column
    pop_col = None
    for c in df.columns:
        if 'population' in c.lower() and 'size' in c.lower():
            pop_col = c
            break
    if pop_col is None:
        for c in df.columns:
            if 'population' in c.lower() or 'pop' in c.lower():
                pop_col = c
                break
    has_pop = pop_col is not None
    if has_pop:
        print(f"    Population column: '{pop_col}'")

    for (pheno, frac, rep), grp in groups:
        grp = grp.sort_values(time_col)
        days = grp[time_col].values.astype(float)
        inf = grp[inf_col].values.astype(float)
        pop = grp[pop_col].values.astype(float) if has_pop else None

        # Skip pre-hatch
        mask = days >= EGG_HATCH_DAY
        days = days[mask]
        inf = inf[mask]
        if pop is not None:
            pop = pop[mask]

        if len(days) < 2:
            continue

        # Track which p-bins we've already visited (first-passage mode)
        visited_bins = set()

        t_start = days[0]
        t_end = days[-1]
        sample_times = np.arange(t_start, t_end - dt + 1, dt)

        for t in sample_times:
            idx_t = np.argmin(np.abs(days - t))
            idx_tdt = np.argmin(np.abs(days - (t + dt)))

            if idx_t == idx_tdt:
                continue

            p_t = float(inf[idx_t])
            p_tdt = float(inf[idx_tdt])
            dp = p_tdt - p_t

            # Skip absorbing boundaries
            if p_t <= 0.01 or p_t >= 0.99:
                continue

            if ascending_only:
                # Only ascending trajectory
                if dp <= 0:
                    continue
                # First-passage: only count each 5%-bin once
                p_bin = int(p_t * 20)  # 0.05 bins
                if p_bin in visited_bins:
                    continue
                visited_bins.add(p_bin)

            record = {
                'Phenotype': pheno,
                'p': p_t,
                'delta_p': dp,
                'initial_fraction': frac,
            }
            if pop is not None:
                record['pop_size'] = float(pop[idx_t])
            records.append(record)

    return pd.DataFrame(records)


def extract_delta_p_initial(df, time_col, inf_col, dt=24):
    """
    Turelli-correct extraction: ONE Δp per replicate.

    For each replicate, measures the change in infection frequency over
    a single generation interval (dt days) starting from the first
    post-hatch time point.  This directly maps to Turelli's recursion:
        "given p now, what is E[Δp] over one generation?"

    The 19 initial fractions (0.05–0.95) provide the p-axis.
    50 replicates per condition give the variance.

    No sign filtering, no trajectory averaging, no contamination from
    equilibrium phases.  One clean measurement per replicate.
    """
    records = []
    groups = df.groupby(['Phenotype', 'Infected Fraction', 'Replicate ID'])
    print(f"    Processing {len(groups)} time series (initial Δp only)...")

    # Detect population size column
    pop_col = None
    for c in df.columns:
        if 'population' in c.lower() and 'size' in c.lower():
            pop_col = c
            break
    if pop_col is None:
        for c in df.columns:
            if 'population' in c.lower() or 'pop' in c.lower():
                pop_col = c
                break

    for (pheno, frac, rep), grp in groups:
        grp = grp.sort_values(time_col)
        days = grp[time_col].values.astype(float)
        inf = grp[inf_col].values.astype(float)
        pop = grp[pop_col].values.astype(float) if pop_col else None

        # Skip to post-hatch
        mask = days >= EGG_HATCH_DAY
        days = days[mask]
        inf = inf[mask]
        if pop is not None:
            pop = pop[mask]

        if len(days) < 2:
            continue

        # Single measurement: first post-hatch point → one generation later
        t0 = days[0]
        idx_t1 = np.argmin(np.abs(days - (t0 + dt)))

        if idx_t1 == 0:
            continue

        p_t0 = float(inf[0])
        p_t1 = float(inf[idx_t1])
        dp = p_t1 - p_t0

        record = {
            'Phenotype': pheno,
            'p': p_t0,
            'delta_p': dp,
            'initial_fraction': frac,
        }
        if pop is not None:
            record['pop_size'] = float(pop[0])
        records.append(record)

    return pd.DataFrame(records)


def bin_delta_p(dp_df, n_bins=20):
    """Bin (p, Δp) and compute summary statistics."""
    records = []
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    for pheno in sorted(dp_df['Phenotype'].unique()):
        sub = dp_df[dp_df['Phenotype'] == pheno]
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = sub[(sub['p'] >= lo) & (sub['p'] < hi)]
            if len(in_bin) < 3:
                continue
            records.append({
                'Phenotype': pheno,
                'p_mid': bin_mids[i],
                'dp_median': in_bin['delta_p'].median(),
                'dp_q25': in_bin['delta_p'].quantile(0.25),
                'dp_q75': in_bin['delta_p'].quantile(0.75),
                'dp_mean': in_bin['delta_p'].mean(),
                'n': len(in_bin),
            })

    return pd.DataFrame(records)


# ======================================================================
#  Curve fitting
# ======================================================================

def fit_analytical(binned_df):
    """Fit CI (asymmetric) and ER models to binned ABM data."""
    params = {}

    # --- CI: A · p^α · (1-p)^γ ---
    ci_data = binned_df[binned_df['Phenotype'] == 'CI']
    if len(ci_data) >= 4:
        try:
            popt, _ = curve_fit(
                model_ci_asymmetric,
                ci_data['p_mid'].values,
                ci_data['dp_median'].values,
                p0=[0.01, 1.5, 0.8],
                bounds=([0, 0.1, 0.1], [1, 5, 5]),
                maxfev=10000,
            )
            params['A'] = float(popt[0])
            params['alpha'] = float(popt[1])
            params['gamma'] = float(popt[2])
            print(f"  CI fit: A={popt[0]:.5f}, α={popt[1]:.2f}, γ={popt[2]:.2f}")
            if popt[1] > popt[2]:
                print(f"    → peak shifted RIGHT (α > γ): spatial/growth effects")
        except Exception as e:
            print(f"  CI fit failed ({e}), using defaults")
            params['A'] = 0.008
            params['alpha'] = 1.5
            params['gamma'] = 0.8
    else:
        print(f"  CI: {len(ci_data)} bins, using defaults")
        params['A'] = 0.008
        params['alpha'] = 1.5
        params['gamma'] = 0.8

    # --- ER: s_0 · (1-p)^β ---
    er_data = binned_df[binned_df['Phenotype'] == 'ER']
    if len(er_data) >= 3:
        try:
            popt, _ = curve_fit(
                model_er,
                er_data['p_mid'].values,
                er_data['dp_median'].values,
                p0=[0.001, 1.0],
                bounds=([0, 0.1], [0.1, 10]),
                maxfev=10000,
            )
            params['s_0'] = float(popt[0])
            params['beta'] = float(popt[1])
            print(f"  ER fit: s_0={popt[0]:.6f}, β={popt[1]:.2f}")
        except Exception as e:
            print(f"  ER fit failed ({e}), using defaults")
            params['s_0'] = 0.0005
            params['beta'] = 1.5
    else:
        print(f"  ER: {len(er_data)} bins, using defaults")
        params['s_0'] = 0.0005
        params['beta'] = 1.5

    return params


# ======================================================================
#  Figure saving
# ======================================================================

def save_fig(fig, path_stem):
    """Save PNG (300dpi) + SVG (editable text)."""
    fig.savefig(f"{path_stem}.png")
    fig.savefig(f"{path_stem}.svg")
    plt.close(fig)
    print(f"    ✓ {Path(path_stem).name}.{{png,svg}}")


# ======================================================================
#  3-panel figure
# ======================================================================

def plot_delta_p_figure(binned_df, params, outdir, dt):
    """3-panel: CI, ER, CI+ER with ABM data + analytical overlay."""
    p_th = np.linspace(0.02, 0.98, 200)

    theory = {
        'CI':    model_ci_asymmetric(p_th, params['A'], params['alpha'], params['gamma']),
        'ER':    model_er(p_th, params['s_0'], params['beta']),
        'CI_ER': (model_ci_asymmetric(p_th, params['A'], params['alpha'], params['gamma'])
                  + model_er(p_th, params['s_0'], params['beta'])),
    }
    titles = {'CI': 'CI only', 'ER': 'ER only', 'CI_ER': 'CI + ER'}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, pheno in zip(axes, ['CI', 'ER', 'CI_ER']):
        color = PHENO_STYLE[pheno]['color']

        # ABM data
        sub = binned_df[binned_df['Phenotype'] == pheno]
        if len(sub) > 0:
            ax.fill_between(sub['p_mid'], sub['dp_q25'], sub['dp_q75'],
                            alpha=0.2, color=color, linewidth=0, label='ABM IQR')
            ax.plot(sub['p_mid'], sub['dp_median'], 'o-', color=color,
                    markersize=4, linewidth=1.5, label='ABM median', zorder=4)

        # Theory
        ax.plot(p_th, theory[pheno], '--', color='#333333', linewidth=2.0,
                alpha=0.7, label='Analytical', zorder=5)

        ax.axhline(0, color=COL_GREY, ls=':', lw=0.7, zorder=0)

        # Unstable equilibrium (zero-crossing) for CI panels
        if pheno in ('CI', 'CI_ER'):
            dp = theory[pheno]
            crossings = np.where(np.diff(np.sign(dp)))[0]
            for idx in crossings:
                p_eq = p_th[idx]
                if 0.03 < p_eq < 0.97:
                    ax.axvline(p_eq, color=color, ls=':', lw=1.0, alpha=0.5)
                    ax.text(p_eq + 0.02, 0.0001, f'p*={p_eq:.2f}',
                            fontsize=7, color=color, va='bottom')

        ax.set_xlabel('Infection frequency (p)')
        ax.set_title(titles[pheno], fontweight='bold', color=color)
        ax.legend(loc='upper right', fontsize=7.5)
        ax.set_xlim(0, 1)

    axes[0].set_ylabel(f'Δp per {dt}-day interval')

    if len(binned_df) > 0:
        y_max = max(0.001, binned_df['dp_q75'].max() * 1.4)
        y_min = min(-0.0005, binned_df['dp_q25'].min() * 1.4)
    else:
        y_max, y_min = 0.005, -0.001
    for ax in axes:
        ax.set_ylim(y_min, y_max)
        apply_symlog_y(ax, binned_df)

    fig.suptitle('Complementary frequency dependence of CI and ER',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, 'delta_p_vs_p'))


# ======================================================================
#  Overlay figure
# ======================================================================

def plot_overlay_figure(binned_df, params, outdir, dt):
    """Single-panel overlay with ER/CI crossing point."""
    p_th = np.linspace(0.02, 0.98, 200)

    dp_ci = model_ci_asymmetric(p_th, params['A'], params['alpha'], params['gamma'])
    dp_er = model_er(p_th, params['s_0'], params['beta'])
    dp_cier = dp_ci + dp_er

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(p_th, dp_ci, '-', color=COL_CI, linewidth=2.2, label='CI (theory)')
    ax.plot(p_th, dp_er, '-', color=COL_ER, linewidth=2.2, label='ER (theory)')
    ax.plot(p_th, dp_cier, '-', color=COL_CIER, linewidth=2.2, label='CI+ER (theory)')

    # ABM scatter
    for pheno in ['CI', 'ER', 'CI_ER']:
        sub = binned_df[binned_df['Phenotype'] == pheno]
        if len(sub) == 0:
            continue
        style = PHENO_STYLE[pheno]
        ax.scatter(sub['p_mid'], sub['dp_median'], color=style['color'],
                   s=30, alpha=0.6, zorder=3, edgecolors='white', linewidths=0.4,
                   label=f'{style["label"]} (ABM)')

    # Crossing point
    diff = dp_ci - dp_er
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    for idx in sign_changes:
        p_cross = p_th[idx]
        dp_cross = dp_ci[idx]
        if 0.03 < p_cross < 0.97:
            ax.plot(p_cross, dp_cross, 'k*', markersize=12, zorder=6)
            ax.annotate(
                f'p = {p_cross:.2f}\nER → CI handoff',
                xy=(p_cross, dp_cross),
                xytext=(p_cross + 0.10, dp_cross + max(0.0005, dp_cross * 0.3)),
                fontsize=8.5, ha='left',
                arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8),
                color='#333333')

    # Shaded regions
    if len(sign_changes) > 0:
        p_cross = p_th[sign_changes[0]]
        ax.axvspan(0, p_cross, alpha=0.04, color=COL_ER, zorder=0)
        ax.axvspan(p_cross, 1, alpha=0.04, color=COL_CI, zorder=0)
        y_top = ax.get_ylim()[1]
        y_label = y_top * 0.85 if y_top > 0.001 else 0.001
        ax.text(p_cross / 2, y_label, 'ER dominant', fontsize=9,
                ha='center', color=COL_ER, alpha=0.8, fontstyle='italic')
        ax.text((1 + p_cross) / 2, y_label, 'CI dominant', fontsize=9,
                ha='center', color=COL_CI, alpha=0.8, fontstyle='italic')

    ax.axhline(0, color=COL_GREY, ls=':', lw=0.7, zorder=0)
    ax.set_xlabel('Infection frequency (p)')
    ax.set_ylabel(f'Δp per {dt}-day interval')
    ax.set_xlim(0, 1)
    ax.set_title('Complementary selection: ER bootstraps, CI amplifies',
                 fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=8, ncol=2)

    apply_symlog_y(ax, binned_df)

    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, 'delta_p_overlay'))


# ======================================================================
#  3D figure: p × N × Δp
# ======================================================================

def plot_delta_p_3d(dp_df, outdir, dt):
    """
    3D scatter: X = infection frequency, Y = population size, Z = Δp.
    One colour per phenotype. Shows how both frequency AND population
    size jointly determine the invasion rate.
    """
    if 'pop_size' not in dp_df.columns:
        print("    [skip] 3D plot — no population size data")
        return

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for pheno in ['CI', 'ER', 'CI_ER']:
        sub = dp_df[dp_df['Phenotype'] == pheno]
        if len(sub) == 0:
            continue
        style = PHENO_STYLE[pheno]

        # Subsample for readability (max 3000 points per phenotype)
        if len(sub) > 3000:
            sub = sub.sample(n=3000, random_state=42)

        ax.scatter(
            sub['p'], sub['pop_size'], sub['delta_p'],
            c=style['color'], s=6, alpha=0.25, label=style['label'],
            edgecolors='none', depthshade=True,
        )

    ax.set_xlabel('Infection frequency (p)', labelpad=10)
    ax.set_ylabel('Population size (N)', labelpad=10)
    ax.set_zlabel(f'Δp per {dt}-day interval', labelpad=10)
    ax.set_title('Δp depends on both frequency and population size',
                 fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=8)

    # Viewing angle: slightly elevated, rotated to show the CI ridge
    ax.view_init(elev=25, azim=-50)

    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, 'delta_p_3d'))


# ======================================================================
#  Δp vs population size overlay
# ======================================================================

def bin_delta_p_by_pop(dp_df, n_bins=20):
    """Bin (N, Δp) by population size and compute summary stats."""
    if 'pop_size' not in dp_df.columns:
        return pd.DataFrame()

    records = []
    # Use log-spaced bins since population grows exponentially
    pop_min = max(dp_df['pop_size'].min(), 10)
    pop_max = dp_df['pop_size'].max()
    bin_edges = np.logspace(np.log10(pop_min), np.log10(pop_max), n_bins + 1)
    bin_mids = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean

    for pheno in sorted(dp_df['Phenotype'].unique()):
        sub = dp_df[dp_df['Phenotype'] == pheno]
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = sub[(sub['pop_size'] >= lo) & (sub['pop_size'] < hi)]
            if len(in_bin) < 3:
                continue
            records.append({
                'Phenotype': pheno,
                'pop_mid': bin_mids[i],
                'dp_median': in_bin['delta_p'].median(),
                'dp_q25': in_bin['delta_p'].quantile(0.25),
                'dp_q75': in_bin['delta_p'].quantile(0.75),
                'n': len(in_bin),
            })

    return pd.DataFrame(records)


def plot_delta_p_vs_pop(dp_df, outdir, dt, n_bins=20):
    """
    Overlay: Δp vs population size for CI, ER, CI+ER.
    Reveals whether invasion rate depends on absolute population size
    (e.g. ER advantage might scale with density, CI might not).
    """
    if 'pop_size' not in dp_df.columns:
        print("    [skip] Δp vs N plot — no population size data")
        return

    binned_pop = bin_delta_p_by_pop(dp_df, n_bins=n_bins)
    if len(binned_pop) == 0:
        print("    [skip] Δp vs N plot — insufficient data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for pheno in ['CI', 'ER', 'CI_ER']:
        sub = binned_pop[binned_pop['Phenotype'] == pheno]
        if len(sub) == 0:
            continue
        style = PHENO_STYLE[pheno]

        ax.fill_between(sub['pop_mid'], sub['dp_q25'], sub['dp_q75'],
                        alpha=0.15, color=style['color'], linewidth=0)
        ax.plot(sub['pop_mid'], sub['dp_median'], 'o-',
                color=style['color'], markersize=4, linewidth=1.5,
                label=style['label'], zorder=4)

    ax.axhline(0, color=COL_GREY, ls=':', lw=0.7, zorder=0)
    ax.set_xscale('log')
    ax.set_xlabel('Population size (N)')
    ax.set_ylabel(f'Δp per {dt}-day interval')
    ax.set_title('Invasion rate vs population size', fontweight='bold', pad=10)
    ax.legend(loc='best', fontsize=9)

    apply_symlog_y(ax, binned_pop)

    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, 'delta_p_vs_pop'))

    # Also save the binned CSV
    binned_pop.to_csv(os.path.join(outdir, 'binned_delta_p_by_pop.csv'), index=False)
    print(f"    ✓ binned_delta_p_by_pop.csv")


# ======================================================================
#  Comparison: 3 extraction methods side by side
# ======================================================================

def plot_method_comparison(binned_all, binned_asc, binned_init, params, outdir, dt):
    """
    3-panel comparison: all-data vs ascending-only vs initial-generation.
    Same analytical curves in each, different ABM data.
    Shows how the choice of extraction window affects the picture.
    """
    p_th = np.linspace(0.02, 0.98, 200)

    dp_ci = model_ci_asymmetric(p_th, params['A'], params['alpha'], params['gamma'])
    dp_er = model_er(p_th, params['s_0'], params['beta'])

    datasets = [
        (binned_all,  'All Δp (trajectory average)'),
        (binned_asc,  'Ascending first-passage only'),
        (binned_init, 'Initial generation only (Turelli)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for ax, (binned, panel_title) in zip(axes, datasets):
        if binned is None or len(binned) == 0:
            ax.set_title(panel_title + '\n(no data)', fontsize=10)
            continue

        # Plot ABM data for each phenotype
        for pheno in ['CI', 'ER', 'CI_ER']:
            sub = binned[binned['Phenotype'] == pheno]
            if len(sub) == 0:
                continue
            style = PHENO_STYLE[pheno]
            ax.fill_between(sub['p_mid'], sub['dp_q25'], sub['dp_q75'],
                            alpha=0.15, color=style['color'], linewidth=0)
            ax.plot(sub['p_mid'], sub['dp_median'], 'o-',
                    color=style['color'], markersize=4, linewidth=1.5,
                    label=style['label'], zorder=4)

        # Theory curves (dashed, light)
        ax.plot(p_th, dp_ci, '--', color=COL_CI, linewidth=1.2, alpha=0.5)
        ax.plot(p_th, dp_er, '--', color=COL_ER, linewidth=1.2, alpha=0.5)

        ax.axhline(0, color=COL_GREY, ls=':', lw=0.7, zorder=0)
        ax.set_xlabel('Infection frequency (p)')
        ax.set_xlim(0, 1)
        ax.set_title(panel_title, fontweight='bold', fontsize=10)
        ax.legend(loc='upper right', fontsize=7)
        apply_symlog_y(ax, binned)

    axes[0].set_ylabel(f'Δp per {dt}-day interval')

    fig.suptitle('Extraction method comparison', fontsize=13,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, 'delta_p_method_comparison'))


# ======================================================================
#  CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="W.I.N.G.S. — Δp vs p figure generator")
    parser.add_argument('--input', required=True,
                        help='Combined CSV from ingest_delta_p.py')
    parser.add_argument('--outdir', default='figures_delta_p',
                        help='Output directory')
    parser.add_argument('--dt', type=int, default=24,
                        help='Δt in days — use 24 for T. confusum generation (default: 24)')
    parser.add_argument('--n-bins', type=int, default=20,
                        help='Number of frequency bins (default: 20)')
    parser.add_argument('--mode', default='all',
                        choices=['all', 'ascending', 'initial', 'compare'],
                        help='Extraction mode: all (trajectory avg), ascending '
                             '(first-passage Δp>0), initial (one Δp per rep, '
                             'Turelli-correct), compare (run all 3 side-by-side)')
    # Manual overrides
    parser.add_argument('--A', type=float, default=None, help='CI amplitude')
    parser.add_argument('--alpha', type=float, default=None, help='CI left exponent')
    parser.add_argument('--gamma', type=float, default=None, help='CI right exponent')
    parser.add_argument('--s-0', type=float, default=None, help='ER base advantage')
    parser.add_argument('--beta', type=float, default=None, help='ER decay exponent')
    parser.add_argument('--semilogy', action='store_true',
                        help='Use symmetric-log y-axis (log scale that handles '
                             'zero/negative values via linear region around 0)')
    args = parser.parse_args()

    # Set module-level flag
    global USE_SYMLOG_Y
    USE_SYMLOG_Y = args.semilogy

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 60)
    print("  W.I.N.G.S. — Δp Figure Generator")
    print("=" * 60)
    print(f"  Input:      {args.input}")
    print(f"  Output:     {args.outdir}/")
    print(f"  Δt:         {args.dt} days")
    print(f"  Bins:       {args.n_bins}")
    print(f"  Mode:       {args.mode}")
    print(f"  Semilogy:   {args.semilogy}")

    df = pd.read_csv(args.input)
    time_col, inf_col = detect_columns(df)
    print(f"  Columns:    time='{time_col}', infection='{inf_col}'")
    print(f"  Rows:       {len(df):,}")

    phenotypes = sorted(df['Phenotype'].unique())
    fractions = sorted(df['Infected Fraction'].unique())
    print(f"  Phenotypes: {', '.join(phenotypes)}")
    print(f"  Fractions:  {len(fractions)} ({min(fractions):.2f}–{max(fractions):.2f})")
    print()

    # =================================================================
    #  Extract Δp — depends on mode
    # =================================================================

    if args.mode == 'compare':
        # Run ALL three methods and produce comparison figure
        print("  ── Mode: COMPARE (all 3 extraction methods) ──")

        print("\n  [1/3] All-data extraction...")
        dp_all = extract_delta_p(df, time_col, inf_col,
                                 dt=args.dt, ascending_only=False)
        print(f"        → {len(dp_all):,} points")
        binned_all = bin_delta_p(dp_all, n_bins=args.n_bins) if len(dp_all) > 0 else pd.DataFrame()

        print("\n  [2/3] Ascending first-passage extraction...")
        dp_asc = extract_delta_p(df, time_col, inf_col,
                                 dt=args.dt, ascending_only=True)
        print(f"        → {len(dp_asc):,} points")
        binned_asc = bin_delta_p(dp_asc, n_bins=args.n_bins) if len(dp_asc) > 0 else pd.DataFrame()

        print("\n  [3/3] Initial-generation extraction (Turelli)...")
        dp_init = extract_delta_p_initial(df, time_col, inf_col, dt=args.dt)
        print(f"        → {len(dp_init):,} points (one per replicate)")
        binned_init = bin_delta_p(dp_init, n_bins=args.n_bins) if len(dp_init) > 0 else pd.DataFrame()

        # Fit on all-data (most data, best fit)
        print("\n  Fitting analytical models (on all-data)...")
        params = fit_analytical(binned_all)

        # Also fit on initial-generation for comparison
        print("  Fitting analytical models (on initial-generation)...")
        params_init = fit_analytical(binned_init)

        # Save all intermediates
        for name, data in [('raw_delta_p_all', dp_all),
                           ('raw_delta_p_ascending', dp_asc),
                           ('raw_delta_p_initial', dp_init),
                           ('binned_delta_p_all', binned_all),
                           ('binned_delta_p_ascending', binned_asc),
                           ('binned_delta_p_initial', binned_init)]:
            if len(data) > 0:
                data.to_csv(os.path.join(args.outdir, f'{name}.csv'), index=False)

        with open(os.path.join(args.outdir, 'fitted_params_all.txt'), 'w') as f:
            for k, v in sorted(params.items()):
                f.write(f"{k} = {v}\n")
        with open(os.path.join(args.outdir, 'fitted_params_initial.txt'), 'w') as f:
            for k, v in sorted(params_init.items()):
                f.write(f"{k} = {v}\n")

        # Use all-data for the primary figures, initial for comparison
        dp_df = dp_all
        binned = binned_all

        # Apply overrides
        for key, arg_val in [('A', args.A), ('alpha', args.alpha), ('gamma', args.gamma),
                              ('s_0', args.s_0), ('beta', args.beta)]:
            if arg_val is not None:
                params[key] = arg_val

        print("\n  Generating figures (PNG + SVG)...")
        plot_delta_p_figure(binned, params, args.outdir, args.dt)
        plot_overlay_figure(binned, params, args.outdir, args.dt)
        plot_delta_p_3d(dp_df, args.outdir, args.dt)
        plot_delta_p_vs_pop(dp_df, args.outdir, args.dt, n_bins=args.n_bins)

        # Comparison figure
        plot_method_comparison(binned_all, binned_asc, binned_init,
                               params, args.outdir, args.dt)

        # Extra: initial-generation overlay with its own fit
        init_dir = os.path.join(args.outdir, 'initial')
        os.makedirs(init_dir, exist_ok=True)
        plot_overlay_figure(binned_init, params_init, init_dir, args.dt)

    else:
        # Single mode
        if args.mode == 'initial':
            print("  ── Mode: INITIAL (one Δp per replicate, Turelli-correct) ──")
            dp_df = extract_delta_p_initial(df, time_col, inf_col, dt=args.dt)
        elif args.mode == 'ascending':
            print("  ── Mode: ASCENDING (first-passage, Δp > 0 only) ──")
            dp_df = extract_delta_p(df, time_col, inf_col,
                                    dt=args.dt, ascending_only=True)
        else:
            print("  ── Mode: ALL (full trajectory average) ──")
            dp_df = extract_delta_p(df, time_col, inf_col,
                                    dt=args.dt, ascending_only=False)

        print(f"  → {len(dp_df):,} (p, Δp) data points")

        if len(dp_df) == 0:
            print("\n  ERROR: No valid Δp data!")
            sys.exit(1)

        binned = bin_delta_p(dp_df, n_bins=args.n_bins)
        for pheno in phenotypes:
            n = len(binned[binned['Phenotype'] == pheno])
            print(f"    {pheno}: {n} bins")

        print("\n  Fitting analytical models...")
        params = fit_analytical(binned)

        for key, arg_val in [('A', args.A), ('alpha', args.alpha), ('gamma', args.gamma),
                              ('s_0', args.s_0), ('beta', args.beta)]:
            if arg_val is not None:
                params[key] = arg_val
                print(f"  [override] {key} = {arg_val}")

        dp_df.to_csv(os.path.join(args.outdir, 'raw_delta_p.csv'), index=False)
        binned.to_csv(os.path.join(args.outdir, 'binned_delta_p.csv'), index=False)
        with open(os.path.join(args.outdir, 'fitted_params.txt'), 'w') as f:
            for k, v in sorted(params.items()):
                f.write(f"{k} = {v}\n")

        print("\n  Generating figures (PNG + SVG)...")
        plot_delta_p_figure(binned, params, args.outdir, args.dt)
        plot_overlay_figure(binned, params, args.outdir, args.dt)
        plot_delta_p_3d(dp_df, args.outdir, args.dt)
        plot_delta_p_vs_pop(dp_df, args.outdir, args.dt, n_bins=args.n_bins)

    print(f"\n  Done — figures in {args.outdir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()