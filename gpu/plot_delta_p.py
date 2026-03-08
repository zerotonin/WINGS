#!/usr/bin/env python3
"""
W.I.N.G.S. — Δp vs p figure (complementary strategies).

Produces publication figures showing how the per-interval change in
infection frequency (Δp) depends on the current frequency (p), for
CI-only, ER-only, and CI+ER phenotypes.

ABM data points are extracted from time series by computing
    Δp = p(t + Δt) - p(t)
then binning by p(t).

Analytical curves overlay the empirical data:
    s_CI(p)  ≈ p(1-p) · s_h · (1-μ) / (1 - p·s_h)    [Turelli 1994]
    s_ER(p)  = s_0 · (1-p)^β                            [diminishing returns]
    s_CI+ER  = s_CI + s_ER                               [additive]

Parameters are fit to the ABM data via least-squares, or set manually.

Outputs (PNG 300dpi + SVG with editable text):
    delta_p_vs_p.{png,svg}      — 3-panel (CI, ER, CI+ER)
    delta_p_overlay.{png,svg}   — single-panel overlay with crossing point

Usage:
    python plot_delta_p.py --input data/combined_delta_p.csv
    python plot_delta_p.py --input data/combined_delta_p.csv --dt 14
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
    # SVG: keep text as editable text objects (not outlines)
    "svg.fonttype": "none",
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

EGG_HATCH_DAY = 23  # skip pre-hatch period


# ======================================================================
#  Column detection
# ======================================================================

def detect_columns(df):
    """
    Auto-detect the time column and infection rate column names.
    Returns (time_col, infection_col).
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}

    # Time column candidates
    time_col = None
    for cand in ['day', 'days', 'time', 'hour', 'hours', 'generation', 'step']:
        if cand in cols_lower:
            time_col = cols_lower[cand]
            break
    if time_col is None:
        for c in df.columns:
            cl = c.lower()
            if 'day' in cl or 'time' in cl or 'generation' in cl:
                time_col = c
                break
    if time_col is None:
        print(f"  ERROR: Cannot find time column.")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)

    # Infection rate column candidates
    inf_col = None
    for cand in ['infection rate', 'infection_rate', 'infectionrate',
                  'inf_rate', 'infected_fraction']:
        if cand in cols_lower:
            inf_col = cols_lower[cand]
            break
    if inf_col is None:
        for c in df.columns:
            cl = c.lower()
            if 'infection' in cl and ('rate' in cl or 'frac' in cl):
                inf_col = c
                break
    if inf_col is None:
        print(f"  ERROR: Cannot find infection rate column.")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)

    return time_col, inf_col


# ======================================================================
#  Analytical models
# ======================================================================

def s_ci(p, s_h, mu):
    """CI selection coefficient (after Turelli 1994)."""
    numerator = p * (1 - p) * s_h * (1 - mu)
    denominator = 1 - p * s_h
    return numerator / np.maximum(denominator, 1e-9)


def s_er(p, s_0, beta):
    """ER selection coefficient (diminishing returns)."""
    return s_0 * (1 - p) ** beta


def s_ci_er(p, s_h, mu, s_0, beta):
    """Combined CI + ER: additive."""
    return s_ci(p, s_h, mu) + s_er(p, s_0, beta)


# ======================================================================
#  Data processing
# ======================================================================

def extract_delta_p(df, time_col, inf_col, dt=7):
    """
    For each replicate time series, compute (p, Δp) pairs.
    """
    records = []
    groups = df.groupby(['Phenotype', 'Infected Fraction', 'Replicate ID'])
    print(f"    Processing {len(groups)} time series...")

    for (pheno, frac, rep), grp in groups:
        grp = grp.sort_values(time_col)
        days = grp[time_col].values.astype(float)
        inf = grp[inf_col].values.astype(float)

        # Skip pre-hatch period
        mask = days >= EGG_HATCH_DAY
        days = days[mask]
        inf = inf[mask]

        if len(days) < 2:
            continue

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

            if p_t <= 0.01 or p_t >= 0.99:
                continue

            records.append({
                'Phenotype': pheno,
                'p': p_t,
                'delta_p': dp,
                'initial_fraction': frac,
            })

    return pd.DataFrame(records)


def bin_delta_p(dp_df, n_bins=20):
    """Bin (p, Δp) data and compute summary statistics per bin."""
    records = []
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    for pheno in dp_df['Phenotype'].unique():
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
    """Fit analytical s_CI and s_ER parameters to binned ABM data."""
    params = {}

    ci_data = binned_df[binned_df['Phenotype'] == 'CI']
    if len(ci_data) >= 3:
        try:
            popt, _ = curve_fit(
                s_ci, ci_data['p_mid'].values, ci_data['dp_median'].values,
                p0=[0.8, 0.05], bounds=([0, 0], [1, 0.5]), maxfev=5000)
            params['s_h'] = float(popt[0])
            params['mu'] = float(popt[1])
            print(f"  CI fit: s_h={popt[0]:.3f}, μ={popt[1]:.3f}")
        except Exception as e:
            print(f"  CI fit failed ({e}), using defaults")
            params['s_h'] = 0.8
            params['mu'] = 0.05
    else:
        print(f"  CI: {len(ci_data)} bins (need ≥3), using defaults")
        params['s_h'] = 0.8
        params['mu'] = 0.05

    er_data = binned_df[binned_df['Phenotype'] == 'ER']
    if len(er_data) >= 3:
        try:
            popt, _ = curve_fit(
                s_er, er_data['p_mid'].values, er_data['dp_median'].values,
                p0=[0.05, 1.0], bounds=([0, 0.1], [1, 5]), maxfev=5000)
            params['s_0'] = float(popt[0])
            params['beta'] = float(popt[1])
            print(f"  ER fit: s_0={popt[0]:.4f}, β={popt[1]:.2f}")
        except Exception as e:
            print(f"  ER fit failed ({e}), using defaults")
            params['s_0'] = 0.03
            params['beta'] = 1.5
    else:
        print(f"  ER: {len(er_data)} bins (need ≥3), using defaults")
        params['s_0'] = 0.03
        params['beta'] = 1.5

    return params


# ======================================================================
#  Figure saving
# ======================================================================

def save_fig(fig, path_stem):
    """Save as PNG (300dpi) + SVG (editable text)."""
    fig.savefig(f"{path_stem}.png")
    fig.savefig(f"{path_stem}.svg")
    plt.close(fig)
    print(f"    ✓ {Path(path_stem).name}.{{png,svg}}")


# ======================================================================
#  Plotting
# ======================================================================

def plot_delta_p_figure(binned_df, params, outdir, dt):
    """3-panel: CI, ER, CI+ER — ABM data + analytical overlay."""
    p_theory = np.linspace(0.02, 0.98, 200)

    theory_funcs = {
        'CI':    lambda p: s_ci(p, params['s_h'], params['mu']),
        'ER':    lambda p: s_er(p, params['s_0'], params['beta']),
        'CI_ER': lambda p: s_ci_er(p, params['s_h'], params['mu'],
                                    params['s_0'], params['beta']),
    }
    panel_titles = {'CI': 'CI only', 'ER': 'ER only', 'CI_ER': 'CI + ER'}
    panel_order = ['CI', 'ER', 'CI_ER']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, pheno in zip(axes, panel_order):
        color = PHENO_STYLE.get(pheno, {"color": COL_GREY})['color']

        # ABM data
        sub = binned_df[binned_df['Phenotype'] == pheno]
        if len(sub) > 0:
            ax.fill_between(sub['p_mid'], sub['dp_q25'], sub['dp_q75'],
                            alpha=0.2, color=color, linewidth=0, label='ABM IQR')
            ax.plot(sub['p_mid'], sub['dp_median'], 'o-', color=color,
                    markersize=4, linewidth=1.5, label='ABM median', zorder=4)

        # Analytical
        dp_theory = theory_funcs[pheno](p_theory)
        ax.plot(p_theory, dp_theory, '--', color='#333333', linewidth=2.0,
                alpha=0.7, label='Analytical', zorder=5)

        ax.axhline(0, color=COL_GREY, ls=':', lw=0.7, zorder=0)

        # Unstable equilibrium marker
        if pheno in ('CI', 'CI_ER'):
            sign_changes = np.where(np.diff(np.sign(dp_theory)))[0]
            for idx in sign_changes:
                p_eq = p_theory[idx]
                if 0.03 < p_eq < 0.97:
                    ax.axvline(p_eq, color=color, ls=':', lw=1.0, alpha=0.5)
                    ax.text(p_eq + 0.02, 0.002, f'p*={p_eq:.2f}',
                            fontsize=7, color=color, va='bottom')

        ax.set_xlabel('Infection frequency (p)')
        ax.set_title(panel_titles[pheno], fontweight='bold', color=color)
        ax.legend(loc='upper right', fontsize=7.5)
        ax.set_xlim(0, 1)

    axes[0].set_ylabel(f'Δp per {dt}-day interval')

    if len(binned_df) > 0:
        y_max = max(0.05, binned_df['dp_q75'].max() * 1.3)
        y_min = min(-0.02, binned_df['dp_q25'].min() * 1.3)
    else:
        y_max, y_min = 0.15, -0.02
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    fig.suptitle('Complementary frequency dependence of CI and ER',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, 'delta_p_vs_p'))


def plot_overlay_figure(binned_df, params, outdir, dt):
    """Single-panel overlay with ER/CI crossing point."""
    p_theory = np.linspace(0.02, 0.98, 200)

    fig, ax = plt.subplots(figsize=(7, 5))

    dp_ci = s_ci(p_theory, params['s_h'], params['mu'])
    dp_er = s_er(p_theory, params['s_0'], params['beta'])
    dp_cier = dp_ci + dp_er

    ax.plot(p_theory, dp_ci, '-', color=COL_CI, linewidth=2.2, label='CI (theory)')
    ax.plot(p_theory, dp_er, '-', color=COL_ER, linewidth=2.2, label='ER (theory)')
    ax.plot(p_theory, dp_cier, '-', color=COL_CIER, linewidth=2.2, label='CI+ER (theory)')

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
        p_cross = p_theory[idx]
        dp_cross = dp_ci[idx]
        if 0.05 < p_cross < 0.95:
            ax.plot(p_cross, dp_cross, 'k*', markersize=12, zorder=6)
            ax.annotate(
                f'p = {p_cross:.2f}\nER → CI handoff',
                xy=(p_cross, dp_cross),
                xytext=(p_cross + 0.10, dp_cross + 0.015),
                fontsize=8.5, ha='left',
                arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8),
                color='#333333')

    # Shaded regions
    if len(sign_changes) > 0:
        p_cross = p_theory[sign_changes[0]]
        ax.axvspan(0, p_cross, alpha=0.04, color=COL_ER, zorder=0)
        ax.axvspan(p_cross, 1, alpha=0.04, color=COL_CI, zorder=0)
        y_top = ax.get_ylim()[1]
        y_label = y_top * 0.85 if y_top > 0.01 else 0.04
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

    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, 'delta_p_overlay'))


# ======================================================================
#  CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="W.I.N.G.S. — Δp vs p figure generator")
    parser.add_argument('--input', required=True,
                        help='Combined CSV from ingest_delta_p.py')
    parser.add_argument('--outdir', default='figures_delta_p',
                        help='Output directory (default: figures_delta_p/)')
    parser.add_argument('--dt', type=int, default=7,
                        help='Δt in days for Δp computation (default: 7)')
    parser.add_argument('--n-bins', type=int, default=20,
                        help='Number of frequency bins (default: 20)')
    parser.add_argument('--s-h', type=float, default=None,
                        help='CI strength (override auto-fit)')
    parser.add_argument('--mu', type=float, default=None,
                        help='Maternal transmission loss (override auto-fit)')
    parser.add_argument('--s-0', type=float, default=None,
                        help='ER base advantage (override auto-fit)')
    parser.add_argument('--beta', type=float, default=None,
                        help='ER diminishing returns exponent (override auto-fit)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 56)
    print("  W.I.N.G.S. — Δp Figure Generator")
    print("=" * 56)
    print(f"  Input:   {args.input}")
    print(f"  Output:  {args.outdir}/")
    print(f"  Δt:      {args.dt} days")
    print(f"  Bins:    {args.n_bins}")

    df = pd.read_csv(args.input)

    # Auto-detect columns
    time_col, inf_col = detect_columns(df)
    print(f"  Columns: time='{time_col}', infection='{inf_col}'")
    print(f"  All columns: {list(df.columns)}")
    print(f"  Rows:    {len(df):,}")

    phenotypes = sorted(df['Phenotype'].unique())
    fractions = sorted(df['Infected Fraction'].unique())
    print(f"  Pheno:   {', '.join(phenotypes)}")
    print(f"  Fracs:   {len(fractions)} levels ({min(fractions):.2f}–{max(fractions):.2f})")
    print()

    # Step 1: Extract Δp
    print("  Extracting Δp from time series...")
    dp_df = extract_delta_p(df, time_col=time_col, inf_col=inf_col, dt=args.dt)
    print(f"  → {len(dp_df):,} (p, Δp) data points")

    if len(dp_df) == 0:
        print("\n  ERROR: No valid Δp data extracted!")
        print("  Check that time series have data after day 23 with 0.01 < p < 0.99")
        sys.exit(1)

    # Step 2: Bin
    print("  Binning by infection frequency...")
    binned = bin_delta_p(dp_df, n_bins=args.n_bins)
    print(f"  → {len(binned)} bins with ≥3 data points")
    for pheno in phenotypes:
        n = len(binned[binned['Phenotype'] == pheno])
        print(f"    {pheno}: {n} bins")

    # Step 3: Fit
    print("\n  Fitting analytical models...")
    params = fit_analytical(binned)

    if args.s_h is not None:
        params['s_h'] = args.s_h; print(f"  [override] s_h = {args.s_h}")
    if args.mu is not None:
        params['mu'] = args.mu; print(f"  [override] μ = {args.mu}")
    if args.s_0 is not None:
        params['s_0'] = args.s_0; print(f"  [override] s_0 = {args.s_0}")
    if args.beta is not None:
        params['beta'] = args.beta; print(f"  [override] β = {args.beta}")

    # Save intermediates
    dp_df.to_csv(os.path.join(args.outdir, 'raw_delta_p.csv'), index=False)
    binned.to_csv(os.path.join(args.outdir, 'binned_delta_p.csv'), index=False)
    with open(os.path.join(args.outdir, 'fitted_params.txt'), 'w') as f:
        for k, v in params.items():
            f.write(f"{k} = {v}\n")
    print(f"\n  Saved: raw_delta_p.csv, binned_delta_p.csv, fitted_params.txt")

    # Step 4: Plot
    print("\n  Generating figures (PNG + SVG)...")
    plot_delta_p_figure(binned, params, args.outdir, args.dt)
    plot_overlay_figure(binned, params, args.outdir, args.dt)

    print(f"\n  Done — 4 figure files saved to {args.outdir}/")
    print("=" * 56)


if __name__ == '__main__':
    main()
