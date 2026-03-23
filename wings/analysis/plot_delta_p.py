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


# ======================================================================
#  Column detection
# ======================================================================

def detect_columns(df):
    """Auto-detect the time and infection rate column names.

    Searches for common variants (``Day``, ``Days``, ``Time``, etc.)
    case-insensitively.

    Args:
        df (pandas.DataFrame): Input data.

    Returns:
        tuple[str, str]: ``(time_column_name, infection_column_name)``.

    Raises:
        SystemExit: If either column cannot be found.
    """
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
    """Asymmetric CI selection coefficient: ``A · p^α · (1-p)^γ``.

    Generalises the Turelli (1994) model ``p(1-p) · s_h / (1 - p·s_h)``
    by allowing different exponents on the left (α) and right (γ)
    sides.  When α > γ the peak shifts right of 0.5, capturing
    spatial clustering and population growth effects.

    Args:
        p (numpy.ndarray): Infection frequency array.
        A (float): Amplitude parameter.
        alpha (float): Left exponent (rise rate at low p).
        gamma (float): Right exponent (decay rate at high p).

    Returns:
        numpy.ndarray: Expected Δp per generation.
    """
    return A * np.power(p, alpha) * np.power(1 - p, gamma)


def model_er(p, s_0, beta):
    """ER selection coefficient with diminishing returns: ``s_0 · (1-p)^β``.

    The mate-finding advantage of increased exploration decays as
    infected females become common and the mating landscape saturates.

    Args:
        p (numpy.ndarray): Infection frequency array.
        s_0 (float): Base advantage at p → 0.
        beta (float): Decay exponent (higher = faster saturation).

    Returns:
        numpy.ndarray: Expected Δp per generation from ER.
    """
    return s_0 * np.power(1 - p, beta)


def model_ci_er(p, A, alpha, gamma, s_0, beta):
    """Combined CI + ER selection (additive).

    Args:
        p (numpy.ndarray): Infection frequency.
        A (float): CI amplitude.
        alpha (float): CI left exponent.
        gamma (float): CI right exponent.
        s_0 (float): ER base advantage.
        beta (float): ER decay exponent.

    Returns:
        numpy.ndarray: Combined expected Δp.
    """
    return model_ci_asymmetric(p, A, alpha, gamma) + model_er(p, s_0, beta)


# ======================================================================
#  Δp extraction
# ======================================================================

def extract_delta_p(df, time_col, inf_col, dt=7, ascending_only=False):
    """Extract ``(p, Δp)`` pairs from replicate time series.

    For each replicate, samples the infection rate at intervals of
    ``dt`` days and computes the change.  Skips the pre-hatch
    period (day < 23) and absorbing boundaries (p < 0.01 or p > 0.99).

    Args:
        df (pandas.DataFrame): Combined simulation data.
        time_col (str): Name of the time column.
        inf_col (str): Name of the infection rate column.
        dt (int): Sampling interval in days. Defaults to ``7``.
        ascending_only (bool): If ``True``, only use intervals where
            Δp > 0 and only the first passage through each 5%-bin.
            Defaults to ``False``.

    Returns:
        pandas.DataFrame: Columns ``Phenotype``, ``p``, ``delta_p``,
        ``initial_fraction``, and optionally ``pop_size``.
    """
    records = []
    groups = df.groupby(['Phenotype', 'Infected Fraction', 'Replicate ID'])
    print(f"    Processing {len(groups)} time series...")

    for (pheno, frac, rep), grp in groups:
        grp = grp.sort_values(time_col)
        days = grp[time_col].values.astype(float)
        inf = grp[inf_col].values.astype(float)

        # Skip pre-hatch
        mask = days >= EGG_HATCH_DAY
        days = days[mask]
        inf = inf[mask]

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

            records.append({
                'Phenotype': pheno,
                'p': p_t,
                'delta_p': dp,
                'initial_fraction': frac,
            })

    return pd.DataFrame(records)


def bin_delta_p(dp_df, n_bins=20):
    """Bin ``(p, Δp)`` data by infection frequency and compute statistics.

    Args:
        dp_df (pandas.DataFrame): Raw Δp data from extraction.
        n_bins (int): Number of equal-width bins over [0, 1].
            Defaults to ``20``.

    Returns:
        pandas.DataFrame: Per-bin summary with ``Phenotype``,
        ``p_mid``, ``dp_median``, ``dp_q25``, ``dp_q75``,
        ``dp_mean``, ``n``.
    """
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
    """Fit the asymmetric CI and diminishing-returns ER models.

    Uses ``scipy.optimize.curve_fit`` on the binned median Δp data.
    Falls back to default parameters if fitting fails or data is
    insufficient.

    Args:
        binned_df (pandas.DataFrame): Binned Δp data.

    Returns:
        dict: Fitted parameters ``{A, alpha, gamma, s_0, beta}``.
    """
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
    """Save a figure as PNG (300 dpi) and SVG (editable text).

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        path_stem (str): Output path without extension.
    """
    fig.savefig(f"{path_stem}.png")
    fig.savefig(f"{path_stem}.svg")
    plt.close(fig)
    print(f"    ✓ {Path(path_stem).name}.{{png,svg}}")


# ======================================================================
#  3-panel figure
# ======================================================================

def plot_delta_p_figure(binned_df, params, outdir, dt):
    """Three-panel figure: CI, ER, CI+ER with ABM data and analytical overlay.

    Each panel shows binned ABM median + IQR ribbon and a dashed
    analytical curve.  CI panels mark the unstable equilibrium
    (Δp = 0 crossing).

    Args:
        binned_df (pandas.DataFrame): Binned Δp data.
        params (dict): Fitted analytical parameters.
        outdir (str): Output directory.
        dt (int): Time interval used for Δp.
    """
    p_th = np.linspace(0.02, 0.98, 200)

    # Build theory curves and titles only for available phenotypes
    all_theory = {}
    all_titles = {}
    if 'A' in params:
        all_theory['CI'] = model_ci_asymmetric(p_th, params['A'], params['alpha'], params['gamma'])
        all_titles['CI'] = 'CI only'
    if 's_0' in params:
        all_theory['ER'] = model_er(p_th, params['s_0'], params['beta'])
        all_titles['ER'] = 'ER only'
    if 'A' in params and 's_0' in params:
        all_theory['CI_ER'] = (model_ci_asymmetric(p_th, params['A'], params['alpha'], params['gamma'])
                               + model_er(p_th, params['s_0'], params['beta']))
        all_titles['CI_ER'] = 'CI + ER'

    # Only plot panels for phenotypes present in data or theory
    available = sorted(set(binned_df['Phenotype'].unique()) | set(all_theory.keys()),
                       key=lambda x: ['CI', 'ER', 'CI_ER'].index(x) if x in ['CI', 'ER', 'CI_ER'] else 99)
    if not available:
        print("    [skip] delta_p_vs_p — no phenotypes to plot")
        return

    n_panels = len(available)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.7 * n_panels, 4.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, pheno in zip(axes, available):
        color = PHENO_STYLE.get(pheno, {'color': COL_GREY})['color']

        # ABM data
        sub = binned_df[binned_df['Phenotype'] == pheno]
        if len(sub) > 0:
            ax.fill_between(sub['p_mid'], sub['dp_q25'], sub['dp_q75'],
                            alpha=0.2, color=color, linewidth=0, label='ABM IQR')
            ax.plot(sub['p_mid'], sub['dp_median'], 'o-', color=color,
                    markersize=4, linewidth=1.5, label='ABM median', zorder=4)

        # Theory
        if pheno in all_theory:
            ax.plot(p_th, all_theory[pheno], '--', color='#333333', linewidth=2.0,
                    alpha=0.7, label='Analytical', zorder=5)

        ax.axhline(0, color=COL_GREY, ls=':', lw=0.7, zorder=0)

        # Unstable equilibrium (zero-crossing) for CI panels
        if pheno in ('CI', 'CI_ER') and pheno in all_theory:
            dp = all_theory[pheno]
            crossings = np.where(np.diff(np.sign(dp)))[0]
            for idx in crossings:
                p_eq = p_th[idx]
                if 0.03 < p_eq < 0.97:
                    ax.axvline(p_eq, color=color, ls=':', lw=1.0, alpha=0.5)
                    ax.text(p_eq + 0.02, 0.0001, f'p*={p_eq:.2f}',
                            fontsize=7, color=color, va='bottom')

        ax.set_xlabel('Infection frequency (p)')
        ax.set_title(all_titles.get(pheno, pheno), fontweight='bold', color=color)
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

    fig.suptitle('Complementary frequency dependence of CI and ER',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, 'delta_p_vs_p'))


# ======================================================================
#  Overlay figure
# ======================================================================

def plot_overlay_figure(binned_df, params, outdir, dt):
    """Single-panel overlay of all three phenotypes with crossing point.

    Marks the frequency where CI overtakes ER as the dominant
    driver of spread ("ER → CI handoff") and shades the ER-dominant
    and CI-dominant frequency regions.

    Args:
        binned_df (pandas.DataFrame): Binned Δp data.
        params (dict): Fitted analytical parameters.
        outdir (str): Output directory.
        dt (int): Time interval used for Δp.
    """
    p_th = np.linspace(0.02, 0.98, 200)

    has_ci = 'A' in params
    has_er = 's_0' in params

    fig, ax = plt.subplots(figsize=(7, 5))

    # Theory curves — only plot what's available
    dp_ci = dp_er = dp_cier = None
    if has_ci:
        dp_ci = model_ci_asymmetric(p_th, params['A'], params['alpha'], params['gamma'])
        ax.plot(p_th, dp_ci, '-', color=COL_CI, linewidth=2.2, label='CI (theory)')
    if has_er:
        dp_er = model_er(p_th, params['s_0'], params['beta'])
        ax.plot(p_th, dp_er, '-', color=COL_ER, linewidth=2.2, label='ER (theory)')
    if has_ci and has_er:
        dp_cier = dp_ci + dp_er
        ax.plot(p_th, dp_cier, '-', color=COL_CIER, linewidth=2.2, label='CI+ER (theory)')

    # ABM scatter — only phenotypes present in data
    for pheno in ['CI', 'ER', 'CI_ER']:
        sub = binned_df[binned_df['Phenotype'] == pheno]
        if len(sub) == 0:
            continue
        style = PHENO_STYLE.get(pheno, {'color': COL_GREY, 'label': pheno})
        ax.scatter(sub['p_mid'], sub['dp_median'], color=style['color'],
                   s=30, alpha=0.6, zorder=3, edgecolors='white', linewidths=0.4,
                   label=f'{style["label"]} (ABM)')

    # Crossing point — only if both CI and ER are present
    # Crossing point — only if both CI and ER are present
    if dp_ci is not None and dp_er is not None:
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

    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, 'delta_p_overlay'))


# ======================================================================
#  CLI
# ======================================================================

def main():
    """CLI entry point for the Δp figure generator.

    Supports four modes via ``--mode``:
        - ``all``: Full trajectory averaging (default).
        - ``ascending``: First-passage, Δp > 0 only.
        - ``initial``: One Δp per replicate (Turelli-correct).
        - ``compare``: Run all three and produce comparison figure.
    """
    parser = argparse.ArgumentParser(
        description="W.I.N.G.S. — Δp vs p figure generator")
    parser.add_argument('--input', required=True,
                        help='Combined CSV from ingest_delta_p.py')
    parser.add_argument('--outdir', default='figures_delta_p',
                        help='Output directory')
    parser.add_argument('--dt', type=int, default=7,
                        help='Δt in days (default: 7)')
    parser.add_argument('--n-bins', type=int, default=20,
                        help='Number of frequency bins (default: 20)')
    parser.add_argument('--ascending-only', action='store_true',
                        help='Only use ascending (Δp > 0) first-passage data')
    # Manual overrides
    parser.add_argument('--A', type=float, default=None, help='CI amplitude')
    parser.add_argument('--alpha', type=float, default=None, help='CI left exponent')
    parser.add_argument('--gamma', type=float, default=None, help='CI right exponent')
    parser.add_argument('--s-0', type=float, default=None, help='ER base advantage')
    parser.add_argument('--beta', type=float, default=None, help='ER decay exponent')
    parser.add_argument('--exclude', default=None,
                        help='Comma-separated Wolbachia mechanics to exclude. '
                             'Valid: CI, MK, ER, IE.  Phenotypes using excluded '
                             'mechanics are removed from all figures.  '
                             'Example: --exclude MK or --exclude MK,IE')
    args = parser.parse_args()

    # Parse exclusion list
    excluded = set()
    if args.exclude:
        excluded = {s.strip().upper() for s in args.exclude.split(",")}
        valid = {"CI", "MK", "ER", "IE"}
        unknown = excluded - valid
        if unknown:
            print(f"  ERROR: Unknown mechanic(s): {unknown}")
            print(f"  Valid options: {', '.join(sorted(valid))}")
            sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 56)
    print("  W.I.N.G.S. — Δp Figure Generator")
    print("=" * 56)
    print(f"  Input:      {args.input}")
    print(f"  Output:     {args.outdir}/")
    print(f"  Δt:         {args.dt} days")
    print(f"  Bins:       {args.n_bins}")
    print(f"  Ascending:  {args.ascending_only}")
    if excluded:
        print(f"  Exclude:    {', '.join(sorted(excluded))}")

    df = pd.read_csv(args.input)
    time_col, inf_col = detect_columns(df)
    print(f"  Columns:    time='{time_col}', infection='{inf_col}'")
    print(f"  Rows:       {len(df):,}")

    # Filter out phenotypes that use excluded mechanics
    if excluded:
        # Map mechanic abbreviations to phenotype label components
        # Phenotype labels in sweep data: CI, ER, CI_ER
        def _pheno_uses_excluded(pheno_label):
            parts = set(pheno_label.split('_'))
            return bool(parts & excluded)
        before = len(df['Phenotype'].unique())
        keep_phenos = [p for p in df['Phenotype'].unique()
                       if not _pheno_uses_excluded(p)]
        df = df[df['Phenotype'].isin(keep_phenos)]
        after = len(keep_phenos)
        print(f"  Filtered:   {before} → {after} phenotypes "
              f"(kept: {', '.join(sorted(keep_phenos)) or 'none'})")
        if len(df) == 0:
            print("\n  ERROR: All phenotypes excluded — nothing to plot!")
            sys.exit(1)

    phenotypes = sorted(df['Phenotype'].unique())
    fractions = sorted(df['Infected Fraction'].unique())
    print(f"  Phenotypes: {', '.join(phenotypes)}")
    print(f"  Fractions:  {len(fractions)} ({min(fractions):.2f}–{max(fractions):.2f})")
    print()

    # --- Extract Δp ---
    print("  Extracting Δp...")
    dp_df = extract_delta_p(df, time_col, inf_col,
                            dt=args.dt, ascending_only=args.ascending_only)
    print(f"  → {len(dp_df):,} (p, Δp) data points")

    if len(dp_df) == 0:
        print("\n  ERROR: No valid Δp data! Check time series have data "
              "after day 23 with 0.01 < p < 0.99")
        sys.exit(1)

    # --- Bin ---
    print("  Binning...")
    binned = bin_delta_p(dp_df, n_bins=args.n_bins)
    for pheno in phenotypes:
        n = len(binned[binned['Phenotype'] == pheno])
        print(f"    {pheno}: {n} bins")

    # --- Fit ---
    print("\n  Fitting analytical models...")
    params = fit_analytical(binned)

    # Overrides
    for key, arg_val in [('A', args.A), ('alpha', args.alpha), ('gamma', args.gamma),
                          ('s_0', args.s_0), ('beta', args.beta)]:
        if arg_val is not None:
            params[key] = arg_val
            print(f"  [override] {key} = {arg_val}")

    # --- Save intermediates ---
    dp_df.to_csv(os.path.join(args.outdir, 'raw_delta_p.csv'), index=False)
    binned.to_csv(os.path.join(args.outdir, 'binned_delta_p.csv'), index=False)
    with open(os.path.join(args.outdir, 'fitted_params.txt'), 'w') as f:
        for k, v in sorted(params.items()):
            f.write(f"{k} = {v}\n")
    print(f"\n  Saved: raw_delta_p.csv, binned_delta_p.csv, fitted_params.txt")

    # --- Plot ---
    print("\n  Generating figures (PNG + SVG)...")
    plot_delta_p_figure(binned, params, args.outdir, args.dt)
    plot_overlay_figure(binned, params, args.outdir, args.dt)

    print(f"\n  Done — figures in {args.outdir}/")
    print("=" * 56)


if __name__ == '__main__':
    main()

#Example usage: python -m wings.analysis.plot_delta_p --input data/combined_delta_p.csv --dt 24 