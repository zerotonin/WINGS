#!/usr/bin/env python3
"""
W.I.N.G.S. — Publication figure generator.

Produces parallel figure sets for the Agent-Based Model (ABM) and
the Wright-Fisher fixed-size Model (WFM), organised by biologically
meaningful combination subsets.

Colour scheme: Paul Tol's qualitative palette (colourblind-safe).
Line plots differentiate combos via colour + dash pattern + line width.
All figures exported as both PNG (300 dpi) and SVG (text as text objects).

Figure catalogue
----------------
For EACH model (ABM / WFM):
  1–2.  infection_over_time_{subset}.{png,svg}
  3–4.  final_infection_{subset}.{png,svg}
  5–6.  time_to_fixation_{subset}.{png,svg}
  7.    heatmap_infection.{png,svg}
  8.    heatmap_fixation_pct.{png,svg}
  9.    heatmap_fixation_time.{png,svg}
  ABM-only:
  10–11. population_over_time_{subset}.{png,svg}
  12.    heatmap_population.{png,svg}

Usage
-----
  python plot_wings.py --model abm --input data/combined_abm.csv
  python plot_wings.py --model wfm --input data/combined_wfm.csv
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ======================================================================
#  Style & constants
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
    "legend.fontsize": 8,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.8",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    # SVG: keep text as text objects (editable in Illustrator)
    "svg.fonttype": "none",
})

FIXATION_THRESHOLD = 0.99
EGG_HATCH_DAY = 23  # Tribolium egg hatching ≈ 552 hours ≈ 23 days


# ======================================================================
#  Paul Tol qualitative palette (colourblind-safe)
# ======================================================================
# Source: https://personal.sron.nl/~pault/data/colourschemes.pdf
#
# We use the "bright" scheme (7 colours) extended with the "vibrant"
# scheme for additional combos.  Each combo also gets a unique dash
# pattern and line width for redundant coding.
#
# The user's biological data convention:
#   skyblue = uninfected,  bright orange = infected,  dark green = treated
# We keep these compatible: "None" (no Wolbachia effect) gets grey,
# infection-driving combos use warm/cool tones.

# -- Tol bright --
_TOL_BLUE    = "#4477AA"
_TOL_CYAN    = "#66CCEE"
_TOL_GREEN   = "#228833"
_TOL_YELLOW  = "#CCBB44"
_TOL_RED     = "#EE6677"
_TOL_PURPLE  = "#AA3377"
_TOL_GREY    = "#BBBBBB"

# -- Tol vibrant (supplements) --
_TOL_ORANGE  = "#EE7733"
_TOL_TEAL    = "#009988"
_TOL_MAGENTA = "#EE3377"
_TOL_DBLUE   = "#0077BB"

# Combo → (colour, dash, linewidth)
# Dash patterns: solid, dashed, dotted, dash-dot, long-dash, etc.
_SOLID   = "solid"
_DASH    = (0, (5, 2))
_DOT     = (0, (1.5, 1.5))
_DASHDOT = (0, (5, 2, 1.5, 2))
_LDASH   = (0, (8, 3))
_LDASHDOT = (0, (8, 3, 1.5, 3))
_DDASH   = (0, (3, 1.5))
_DDASHDOT = (0, (3, 1.5, 1.5, 1.5))

COMBO_STYLE = {
    # --- Subset A: Individual effects ---
    "None":          (_TOL_GREY,    _SOLID,   1.6),
    "CI":            (_TOL_BLUE,    _SOLID,   2.2),
    "MK":            (_TOL_RED,     _SOLID,   2.0),
    "ER":            (_TOL_GREEN,   _SOLID,   2.0),
    "IE":            (_TOL_PURPLE,  _SOLID,   2.0),
    "CI+MK+ER+IE":  ("#222222",    _SOLID,   2.4),  # near-black for "all"

    # --- Subset B: ER-centric ---
    "ER+IE":         (_TOL_YELLOW,  _DASH,    2.0),
    "MK+ER":         (_TOL_ORANGE,  _DASH,    2.0),
    "CI+ER":         (_TOL_TEAL,    _DASH,    2.2),
    "MK+ER+IE":      (_TOL_MAGENTA, _DASHDOT, 2.0),
    "CI+ER+IE":      (_TOL_CYAN,    _DASHDOT, 2.0),

    # --- Remaining combos (for heatmap annotations, etc.) ---
    "CI+IE":         (_TOL_DBLUE,   _DOT,     1.8),
    "CI+MK":         (_TOL_ORANGE,  _DOT,     1.8),
    "MK+IE":         (_TOL_RED,     _DDASH,   1.8),
    "CI+MK+ER":      (_TOL_TEAL,    _LDASH,   1.8),
    "CI+MK+IE":      (_TOL_YELLOW,  _DDASHDOT,1.8),
}


def get_style(label):
    """Return (colour, dash, linewidth) for a combo label."""
    return COMBO_STYLE.get(label, (_TOL_GREY, _SOLID, 1.5))


# -- Combo subsets --
SUBSET_A_NAME = "Individual Effects"
SUBSET_A = [
    (False, False, False, False),  # None
    (False, False, True,  False),  # ER
    (False, False, False, True),   # IE
    (False, True,  False, False),  # MK
    (True,  False, False, False),  # CI
    (True,  True,  True,  True),   # CI+MK+ER+IE
]

SUBSET_B_NAME = "ER-Centric Combinations"
SUBSET_B = [
    (False, False, False, False),  # None
    (False, False, True,  False),  # ER
    (False, False, True,  True),   # ER+IE
    (False, True,  True,  False),  # MK+ER
    (True,  False, True,  False),  # CI+ER
    (False, True,  True,  True),   # MK+ER+IE
    (True,  False, True,  True),   # CI+ER+IE
    (True,  True,  True,  True),   # CI+MK+ER+IE
]


# ======================================================================
#  Helpers
# ======================================================================

def combo_label(ci, mk, er, ie):
    parts = []
    if ci: parts.append("CI")
    if mk: parts.append("MK")
    if er: parts.append("ER")
    if ie: parts.append("IE")
    return "+".join(parts) if parts else "None"


def save_fig(fig, path_stem):
    """Save figure as both PNG and SVG."""
    fig.savefig(f"{path_stem}.png")
    fig.savefig(f"{path_stem}.svg")
    plt.close(fig)
    print(f"    ✓ {Path(path_stem).name}.{{png,svg}}")


def load_data(path):
    """Load combined simulation CSV produced by ingest_data.py."""
    df = pd.read_csv(path)
    for col in ["Cytoplasmic Incompatibility", "Male Killing",
                "Increased Exploration Rate", "Increased Eggs"]:
        df[col] = df[col].astype(str).str.strip().str.lower() == "true"
    df["Combo"] = df.apply(
        lambda r: combo_label(
            r["Cytoplasmic Incompatibility"],
            r["Male Killing"],
            r["Increased Exploration Rate"],
            r["Increased Eggs"],
        ),
        axis=1,
    )
    return df


def filter_combos(df, subset):
    masks = []
    for ci, mk, er, ie in subset:
        m = (
            (df["Cytoplasmic Incompatibility"] == ci)
            & (df["Male Killing"] == mk)
            & (df["Increased Exploration Rate"] == er)
            & (df["Increased Eggs"] == ie)
        )
        masks.append(m)
    return df[pd.concat(masks, axis=1).any(axis=1)].copy()


def get_ordered_labels(subset):
    return [combo_label(ci, mk, er, ie) for ci, mk, er, ie in subset]


# ======================================================================
#  Summary statistics
# ======================================================================

def compute_timeseries_stats(df, time_col="Day"):
    group_cols = ["Combo", time_col]
    stats = df.groupby(group_cols).agg(
        inf_median=("Infection Rate", "median"),
        inf_q25=("Infection Rate", lambda x: x.quantile(0.25)),
        inf_q75=("Infection Rate", lambda x: x.quantile(0.75)),
        inf_q05=("Infection Rate", lambda x: x.quantile(0.05)),
        inf_q95=("Infection Rate", lambda x: x.quantile(0.95)),
        pop_median=("Population Size", "median"),
        pop_q25=("Population Size", lambda x: x.quantile(0.25)),
        pop_q75=("Population Size", lambda x: x.quantile(0.75)),
    ).reset_index()
    return stats


def compute_final_values(df, time_col="Day"):
    idx = df.groupby(["Combo", "Replicate ID"])[time_col].idxmax()
    return df.loc[idx].copy()


def compute_time_to_fixation(df, time_col="Day", threshold=FIXATION_THRESHOLD):
    records = []
    for (combo, rep), grp in df.groupby(["Combo", "Replicate ID"]):
        above = grp[grp["Infection Rate"] >= threshold]
        t_fix = above[time_col].min() if len(above) > 0 else np.nan
        records.append({"Combo": combo, "Replicate ID": rep, "t_fix": t_fix})
    return pd.DataFrame(records)


# ======================================================================
#  Plot: time series with ribbons
# ======================================================================

def plot_timeseries(
    df, subset, subset_name, metric, ylabel, title,
    path_stem, time_col="Day", is_abm=True, skip_before=None
):
    sub = filter_combos(df, subset)
    labels = get_ordered_labels(subset)
    stats = compute_timeseries_stats(sub, time_col)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for label in labels:
        s = stats[stats["Combo"] == label].sort_values(time_col)
        if s.empty:
            continue
        t = s[time_col].values
        color, dash, lw = get_style(label)

        if skip_before is not None:
            mask = t >= skip_before
            t = t[mask]
            s = s.iloc[mask]
            if len(t) == 0:
                continue

        med = s[f"{metric}_median"].values
        q25 = s[f"{metric}_q25"].values
        q75 = s[f"{metric}_q75"].values

        # Light ribbon: 5th–95th (infection only)
        if f"{metric}_q05" in s.columns:
            q05 = s[f"{metric}_q05"].values
            q95 = s[f"{metric}_q95"].values
            ax.fill_between(t, q05, q95, alpha=0.07, color=color, linewidth=0)

        # IQR ribbon
        ax.fill_between(t, q25, q75, alpha=0.18, color=color, linewidth=0)
        # Median line with dash pattern
        ax.plot(t, med, color=color, linewidth=lw, linestyle=dash, label=label)

    # Semilog x for ABM only
    if is_abm:
        ax.set_xscale("symlog", linthresh=10)
        ax.xaxis.set_major_locator(mticker.FixedLocator(
            [1, 5, 10, 25, 50, 100, 200, 365]
        ))
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlim(left=max(1, skip_before or 1))
    else:
        ax.set_xlim(0, df[time_col].max())
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Egg hatch marker (ABM only)
    if is_abm and skip_before is None:
        ax.axvline(EGG_HATCH_DAY, color="#cccccc", ls="--", lw=0.8, zorder=0)
        ax.text(
            EGG_HATCH_DAY + 1, ax.get_ylim()[1] * 0.02,
            "eggs hatch", fontsize=7, color="#999999", va="bottom",
        )

    ax.set_xlabel("Generation" if not is_abm else "Day")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=10)

    # Legend with line styles shown
    leg = ax.legend(loc="best", ncol=2 if len(labels) > 4 else 1,
                    handlelength=3.0)

    if metric == "inf":
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(1.0, color="#eeeeee", ls=":", lw=0.7, zorder=0)

    fig.tight_layout()
    save_fig(fig, path_stem)


# ======================================================================
#  Plot: final infection — strip + bar hybrid
# ======================================================================

def plot_final_infection(
    df, subset, subset_name, title, path_stem, time_col="Day"
):
    """
    Strip plot of final infection rate with fixation-% annotation.

    Solves the problem of boxplots collapsing when most values = 1.0.
    Each replicate is a jittered dot; a horizontal bar shows the median;
    annotation gives the fixation percentage.
    """
    sub = filter_combos(df, subset)
    labels = get_ordered_labels(subset)
    finals = compute_final_values(sub, time_col)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rng = np.random.default_rng(42)

    for i, label in enumerate(labels):
        vals = finals.loc[finals["Combo"] == label, "Infection Rate"].values
        if len(vals) == 0:
            continue

        color, _, _ = get_style(label)

        # Jittered strip
        jitter = rng.uniform(-0.25, 0.25, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            color=color, s=10, alpha=0.35, zorder=3, edgecolors="none",
        )

        # Median diamond
        med = np.median(vals)
        ax.scatter(
            [i], [med], color=color, s=70, zorder=5,
            marker="D", edgecolors="white", linewidths=0.8,
        )

        # IQR bar
        q25, q75 = np.percentile(vals, [25, 75])
        ax.plot([i, i], [q25, q75], color=color, lw=2.5, solid_capstyle="round",
                zorder=4, alpha=0.7)

        # Fixation % annotation above
        n_fixed = (vals >= FIXATION_THRESHOLD).sum()
        pct = 100 * n_fixed / len(vals)
        ax.text(
            i, 1.07, f"{pct:.0f}%",
            ha="center", va="bottom", fontsize=7.5, color=color,
            fontweight="bold",
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Final Infection Rate")
    ax.set_ylim(-0.05, 1.18)
    ax.axhline(1.0, color="#eeeeee", ls=":", lw=0.7, zorder=0)

    # Add subtle "% fixed" header
    ax.text(
        0.5, 1.14, "% reaching fixation",
        ha="center", va="bottom", fontsize=7, color="#888888",
        style="italic", transform=ax.get_xaxis_transform(),
    )

    ax.set_title(title, fontweight="bold", pad=10)
    fig.tight_layout()
    save_fig(fig, path_stem)

    # --- CSV export ---
    csv_rows = []
    for label in labels:
        vals = finals.loc[finals["Combo"] == label, "Infection Rate"].values
        for v in vals:
            csv_rows.append({"mechanic": label, "final_infection_rate": v})
    pd.DataFrame(csv_rows).to_csv(f"{path_stem}.csv", index=False)
    print(f"    ✓ {Path(path_stem).name}.csv")


# ======================================================================
#  Plot: time to fixation — violin + strip
# ======================================================================

def plot_time_to_fixation(
    df, subset, subset_name, title, path_stem,
    time_col="Day", is_abm=True
):
    sub = filter_combos(df, subset)
    labels = get_ordered_labels(subset)
    ttf = compute_time_to_fixation(sub, time_col)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rng = np.random.default_rng(42)

    positions_used = []
    labels_used = []
    never_fixed = []

    pos = 0
    for label in labels:
        group = ttf[ttf["Combo"] == label]
        vals = group["t_fix"].dropna().values
        total = len(group)
        n_fixed = len(vals)
        color, _, _ = get_style(label)

        if n_fixed == 0:
            never_fixed.append(label)
            continue

        positions_used.append(pos)
        labels_used.append(label)

        # Violin (if enough data)
        if n_fixed >= 3:
            parts = ax.violinplot(vals, positions=[pos], showextrema=False, widths=0.6)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.35)
                pc.set_edgecolor(color)
                pc.set_linewidth(0.5)

        # Median marker
        ax.scatter([pos], [np.median(vals)], color=color,
                   s=50, zorder=5, marker="D",
                   edgecolors="white", linewidths=0.6)

        # Jittered strip
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter, vals,
            color=color, s=6, alpha=0.3, zorder=4, edgecolors="none",
        )

        # Annotate n_fixed / total
        if n_fixed < total:
            ax.text(
                pos, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else vals.min() - 0.5,
                f"{n_fixed}/{total}",
                ha="center", va="top", fontsize=6.5, color=color,
            )

        pos += 1

    if never_fixed:
        note = "No fixation: " + ", ".join(never_fixed)
        ax.text(
            0.02, 0.98, note, transform=ax.transAxes,
            fontsize=7.5, va="top", color="#888888", style="italic",
        )

    ax.set_xticks(positions_used)
    ax.set_xticklabels(labels_used, rotation=30, ha="right")
    ax.set_ylabel("Generation" if not is_abm else "Day")
    ax.set_title(title, fontweight="bold", pad=10)

    fig.tight_layout()
    save_fig(fig, path_stem)

    # --- CSV export (all replicates, NaN if never fixed) ---
    csv_rows = []
    for label in labels:
        group = ttf[ttf["Combo"] == label]
        for _, row in group.iterrows():
            csv_rows.append({
                "mechanic": label,
                "time_to_fixation": row["t_fix"],
            })
    pd.DataFrame(csv_rows).to_csv(f"{path_stem}.csv", index=False)
    print(f"    ✓ {Path(path_stem).name}.csv")


# ======================================================================
#  Plot: heatmaps (all 16 combos)
# ======================================================================

def plot_heatmap(df, metric_func, cmap, cbar_label, title, path_stem,
                 time_col="Day", fmt=".2f", vmin=None, vmax=None,
                 csv_raw_func=None, csv_value_col="value"):
    """
    4×4 heatmap:
      rows:    —, MK, CI, CI+MK    (CI/MK severity axis)
      columns: —, IE, ER, ER+IE    (exploration/fecundity axis)

    If csv_raw_func is provided, exports a CSV with per-replicate raw
    values (mechanic, replicate, value) for statistical analysis.
    csv_raw_func(df_sub, time_col) should return a list of dicts with
    keys 'Replicate ID' and the value column.
    """
    row_configs = [
        (False, False, "—"),
        (False, True,  "MK"),
        (True,  False, "CI"),
        (True,  True,  "CI+MK"),
    ]
    col_configs = [
        (False, False, "—"),
        (False, True,  "IE"),
        (True,  False, "ER"),
        (True,  True,  "ER+IE"),
    ]

    matrix = np.full((4, 4), np.nan)
    annot = np.empty((4, 4), dtype=object)

    for ri, (r_ci, r_mk, _) in enumerate(row_configs):
        for ci_col, (c_er, c_ie, _) in enumerate(col_configs):
            sub = df[
                (df["Cytoplasmic Incompatibility"] == r_ci)
                & (df["Male Killing"] == r_mk)
                & (df["Increased Exploration Rate"] == c_er)
                & (df["Increased Eggs"] == c_ie)
            ]
            if len(sub) == 0:
                annot[ri, ci_col] = "—"
                continue
            val = metric_func(sub, time_col)
            matrix[ri, ci_col] = val
            annot[ri, ci_col] = "—" if np.isnan(val) else f"{val:{fmt}}"

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Use explicit vmin/vmax so we can judge text colour
    v0 = vmin if vmin is not None else np.nanmin(matrix)
    v1 = vmax if vmax is not None else np.nanmax(matrix)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=v0, vmax=v1)

    for ri in range(4):
        for ci_col in range(4):
            val = matrix[ri, ci_col]
            # Text colour: white on dark cells, black on light
            if np.isnan(val):
                tc = "#999999"
            else:
                norm_val = (val - v0) / (v1 - v0 + 1e-9)
                tc = "white" if norm_val > 0.55 else "black"
            ax.text(
                ci_col, ri, annot[ri, ci_col],
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=tc,
            )

    ax.set_xticks(range(4))
    ax.set_xticklabels([c[2] for c in col_configs])
    ax.set_yticks(range(4))
    ax.set_yticklabels([r[2] for r in row_configs])
    ax.set_xlabel("Exploration / Fecundity axis", fontsize=10)
    ax.set_ylabel("CI / MK axis", fontsize=10)
    ax.set_title(title, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.08)
    cbar.set_label(cbar_label, fontsize=9)

    fig.tight_layout()
    save_fig(fig, path_stem)

    # --- CSV export: per-replicate raw data for all 16 combos ---
    if csv_raw_func is not None:
        csv_rows = []
        for ri, (r_ci, r_mk, _) in enumerate(row_configs):
            for ci_col, (c_er, c_ie, _) in enumerate(col_configs):
                sub = df[
                    (df["Cytoplasmic Incompatibility"] == r_ci)
                    & (df["Male Killing"] == r_mk)
                    & (df["Increased Exploration Rate"] == c_er)
                    & (df["Increased Eggs"] == c_ie)
                ]
                if len(sub) == 0:
                    continue
                label = combo_label(r_ci, r_mk, c_er, c_ie)
                raw = csv_raw_func(sub, time_col)
                for rec in raw:
                    rec["mechanic"] = label
                csv_rows.extend(raw)
        out_df = pd.DataFrame(csv_rows)
        # Reorder so mechanic is first
        cols = ["mechanic"] + [c for c in out_df.columns if c != "mechanic"]
        out_df[cols].to_csv(f"{path_stem}.csv", index=False)
        print(f"    ✓ {Path(path_stem).name}.csv")


# -- Heatmap metric functions (summary for cell values) --

def metric_median_final_infection(df_sub, time_col):
    idx = df_sub.groupby("Replicate ID")[time_col].idxmax()
    return df_sub.loc[idx, "Infection Rate"].median()


def metric_fixation_pct(df_sub, time_col):
    n_total = df_sub["Replicate ID"].nunique()
    if n_total == 0:
        return np.nan
    n_fixed = sum(
        grp["Infection Rate"].max() >= FIXATION_THRESHOLD
        for _, grp in df_sub.groupby("Replicate ID")
    )
    return 100.0 * n_fixed / n_total


def metric_median_time_to_fixation(df_sub, time_col):
    times = []
    for _, grp in df_sub.groupby("Replicate ID"):
        above = grp[grp["Infection Rate"] >= FIXATION_THRESHOLD]
        if len(above) > 0:
            times.append(above[time_col].min())
    if len(times) < df_sub["Replicate ID"].nunique() * 0.1:
        return np.nan
    return np.median(times)


# -- Per-replicate raw extraction functions (for CSV export) --

def raw_final_infection(df_sub, time_col):
    """One row per replicate: final infection rate."""
    idx = df_sub.groupby("Replicate ID")[time_col].idxmax()
    rows = df_sub.loc[idx]
    return [
        {"replicate": int(r["Replicate ID"]),
         "final_infection_rate": r["Infection Rate"]}
        for _, r in rows.iterrows()
    ]


def raw_fixation_binary(df_sub, time_col):
    """One row per replicate: 1 if reached fixation, 0 otherwise."""
    records = []
    for rep, grp in df_sub.groupby("Replicate ID"):
        fixed = int(grp["Infection Rate"].max() >= FIXATION_THRESHOLD)
        records.append({"replicate": int(rep), "reached_fixation": fixed})
    return records


def raw_time_to_fixation(df_sub, time_col):
    """One row per replicate: time to fixation (NaN if never)."""
    records = []
    for rep, grp in df_sub.groupby("Replicate ID"):
        above = grp[grp["Infection Rate"] >= FIXATION_THRESHOLD]
        t = above[time_col].min() if len(above) > 0 else np.nan
        records.append({"replicate": int(rep), "time_to_fixation": t})
    return records


def raw_final_population(df_sub, time_col):
    """One row per replicate: final population size."""
    idx = df_sub.groupby("Replicate ID")[time_col].idxmax()
    rows = df_sub.loc[idx]
    return [
        {"replicate": int(r["Replicate ID"]),
         "final_population_size": r["Population Size"]}
        for _, r in rows.iterrows()
    ]


def metric_median_final_population(df_sub, time_col):
    idx = df_sub.groupby("Replicate ID")[time_col].idxmax()
    return df_sub.loc[idx, "Population Size"].median()


# ======================================================================
#  Main figure generation pipeline
# ======================================================================

def generate_figures(df, model, outdir, time_col="Day"):
    is_abm = (model == "abm")
    os.makedirs(outdir, exist_ok=True)

    model_upper = model.upper()
    time_label = "Day" if is_abm else "Generation"
    skip = EGG_HATCH_DAY if is_abm else None

    print(f"\n  Generating {model_upper} figures → {outdir}/")
    print(f"  {'—' * 48}")

    # 1–2: Infection over time
    for subset, sname, tag in [
        (SUBSET_A, SUBSET_A_NAME, "individual"),
        (SUBSET_B, SUBSET_B_NAME, "er_centric"),
    ]:
        plot_timeseries(
            df, subset, sname, metric="inf",
            ylabel="Infection Rate",
            title=f"{model_upper} — Infection Rate ({sname})",
            path_stem=os.path.join(outdir, f"infection_over_time_{tag}"),
            time_col=time_col, is_abm=is_abm, skip_before=skip,
        )

    # 3–4: Final infection (strip + bar hybrid)
    for subset, sname, tag in [
        (SUBSET_A, SUBSET_A_NAME, "individual"),
        (SUBSET_B, SUBSET_B_NAME, "er_centric"),
    ]:
        plot_final_infection(
            df, subset, sname,
            title=f"{model_upper} — Final Infection Rate ({sname})",
            path_stem=os.path.join(outdir, f"final_infection_{tag}"),
            time_col=time_col,
        )

    # 5–6: Time to fixation
    for subset, sname, tag in [
        (SUBSET_A, SUBSET_A_NAME, "individual"),
        (SUBSET_B, SUBSET_B_NAME, "er_centric"),
    ]:
        plot_time_to_fixation(
            df, subset, sname,
            title=f"{model_upper} — Time to Fixation ({sname})",
            path_stem=os.path.join(outdir, f"time_to_fixation_{tag}"),
            time_col=time_col, is_abm=is_abm,
        )

    # 7: Heatmap — median final infection
    plot_heatmap(
        df, metric_median_final_infection,
        cmap="YlOrRd", cbar_label="Median Final Infection Rate",
        title=f"{model_upper} — Final Infection Rate (all combos)",
        path_stem=os.path.join(outdir, "heatmap_infection"),
        time_col=time_col, fmt=".2f", vmin=0, vmax=1,
        csv_raw_func=raw_final_infection,
    )

    # 8: Heatmap — fixation percentage
    plot_heatmap(
        df, metric_fixation_pct,
        cmap="YlGnBu", cbar_label="Replicates Reaching Fixation (%)",
        title=f"{model_upper} — Fixation Success (all combos)",
        path_stem=os.path.join(outdir, "heatmap_fixation_pct"),
        time_col=time_col, fmt=".0f", vmin=0, vmax=100,
        csv_raw_func=raw_fixation_binary,
    )

    # 9: Heatmap — median time to fixation
    max_time = df[time_col].max()
    plot_heatmap(
        df, metric_median_time_to_fixation,
        cmap="YlGnBu_r", cbar_label=f"Median {time_label} to Fixation",
        title=f"{model_upper} — Speed of Fixation (all combos)",
        path_stem=os.path.join(outdir, "heatmap_fixation_time"),
        time_col=time_col, fmt=".0f", vmin=0, vmax=max_time,
        csv_raw_func=raw_time_to_fixation,
    )

    # ABM-only: population over time
    if is_abm:
        for subset, sname, tag in [
            (SUBSET_A, SUBSET_A_NAME, "individual"),
            (SUBSET_B, SUBSET_B_NAME, "er_centric"),
        ]:
            plot_timeseries(
                df, subset, sname, metric="pop",
                ylabel="Adult Population Size",
                title=f"{model_upper} — Population Size ({sname})",
                path_stem=os.path.join(outdir, f"population_over_time_{tag}"),
                time_col=time_col, is_abm=True, skip_before=skip,
            )

        # ABM: heatmap of final population
        plot_heatmap(
            df, metric_median_final_population,
            cmap="viridis", cbar_label="Median Final Adult Population",
            title=f"{model_upper} — Final Population Size (all combos)",
            path_stem=os.path.join(outdir, "heatmap_population"),
            time_col=time_col, fmt=".0f",
            csv_raw_func=raw_final_population,
        )


# ======================================================================
#  CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="W.I.N.G.S. — Publication figure generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_wings.py --model abm --input data/combined_abm.csv
  python plot_wings.py --model wfm --input data/combined_wfm.csv
  python plot_wings.py --model wfm --input data.csv --outdir ./my_figs
        """,
    )
    parser.add_argument("--model", required=True, choices=["abm", "wfm"])
    parser.add_argument("--input", required=True, help="Combined CSV from ingest_data.py")
    parser.add_argument("--outdir", default=None, help="Output directory (default: figures_{model}/)")
    args = parser.parse_args()

    outdir = args.outdir or f"figures_{args.model}"

    print("=" * 56)
    print("  W.I.N.G.S. — Figure Generator")
    print("=" * 56)
    print(f"  Model:  {args.model.upper()}")
    print(f"  Input:  {args.input}")
    print(f"  Output: {outdir}/")

    df = load_data(args.input)
    n_combos = df["Combo"].nunique()
    n_reps = df["Replicate ID"].nunique()
    n_time = df["Day"].max()

    print(f"  Combos: {n_combos}  |  Reps: {n_reps}  |  Max time: {n_time}")
    print("=" * 56)

    generate_figures(df, args.model, outdir, time_col="Day")

    n_png = len([f for f in os.listdir(outdir) if f.endswith(".png")])
    n_csv = len([f for f in os.listdir(outdir) if f.endswith(".csv")])
    print(f"\n  Done — {n_png} PNG + {n_png} SVG + {n_csv} CSV saved to {outdir}/")


if __name__ == "__main__":
    main()