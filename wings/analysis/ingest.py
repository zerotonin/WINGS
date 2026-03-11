"""
W.I.N.G.S. — Ingest GPU simulation results
============================================
Reads all CSV files from the GPU results directory, parses the
experimental conditions from the filenames, and produces a single
combined CSV ready for plot_results.py.

Filename format (matches both old CPU and new GPU outputs):
  cytoplasmic_incompatibility_<True|False>_male_killing_<True|False>_
  increased_exploration_rate_<True|False>_increased_eggs_<True|False>_<replicate>.csv

CSV columns expected: Population Size, Infection Rate
  (Day is added automatically as row index + 1)

Output columns (what plot_results.py expects):
  Day, Population Size, Infection Rate,
  Cytoplasmic Incompatibility, Male Killing,
  Increased Exploration Rate, Increased Eggs, Replicate ID
"""

import os
import re
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ── Filename parser ──────────────────────────────────────────────
FILENAME_PATTERN = re.compile(
    r'cytoplasmic_incompatibility_(True|False)_'
    r'male_killing_(True|False)_'
    r'increased_exploration_rate_(True|False)_'
    r'increased_eggs_(True|False)_'
    r'(\d+)\.csv$'
)

def parse_conditions_from_filename(filename: str):
    """
    Extract experimental conditions and replicate ID from filename.

    Returns
    -------
    tuple : (ci: bool, mk: bool, er: bool, eg: bool, rep_id: int)
    """
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None  # not a simulation file — skip silently
    ci, mk, er, eg, rep_id = match.groups()
    return (
        ci == 'True',
        mk == 'True',
        er == 'True',
        eg == 'True',
        int(rep_id),
    )


# ── Column name normalisation ────────────────────────────────────
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has 'Population Size' and 'Infection Rate'
    columns regardless of what the CSV header actually says.
    Handles the GPU output (already correct) and any minor variations.
    """
    col_map = {}
    for col in df.columns:
        low = col.strip().lower().replace('_', ' ')
        if 'pop' in low and 'size' in low:
            col_map[col] = 'Population Size'
        elif 'pop' in low and 'size' not in low:
            col_map[col] = 'Population Size'
        elif 'infect' in low and 'rate' in low:
            col_map[col] = 'Infection Rate'
        elif 'infect' in low:
            col_map[col] = 'Infection Rate'
    if col_map:
        df = df.rename(columns=col_map)
    return df


# ── Main ─────────────────────────────────────────────────────────
def main(data_dir: str, output_file: str):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: Directory not found: {data_dir}")
        sys.exit(1)

    # Gather all CSV files (non-recursive)
    all_csv = sorted(data_dir.glob("*.csv"))
    if not all_csv:
        print(f"ERROR: No CSV files found in {data_dir}")
        sys.exit(1)

    combined = []
    skipped = 0
    errors = 0

    for fpath in tqdm(all_csv, desc="Reading simulation outputs"):
        result = parse_conditions_from_filename(fpath.name)
        if result is None:
            skipped += 1
            continue

        ci, mk, er, eg, rep_id = result

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"  WARNING: Could not read {fpath.name}: {e}")
            errors += 1
            continue

        df = normalise_columns(df)

        # Validate required columns exist
        missing = [c for c in ('Population Size', 'Infection Rate') if c not in df.columns]
        if missing:
            print(f"  WARNING: {fpath.name} missing columns {missing} — skipping")
            errors += 1
            continue

        # Drop any rows that are all-NaN (safety)
        df = df.dropna(subset=['Population Size', 'Infection Rate'])

        if len(df) == 0:
            print(f"  WARNING: {fpath.name} has no data rows — skipping")
            errors += 1
            continue

        # Add Day column (1-indexed)
        if 'Day' not in df.columns:
            df.insert(0, 'Day', range(1, len(df) + 1))

        # Add condition columns (names must match plot_results.py)
        df['Cytoplasmic Incompatibility'] = ci
        df['Male Killing'] = mk
        df['Increased Exploration Rate'] = er
        df['Increased Eggs'] = eg
        df['Replicate ID'] = rep_id

        combined.append(df)

    if not combined:
        print("ERROR: No valid simulation files found.")
        sys.exit(1)

    combined_df = pd.concat(combined, ignore_index=True)

    # Keep only the columns plot_results.py needs, in the right order
    output_cols = [
        'Day', 'Population Size', 'Infection Rate',
        'Cytoplasmic Incompatibility', 'Male Killing',
        'Increased Exploration Rate', 'Increased Eggs',
        'Replicate ID',
    ]
    combined_df = combined_df[output_cols]

    # Summary
    n_files = len(combined)
    n_combos = combined_df.groupby([
        'Cytoplasmic Incompatibility', 'Male Killing',
        'Increased Exploration Rate', 'Increased Eggs',
    ]).ngroups
    n_reps = combined_df['Replicate ID'].nunique()
    n_days = combined_df['Day'].max()

    print(f"\n{'='*55}")
    print(f"  Ingestion complete")
    print(f"{'='*55}")
    print(f"  Files read:   {n_files}")
    print(f"  Skipped:      {skipped} (non-matching filenames)")
    print(f"  Errors:       {errors}")
    print(f"  Combos:       {n_combos}")
    print(f"  Replicates:   {n_reps} unique IDs")
    print(f"  Days:         {n_days}")
    print(f"  Total rows:   {len(combined_df):,}")
    print(f"  Output:       {output_file}")
    print(f"{'='*55}")

    # Per-combo summary
    print(f"\n  Replicates per combo:")
    for (ci, mk, er, eg), grp in combined_df.groupby([
        'Cytoplasmic Incompatibility', 'Male Killing',
        'Increased Exploration Rate', 'Increased Eggs',
    ]):
        reps = grp['Replicate ID'].nunique()
        label = []
        if ci: label.append('CI')
        if mk: label.append('MK')
        if er: label.append('ER')
        if eg: label.append('IE')
        name = '+'.join(label) if label else 'None'
        print(f"    {name:40s}  {reps:3d} replicates")

    # Save
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    combined_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest WINGS simulation CSVs")
    parser.add_argument('--input-dir', type=str,
                        default="data/compare_spread_features",
                        help="Directory containing simulation CSV files")
    parser.add_argument('--output', type=str,
                        default="./data/combined_simulation_data.csv",
                        help="Output combined CSV path")
    args = parser.parse_args()
    main(args.input_dir, args.output)