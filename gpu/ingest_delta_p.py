#!/usr/bin/env python3
"""
W.I.N.G.S. — Δp sweep data ingestion.

Reads individual ABM CSV outputs from the delta-p sweep and combines
them into a single CSV for plotting.

Filename convention:
    {PHENOTYPE}_frac{NNN}_rep{R}.csv
    e.g. CI_frac050_rep3.csv  → phenotype=CI, fraction=0.50, rep=3

Each per-run CSV has columns: Day, Population Size, Infection Rate, ...

Output CSV adds: Phenotype, Infected Fraction, Replicate ID

Usage:
    python ingest_delta_p.py \
        --input-dir /projects/.../abm_delta_p \
        --output data/combined_delta_p.csv
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

# Filename pattern: PHENOTYPE_fracNNN_repR.csv
FNAME_RE = re.compile(
    r'^(?P<pheno>[A-Z_]+)_frac(?P<frac>\d{3})_rep(?P<rep>\d+)\.csv$'
)


def parse_filename(fname):
    """Extract (phenotype, fraction, replicate) from filename."""
    m = FNAME_RE.match(fname)
    if not m:
        return None
    pheno = m.group('pheno')
    frac = int(m.group('frac')) / 100.0  # 050 → 0.50
    rep = int(m.group('rep'))
    return pheno, frac, rep


def main():
    parser = argparse.ArgumentParser(
        description="W.I.N.G.S. — Δp sweep data ingestion"
    )
    parser.add_argument(
        '--input-dir', required=True,
        help='Directory containing per-run CSV files'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output combined CSV path'
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    csv_files = sorted(input_dir.glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files in {input_dir}")

    frames = []
    skipped = 0

    for fpath in csv_files:
        parsed = parse_filename(fpath.name)
        if parsed is None:
            skipped += 1
            continue

        pheno, frac, rep = parsed

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"  [skip] {fpath.name}: {e}")
            skipped += 1
            continue

        if df.empty or 'Infection Rate' not in df.columns:
            skipped += 1
            continue

        # Raw CSVs have no time column — row index is the day
        if 'Day' not in df.columns:
            df.insert(0, 'Day', range(len(df)))

        df['Phenotype'] = pheno
        df['Infected Fraction'] = frac
        df['Replicate ID'] = rep
        frames.append(df)

    if not frames:
        print("Error: no valid CSV files found!", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    combined.to_csv(args.output, index=False)

    n_pheno = combined['Phenotype'].nunique()
    n_frac = combined['Infected Fraction'].nunique()
    n_reps = combined.groupby(['Phenotype', 'Infected Fraction'])['Replicate ID'].nunique().median()

    print(f"\n  Combined: {len(combined):,} rows")
    print(f"  Phenotypes:  {n_pheno} ({', '.join(sorted(combined['Phenotype'].unique()))})")
    print(f"  Fractions:   {n_frac}")
    print(f"  Reps/cond:   ~{n_reps:.0f}")
    print(f"  Skipped:     {skipped}")
    print(f"  Saved to:    {args.output}")


if __name__ == '__main__':
    main()