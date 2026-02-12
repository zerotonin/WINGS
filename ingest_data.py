import os
import re
import pandas as pd
from pathlib import Path

def parse_conditions_from_filename(filename: str):
    """
    Extract experimental conditions and replicate ID from filename.
    Expected filename format:
      cytoplasmic_incompatibility_<True/False>_male_killing_<True/False>_increased_exploration_rate_<True/False>_increased_eggs_<True/False>_<replicate>.csv
    Returns: tuple (ci, mk, er, eg, rep_id) with bools for conditions and int for replicate.
    """
    # Regular expression to extract conditions and replicate ID
    pattern = r'cytoplasmic_incompatibility_(True|False)_male_killing_(True|False)_increased_exploration_rate_(True|False)_increased_eggs_(True|False)_(\d+)\.csv$'
    match = re.match(pattern, filename)
    if match:
        ci, mk, er, eg, rep_id = match.groups()
        return ci == 'True', mk == 'True', er == 'True', eg == 'True', int(rep_id)
    else:
        raise ValueError(f"Filename {filename} does not match the expected pattern.")

def main():
    data_dir = Path("/home/geuba03p/PyProjects/WINGS/data/compare_spread_features")
    all_files = sorted(data_dir.glob("*.csv"))  # List all CSV files

    combined_data = []  # Collect data frames for each file

    for file_path in all_files:
        filename = file_path.name  # Extract file name
        try:
            # Parse conditions and replicate ID from filename
            ci, mk, er, eg, rep_id = parse_conditions_from_filename(filename)
        except ValueError as e:
            print(e)
            continue  # Skip files that don't match the expected pattern

        # Read CSV
        df = pd.read_csv(file_path)
        # Ensure the DataFrame has expected columns
        if 'Population Size' not in df.columns:
            for col in df.columns:
                if 'pop' in col.lower():
                    df.rename(columns={col: 'Population Size'}, inplace=True)
        if 'Infection Rate' not in df.columns:
            for col in df.columns:
                if 'infect' in col.lower():
                    df.rename(columns={col: 'Infection Rate'}, inplace=True)
        # Assign a Day column (starting at 1, assuming each row is a day)
        if 'Day' not in df.columns:
            df.insert(0, 'Day', range(1, len(df) + 1))
        # Add condition columns
        df['Cytoplasmic Incompatibility'] = ci
        df['Male Killing'] = mk
        df['Increased Exploration Rate'] = er
        df['Increased Eggs'] = eg
        df['Replicate ID'] = rep_id
        combined_data.append(df)

    # Concatenate all data into one DataFrame
    combined_df = pd.concat(combined_data, ignore_index=True)
    # Save the combined dataset to CSV
    combined_df.to_csv("./data/combined_simulation_data.csv", index=False)
    print(f"Combined data saved to combined_simulation_data.csv (Total rows: {len(combined_df)})")

if __name__ == "__main__":
    main()
