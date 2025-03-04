import pandas as pd
import numpy as np
import glob, os
from tqdm import tqdm

def parse_filename(filename):
    """
    Parse simulation parameters from the CSV filename.
    Returns a dict of parameters:
      - cytoplasmic_incompatibility (bool)
      - ci_strength (float or NaN if CI is False)
      - male_killing (bool)
      - increased_exploration_rate (bool)
      - increased_eggs (bool)
      - reduced_eggs (bool)
      - replicate_id (int)
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    # Regex pattern matching the expected filename format
    import re
    pattern = re.compile(
        r'cytoplasmic_incompatibility_(True|False)_'
        r'ci_strength_([0-9]*\.?[0-9]+)_'
        r'male_killing_(True|False)_'
        r'increased_exploration_rate_(True|False)_'
        r'increased_eggs_(True|False)_'
        r'reduced_eggs_(True|False)_'
        r'(\d+)$'  # replicate ID at the end
    )
    match = pattern.match(base)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern.")
    ci_str, ci_strength_str, mk_str, ie_str, inc_eggs_str, red_eggs_str, rep_str = match.groups()
    params = {
        'cytoplasmic_incompatibility': True if ci_str == 'True' else False,
        'ci_strength': float(ci_strength_str),
        'male_killing': True if mk_str == 'True' else False,
        'increased_exploration_rate': True if ie_str == 'True' else False,
        'increased_eggs': True if inc_eggs_str == 'True' else False,
        'reduced_eggs': True if red_eggs_str == 'True' else False,
        'replicate_id': int(rep_str)
    }
    # If CI is False, ci_strength is not applicable
    if not params['cytoplasmic_incompatibility']:
        params['ci_strength'] = np.nan
    return params

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """
    Compute a bootstrap confidence interval for the median of a 1D numpy array.
    Returns a tuple (lower_bound, upper_bound) for the given confidence level.
    """
    data = np.asarray(data)
    boot_medians = []
    # Perform bootstrap resampling
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=data.size, replace=True)
        boot_medians.append(np.median(sample))
    # Compute confidence interval bounds
    lower_bound = np.percentile(boot_medians, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(boot_medians, (1 + ci) / 2 * 100)
    return lower_bound, upper_bound

def calculate_statistics(data):
    """
    Calculate mean, median, standard deviation, and 95% CI for the median for the given data.
    """
    arr = np.asarray(data)
    stats_dict = {
        'mean': np.mean(arr),
        'median': np.median(arr),
        'std': np.std(arr, ddof=1) if arr.size > 1 else 0.0  # sample std; 0.0 if only one value
    }
    # 95% CI for the median via bootstrap
    ci_lower, ci_upper = bootstrap_ci(arr, n_bootstrap=1000, ci=0.95)
    stats_dict['median_ci_lower'] = ci_lower
    stats_dict['median_ci_upper'] = ci_upper
    return stats_dict

def main(input_dir, combined_output_file, stats_output_file):
    """
    Read all simulation CSV files from input_dir, combine data with parameter labels,
    compute daily summary stats per parameter combination, and save results.
    """
    # Get all CSV files matching the simulation filename pattern in the directory
    pattern = os.path.join(input_dir, "cytoplasmic_incompatibility_*_ci_strength_*_*.csv")
    file_list = glob.glob(pattern)
    if not file_list:
        raise FileNotFoundError(f"No CSV files found in directory '{input_dir}' matching the simulation output pattern.")
    
    combined_data = []  # will collect data from each file
    for file_path in tqdm(file_list, desc="Reading simulation outputs"):
        params = parse_filename(file_path)
        # Read the CSV data for this replicate
        df_run = pd.read_csv(file_path)
        # Add a Day column if not already present (assuming each row is sequential day data)
        if 'Day' not in df_run.columns and 'days' not in df_run.columns:
            df_run.insert(0, 'Day', range(1, len(df_run) + 1))
        # Attach parameter values to each row
        for key, value in params.items():
            df_run[key] = value
        # Ensure boolean columns are actual bool dtype for memory efficiency
        df_run['cytoplasmic_incompatibility'] = df_run['cytoplasmic_incompatibility'].astype(bool)
        df_run['male_killing'] = df_run['male_killing'].astype(bool)
        df_run['increased_exploration_rate'] = df_run['increased_exploration_rate'].astype(bool)
        df_run['increased_eggs'] = df_run['increased_eggs'].astype(bool)
        df_run['reduced_eggs'] = df_run['reduced_eggs'].astype(bool)
        combined_data.append(df_run)
    # Combine all runs into a single DataFrame
    combined_df = pd.concat(combined_data, ignore_index=True)
    # Reorder columns for clarity
    cols_order = [
        'cytoplasmic_incompatibility', 'ci_strength', 'male_killing',
        'increased_exploration_rate', 'increased_eggs', 'reduced_eggs',
        'replicate_id', 'Day', 'Population Size', 'Infection Rate'
    ]
    combined_df = combined_df[cols_order]
    # Save the full combined dataset (for boxplots or further analysis)
    combined_df.to_csv(combined_output_file, index=False)
    
    # Prepare to compute daily stats per parameter combination
    group_cols = [
        'cytoplasmic_incompatibility', 'ci_strength', 'male_killing',
        'increased_exploration_rate', 'increased_eggs', 'reduced_eggs', 'Day'
    ]
    stats_records = []  # will collect stats for each combo per day
    grouped = combined_df.groupby(group_cols, dropna=False)
    # Iterate over each group (unique parameter combination + day)
    for group_vals, group_df in tqdm(grouped, total=grouped.ngroups, desc="Computing daily statistics"):
        # Unpack grouping values (parameters and day)
        (ci_val, ci_strength_val, mk_val, ie_val, inc_eggs_val, red_eggs_val, day_val) = group_vals
        # Compute stats for this group across replicates
        pop_stats = calculate_statistics(group_df['Population Size'])
        inf_stats = calculate_statistics(group_df['Infection Rate'])
        stats_records.append({
            'cytoplasmic_incompatibility': ci_val,
            'ci_strength': ci_strength_val,
            'male_killing': mk_val,
            'increased_exploration_rate': ie_val,
            'increased_eggs': inc_eggs_val,
            'reduced_eggs': red_eggs_val,
            'Day': day_val,
            'pop_size_mean': pop_stats['mean'],
            'pop_size_median': pop_stats['median'],
            'pop_size_std': pop_stats['std'],
            'pop_size_ci_lower': pop_stats['median_ci_lower'],
            'pop_size_ci_upper': pop_stats['median_ci_upper'],
            'infection_rate_mean': inf_stats['mean'],
            'infection_rate_median': inf_stats['median'],
            'infection_rate_std': inf_stats['std'],
            'infection_rate_ci_lower': inf_stats['median_ci_lower'],
            'infection_rate_ci_upper': inf_stats['median_ci_upper']
        })
    # Create DataFrame of aggregated statistics and save to CSV
    stats_df = pd.DataFrame(stats_records)
    stats_df.to_csv(stats_output_file, index=False)
    print(f"Combined data saved to {combined_output_file}")
    print(f"Daily statistics saved to {stats_output_file}")

# Example usage (update paths as needed):
if __name__ == "__main__":
    input_directory = "/home/geuba03p/PyProjects/WINGS/data/compare_spread_features/"  # directory containing the simulation CSV files
    combined_output = "combined_data.csv"
    stats_output = "aggregated_daily_stats.csv"
    main(input_directory, combined_output, stats_output)
