import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools

def read_data(file_paths):
    """
    Reads one or multiple CSV files and returns a combined DataFrame.
    Accepts a single file path or a list of paths.
    """
    if isinstance(file_paths, list):
        df_list = [pd.read_csv(fp) for fp in file_paths]
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.read_csv(file_paths)
    return df

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """
    Calculate bootstrap confidence interval for the median of the data.
    """
    medians = np.empty(n_bootstrap)
    n = len(data)
    # Bootstrap resampling
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        medians[i] = np.median(sample)
    lower = np.percentile(medians, ((1-ci)/2)*100)
    upper = np.percentile(medians, (ci + (1-ci)/2)*100)
    return lower, upper

def get_wolbachia_effects():
    # Include all binary Wolbachia effect factors
    return {
        'cytoplasmic_incompatibility': 'ci',
        'male_killing': 'mk',
        'increased_exploration_rate': 'er',
        'increased_eggs': 'eg',
        'reduced_eggs': 're'
    }

def combination_to_string(effect_map, combo):
    """
    Create a short label for a combination of effects.
    Example: combo (True, False, True, False, False) -> "ci_er"
    """
    labels = [effect_map[e] for e, flag in zip(effect_map, combo) if flag]
    return "_".join(labels) if labels else "no_effects"

def calculate_statistics(series):
    """
    Compute mean, median, SEM, and 95% CI of median for a pandas Series.
    """
    stats = {
        'mean': series.mean(),
        'median': series.median(),
        'sem': series.sem()  # Standard error of the mean
    }
    stats['ci_median_lower'], stats['ci_median_upper'] = bootstrap_ci(series.values)
    return stats

def main(input_path, output_file):
    # Load data (supports single file or list of files)
    df = read_data(input_path)
    effects = get_wolbachia_effects()
    all_combinations = list(itertools.product([True, False], repeat=len(effects)))
    stats_list = []

    # Determine if ci_strength is a factor in the data
    ci_strength_values = df['ci_strength'].unique() if 'ci_strength' in df.columns else [None]

    with tqdm(total=len(ci_strength_values) * len(all_combinations), desc="Analyzing") as pbar:
        for ci_val in ci_strength_values:
            # Filter by ci_strength if applicable
            subset = df if ci_val is None else df[df['ci_strength'] == ci_val]
            for combo in all_combinations:
                combo_dict = {
                    'cytoplasmic_incompatibility': combo[0],
                    'male_killing': combo[1],
                    'increased_exploration_rate': combo[2],
                    'increased_eggs': combo[3],
                    'reduced_eggs': combo[4]
                }
                combo_data = subset
                for effect, value in combo_dict.items():
                    combo_data = combo_data[combo_data[effect] == value]
                if combo_data.empty:
                    pbar.update(1)
                    continue  # skip combos not present in data
                combo_label = combination_to_string(effects, combo)
                # Iterate over each unique time point (day)
                for day in combo_data['days'].unique():
                    day_data = combo_data[combo_data['days'] == day]
                    # Compute stats for population size and infection rate
                    pop_stats = calculate_statistics(day_data['Population Size'])
                    inf_stats = calculate_statistics(day_data['Infection Rate'])
                    entry = {
                        'ci_strength': ci_val if ci_val is not None else np.nan,
                        'combination': combo_label,
                        'day': day,
                        'pop_size_mean': pop_stats['mean'],
                        'pop_size_median': pop_stats['median'],
                        'pop_size_sem': pop_stats['sem'],
                        'pop_size_ci_lower': pop_stats['ci_median_lower'],
                        'pop_size_ci_upper': pop_stats['ci_median_upper'],
                        'infection_rate_mean': inf_stats['mean'],
                        'infection_rate_median': inf_stats['median'],
                        'infection_rate_sem': inf_stats['sem'],
                        'infection_rate_ci_lower': inf_stats['ci_median_lower'],
                        'infection_rate_ci_upper': inf_stats['ci_median_upper']
                    }
                    stats_list.append(entry)
                pbar.update(1)
    # Create DataFrame of statistics and save to CSV
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(output_file, index=False)
    print(f"Aggregated statistics saved to {output_file}")

if __name__ == "__main__":
    # Example usage: adjust file path(s) as needed
    input_data = "wolbachia_data.csv"  # or a list of CSV files
    main(input_data, "Wolbachia_analysis_summary.csv")
