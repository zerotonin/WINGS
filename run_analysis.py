import pandas as pd
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import itertools

def read_data(file_path):
    """
    Reads the CSV file and returns a DataFrame.
    """
    return pd.read_csv(file_path)

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """
    Calculate bootstrap confidence interval for the median.
    """
    bootstrapped_medians = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_medians.append(np.median(sample))
    
    lower_bound = np.percentile(bootstrapped_medians, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_medians, (1 + ci) / 2 * 100)
    return lower_bound, upper_bound

def calculate_statistics(data):
    """
    Calculate required statistics for population size and infection rate.
    """
    stats_dict = {
        'mean': data.mean(),
        'median': data.median(),
        'sem': data.sem(),  # Standard Error of the Mean
    }

    # 95% Confidence Interval of the Median
    stats_dict['ci_median'] = bootstrap_ci(data)

    return stats_dict

def get_wolbachia_effects():
    return {'cytoplasmic_incompatibility': 'ci',
            'male_killing': 'mk',
            'increased_exploration_rate': 'er',
            'increased_eggs': 'eg'
            }

def combination_to_string(effects, combination):
    
    return ''.join(effects[e] for e, c in zip(effects, combination) if c) or 'no_effects'

def main(file_path,output_file):
    # Read the data
    df = read_data(file_path)

    effects = get_wolbachia_effects()

    # Generate all combinations of Wolbachia effects
    all_combinations = list(itertools.product([True, False], repeat=len(effects)))

    # List to store statistics for each combination and day
    stats_list = []

    with tqdm(total=len(all_combinations), desc="Simulating") as pbar:
        for combination in all_combinations:
            combination_str = combination_to_string(effects,combination)
            # Filter data for the current combination
            combination_data = df[(df['cytoplasmic_incompatibility'] == combination[0]) &
                                (df['male_killing'] == combination[1]) &
                                (df['increased_exploration_rate'] == combination[2]) &
                                (df['increased_eggs'] == combination[3])]

            for day in combination_data['days'].unique():
                day_data = combination_data[combination_data['days'] == day]
                pop_size_stats = calculate_statistics(day_data['Population Size'])
                infection_rate_stats = calculate_statistics(day_data['Infection Rate'])
                stats_entry = {
                    'combination': combination_str,
                    'day': day,
                    'pop_size_mean': pop_size_stats['mean'],
                    'pop_size_median': pop_size_stats['median'],
                    'pop_size_ci_lower': pop_size_stats['ci_median'][0],
                    'pop_size_ci_upper': pop_size_stats['ci_median'][1],
                    'pop_size_sem': pop_size_stats['sem'],
                    'infection_rate_mean': infection_rate_stats['mean'],
                    'infection_rate_median': infection_rate_stats['median'],
                    'infection_rate_ci_lower': infection_rate_stats['ci_median'][0],
                    'infection_rate_ci_upper': infection_rate_stats['ci_median'][1],
                    'infection_rate_sem': infection_rate_stats['sem']
                }

                stats_list.append(stats_entry)
                pbar.update(1)

    # Convert the list of statistics to a DataFrame
    stats_df = pd.DataFrame(stats_list)

    # Save the DataFrame to CSV
    stats_df.to_csv(output_file, index=False)

    print(f"Statistics saved to {output_file}")

if __name__ == "__main__":
    file_path = 'wolbachia_data.csv'  # Update with the path to your CSV file
    main(file_path,"Wolbachia_analysis.csv")
