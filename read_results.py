import pandas as pd
import os
import re
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
SAVE_PATH = "/home/bgeurten/wolbachia_spread_model/raw_data/compare_spread_features/"  # Update with your actual save path
GRID_SIZE = 500
INITIAL_POPULATION = 10
INFECTED_FRACTION = 0.2

def extract_metadata_from_filename(filename):
    """
    Extracts metadata from the filename.

    Parameters:
        filename (str): The name of the file.

    Returns:
        dict: A dictionary containing extracted metadata.
    """
    # Extracting parts of the filename
    parts = os.path.basename(filename).split('_')
    ci = parts[2] == 'True'
    mk = parts[5] == 'True'
    er = parts[9] == 'True'
    eg = parts[12] == 'True'
    trial_number = int(parts[-1].split('.')[0])
    
    return {
        'cytoplasmic_incompatibility': ci,
        'male_killing': mk,
        'increased_exploration_rate': er,
        'increased_eggs': eg,
        'trial_number': trial_number
    }

def read_and_process_data():
    """
    Reads all CSV files, processes them, and combines into a single DataFrame.
    """
    all_data = []
    files = [f for f in os.listdir(SAVE_PATH) if f.endswith(".csv")]
    
    for file in tqdm(files, desc="Processing CSV files"):
        file_path = os.path.join(SAVE_PATH, file)
        data = pd.read_csv(file_path)
        metadata = extract_metadata_from_filename(file)
        data['days'] = data.index
        data = data.assign(**metadata)
        data['grid_size'] = GRID_SIZE
        data['initial_population'] = INITIAL_POPULATION
        data['infected_fraction'] = INFECTED_FRACTION
        all_data.append(data)

    return pd.concat(all_data, ignore_index=True)

def plot_time_series(combined_data):
    """
    Plots the median and confidence interval for each combination of Wolbachia effects over time.

    Parameters:
        combined_data (DataFrame): Pandas DataFrame containing the combined data from all simulations.
    """

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=combined_data, x='days', y='Population Size', hue='id_string', ci=95)
    plt.title('Population Size Over Time by Wolbachia Effects')
    plt.xlabel('Days')
    plt.ylabel('Population Size')
    plt.legend(title='Wolbachia Effects', loc='upper left')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=combined_data, x='days', y='Infection Rate', hue='id_string', ci=95)
    plt.title('Infection Rate Over Time by Wolbachia Effects')
    plt.xlabel('Days')
    plt.ylabel('Infection Rate')
    plt.legend(title='Wolbachia Effects', loc='upper left')
    plt.show()

def generate_id_string(row):
    """
    Generates a unique ID string based on the combination of active Wolbachia effects in a row.
    """
    effects = {
        'cytoplasmic_incompatibility': 'ci',
        'male_killing': 'mk',
        'increased_exploration_rate': 'er',
        'increased_eggs': 'eg'
    }
    id_string = ''.join(abbrev for effect, abbrev in effects.items() if row[effect])

    return id_string if id_string else 'no_effects'



# Read and process the data
combined_data = read_and_process_data()
combined_data['id_string'] = combined_data.apply(generate_id_string, axis=1)
combined_data.to_csv('wolbachia_data.csv',index=False)
# Example: Display the first few rows of the combined DataFrame
print(combined_data.head())
plot_time_series(combined_data)