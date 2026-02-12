import pandas as pd
import os
import re
from tqdm import tqdm

def extract_metadata_from_filename(filename):
    """
    Extracts metadata (effect toggles and CI strength) from a result filename.
    Returns a dict with keys for each effect and 'trial_number'.
    """
    base = os.path.splitext(os.path.basename(filename))[0]  # filename without extension
    # Remove the trial number at the end
    # Filename format: <effects>_<trial>. Example: ..._True_False_False_001
    # We assume the last '_' separates the trial number.
    # Also extract ci_strength if present in the middle.
    ci_strength_val = None
    if "_ci_strength_" in base:
        match = re.search(r"_ci_strength_([0-9]+(?:\.[0-9]+)?)", base)
        if match:
            ci_strength_val = float(match.group(1))
        # Remove the ci_strength segment for parsing booleans
        base = re.sub(r"_ci_strength_[0-9]+(?:\.[0-9]+)?", "", base)
    # Split by underscores
    parts = base.split('_')
    # Last part is trial number
    trial_str = parts[-1]
    try:
        trial_num = int(trial_str)
        parts = parts[:-1]  # remove trial part from key parsing
    except ValueError:
        trial_num = None
    metadata = {}
    current_key_segments = []
    for token in parts:
        if token in ['True', 'False']:
            # End of a key name
            key_name = '_'.join(current_key_segments)
            metadata[key_name] = True if token == 'True' else False
            current_key_segments = []
        else:
            current_key_segments.append(token)
    if trial_num is not None:
        metadata['trial_number'] = trial_num
    if ci_strength_val is not None:
        metadata['ci_strength'] = ci_strength_val
    return metadata

def read_and_process_data():
    """
    Reads all CSV result files, attaches metadata, and combines into one DataFrame.
    """
    SAVE_PATH = "./data/compare_spread_features/"
    all_data = []
    files = [f for f in os.listdir(SAVE_PATH) if f.endswith(".csv")]
    for file in tqdm(files, desc="Processing CSV files"):
        file_path = os.path.join(SAVE_PATH, file)
        df = pd.read_csv(file_path)
        metadata = extract_metadata_from_filename(file)
        # Add a 'days' column (index in DataFrame corresponds to days)
        df['days'] = df.index
        # Assign metadata columns
        df = df.assign(**metadata)
        # Build a concise id string for the scenario
        effect_keys = ['cytoplasmic_incompatibility', 'ci_strength', 'male_killing', 
                       'increased_exploration_rate', 'increased_eggs', 'reduced_eggs']
        id_tokens = []
        for key in effect_keys:
            if key in metadata:
                val = metadata[key]
                id_tokens.append(f"{key}_{val}")
        df['id_string'] = '_'.join(id_tokens)
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def plot_time_series(combined_data):
    """
    Plots the median infection rate and population size over time for each scenario (id_string).
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Plot infection rate over time with CI bands for each scenario
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_data, x='days', y='Infection Rate', hue='id_string', ci=95)
    plt.title('Infection Rate Over Time by Scenario')
    plt.xlabel('Days')
    plt.ylabel('Infection Rate')
    plt.legend(title='Wolbachia Effects Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
