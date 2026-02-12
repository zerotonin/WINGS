import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set up colorblind-friendly theme for all plots
sns.set_theme(style="whitegrid", palette="colorblind")

# Ensure output directory exists
os.makedirs("figures", exist_ok=True)

# **Load and prepare simulation data**  
# Try to load a combined CSV of simulation results; if not, gather from individual files
try:
    df_raw = pd.read_csv("./data/combined_data.csv")
except FileNotFoundError:
    # If a combined file isn't found, attempt to read all individual result files in a data directory
    import glob
    file_list = glob.glob("data/compare_spread_features/*.csv")  # adjust path/pattern as needed
    data_frames = []
    for filepath in file_list:
        # Parse effect parameters from filename (assuming format key_value in name)
        fname = os.path.basename(filepath)
        fname_parts = fname.replace('.csv', '').split('_')
        params = {}
        replicate_id_id = None
        i = 0
        # Extract key/value pairs from filename parts
        while i < len(fname_parts):
            part = fname_parts[i]
            if part.lower() in ["true", "false"]:
                # This case handles when key is split into multiple parts (like "male" "killing")
                # We should not get here because value True/False should appear after a complete key
                i += 1
                continue
            # If next part exists and is a value (True/False or numeric), then current part(s) form a key
            if i < len(fname_parts)-1:
                next_part = fname_parts[i+1]
                # Check if next part is a boolean or numeric value
                if next_part in ["True", "False"]:
                    key = part
                    params[key] = True if next_part == "True" else False
                    i += 2
                    continue
                try:
                    # If next part can be a float, treat it as numeric value
                    val = float(next_part)
                    key = part
                    params[key] = val
                    i += 2
                    continue
                except ValueError:
                    # If next part is not a value, it means the key has multiple words
                    # e.g., "male", "killing" form "male_killing"
                    key = part + "_" + fname_parts[i+1]
                    # Move index to the second part of the key
                    i += 1
                    continue
            # If we reach here, handle any remaining part or replicate_id id
            # If part looks like an integer (replicate_id index), use it as replicate_id_id
            try:
                rep_val = int(part)
                replicate_id_id = rep_val
            except ValueError:
                # If not a number, it might be a leftover part of a multi-word key; skip it
                pass
            i += 1
        # Combine split key names (e.g., 'male', 'killing' into 'male_killing') in params
        # (Already handled in parsing loop above by concatenating when needed)

        # Read the CSV file
        df_run = pd.read_csv(filepath)
        # If the file has no 'Day' column, create one as sequential Day (starting at 0)
        if 'Day' not in df_run.columns:
            df_run.insert(0, 'Day', range(len(df_run)))
        # Assign effect parameter columns from filename
        # Initialize expected effect keys if missing
        expected_params = ['cytoplasmic_incompatibility', 'ci_strength', 'male_killing',
                            'increased_exploration_rate', 'increased_eggs', 'reduced_eggs']
        for key in expected_params:
            if key in params:
                df_run[key] = params[key]
            else:
                # If not specified in filename, assume False or default
                if key == 'ci_strength':
                    # If CI strength not specified, set to default (e.g., 1.0 or 0.0 depending on CI presence)
                    df_run[key] = 1.0 if params.get('cytoplasmic_incompatibility', False) else 0.0
                else:
                    df_run[key] = False
        # Assign replicate_id ID if parsed
        if replicate_id_id is not None:
            df_run['replicate_id'] = replicate_id_id
        else:
            # If replicate_id not in filename, assign an incremental id
            df_run['replicate_id'] = np.arange(len(df_run['Day']))  # fallback (not ideal, all rows unique)
        data_frames.append(df_run)
    # Concatenate all runs into one DataFrame
    df_raw = pd.concat(data_frames, ignore_index=True)

# Convert boolean columns from strings "True"/"False" to actual booleans if needed
bool_cols = ['cytoplasmic_incompatibility', 'male_killing',
             'increased_exploration_rate', 'increased_eggs', 'reduced_eggs']
for col in bool_cols:
    if col in df_raw.columns and df_raw[col].dtype == object:
        df_raw[col] = df_raw[col].map({"True": True, "False": False})

# **1. Population Size Over Time (Median, Mean, 95% CI)**  
# Focus on a scenario to highlight the effect of increased exploration rate. 
# Here we compare a scenario with only Cytoplasmic Incompatibility (CI) vs the same with Increased Exploration.
# Filter data for the two scenarios: 
# Scenario A: CI = True, other effects = False, exploration = False (CI only)
# Scenario B: CI = True, other effects = False, exploration = True (CI + increased exploration)
cond_base = ((df_raw['cytoplasmic_incompatibility'] == True) &
             (df_raw['male_killing'] == False) &
             (df_raw['increased_eggs'] == False))
if 'reduced_eggs' in df_raw.columns:
    cond_base &= (df_raw['reduced_eggs'] == False)
cond_no_exp = cond_base & (df_raw['increased_exploration_rate'] == False)
cond_yes_exp = cond_base & (df_raw['increased_exploration_rate'] == True)

df_ci = df_raw[cond_no_exp]
df_ci_er = df_raw[cond_yes_exp]

# Compute daily statistics (mean, median, 95% CI) for population size in each scenario
def q025(x):
    return np.quantile(x, 0.025)
def q975(x):
    return np.quantile(x, 0.975)

pop_stats_ci = df_ci.groupby('Day')['Population Size'].agg(mean='mean', median='median',
                                                            ci_lower=q025, ci_upper=q975).reset_index()
pop_stats_er = df_ci_er.groupby('Day')['Population Size'].agg(mean='mean', median='median',
                                                               ci_lower=q025, ci_upper=q975).reset_index()

# Plot population size over time for both scenarios
plt.figure(figsize=(8, 6))
palette = sns.color_palette("colorblind")
# Scenario A: CI only (no exploration)
plt.fill_between(pop_stats_ci['Day'], pop_stats_ci['ci_lower'], pop_stats_ci['ci_upper'],
                 color=palette[0], alpha=0.3)
plt.plot(pop_stats_ci['Day'], pop_stats_ci['median'], color=palette[0],
         label='CI Only (median)')
plt.plot(pop_stats_ci['Day'], pop_stats_ci['mean'], color=palette[0],
         linestyle='--', label='CI Only (mean)')
# Scenario B: CI + Increased Exploration
plt.fill_between(pop_stats_er['Day'], pop_stats_er['ci_lower'], pop_stats_er['ci_upper'],
                 color=palette[1], alpha=0.3)
plt.plot(pop_stats_er['Day'], pop_stats_er['median'], color=palette[1],
         label='CI + Exploration (median)')
plt.plot(pop_stats_er['Day'], pop_stats_er['mean'], color=palette[1],
         linestyle='--', label='CI + Exploration (mean)')

plt.title('Population Size Over Time')
plt.xlabel('Day')
plt.ylabel('Population Size')
plt.legend(title='Scenario')
plt.tight_layout()
plt.savefig("figures/population_size.png")
plt.savefig("figures/population_size.svg")
plt.close()

# Save CSV data for population size plot
pop_plot_df = pd.concat([
    pop_stats_ci.assign(Combination="CI Only"),
    pop_stats_er.assign(Combination="CI + Exploration")
], ignore_index=True)
pop_plot_df.rename(columns={
    'Day': 'Day',
    'mean': 'Population Size Mean',
    'median': 'Population Size Median',
    'ci_lower': 'Population Size CI Lower',
    'ci_upper': 'Population Size CI Upper'
}, inplace=True)
pop_plot_df.to_csv("figures/population_size_data.csv", index=False)

# **2. Infection Rate Over Time (Median, Mean, 95% CI)**  
# Using the same two scenarios (CI only vs CI + exploration)
inf_stats_ci = df_ci.groupby('Day')['Infection Rate'].agg(mean='mean', median='median',
                                                           ci_lower=q025, ci_upper=q975).reset_index()
inf_stats_er = df_ci_er.groupby('Day')['Infection Rate'].agg(mean='mean', median='median',
                                                              ci_lower=q025, ci_upper=q975).reset_index()

# Plot infection rate over time for both scenarios
plt.figure(figsize=(8, 6))
# Scenario A: CI only
plt.fill_between(inf_stats_ci['Day'], inf_stats_ci['ci_lower'], inf_stats_ci['ci_upper'],
                 color=palette[0], alpha=0.3)
plt.plot(inf_stats_ci['Day'], inf_stats_ci['median'], color=palette[0],
         label='CI Only (median)')
plt.plot(inf_stats_ci['Day'], inf_stats_ci['mean'], color=palette[0],
         linestyle='--', label='CI Only (mean)')
# Scenario B: CI + Exploration
plt.fill_between(inf_stats_er['Day'], inf_stats_er['ci_lower'], inf_stats_er['ci_upper'],
                 color=palette[1], alpha=0.3)
plt.plot(inf_stats_er['Day'], inf_stats_er['median'], color=palette[1],
         label='CI + Exploration (median)')
plt.plot(inf_stats_er['Day'], inf_stats_er['mean'], color=palette[1],
         linestyle='--', label='CI + Exploration (mean)')

plt.title('Infection Rate Over Time')
plt.xlabel('Day')
plt.ylabel('Infection Rate')
plt.legend(title='Scenario')
plt.tight_layout()
plt.savefig("figures/infection_rate.png")
plt.savefig("figures/infection_rate.svg")
plt.close()

# Save CSV data for infection rate plot
inf_plot_df = pd.concat([
    inf_stats_ci.assign(Combination="CI Only"),
    inf_stats_er.assign(Combination="CI + Exploration")
], ignore_index=True)
inf_plot_df.rename(columns={
    'Day': 'Day',
    'mean': 'Infection Rate Mean',
    'median': 'Infection Rate Median',
    'ci_lower': 'Infection Rate CI Lower',
    'ci_upper': 'Infection Rate CI Upper'
}, inplace=True)
inf_plot_df.to_csv("figures/infection_rate_data.csv", index=False)

# **3. Boxplots for Wolbachia Effects (Final Population Size)**  
# Calculate the final population size for each replicate_id under each combination of effects
# Determine final (last day) entry for each replicate_id in each combination
group_cols = ['cytoplasmic_incompatibility', 'male_killing',
              'increased_exploration_rate', 'increased_eggs']
if 'reduced_eggs' in df_raw.columns:
    group_cols.append('reduced_eggs')
group_cols.append('replicate_id')
# Get index of last day for each replicate_id (max 'Day')
last_idx = df_raw.groupby(group_cols[:-1])['Day'].idxmax()
df_final = df_raw.loc[last_idx].copy().reset_index(drop=True)

# Prepare a categorical label for combinations (excluding exploration) for plotting
# This label lists which effects (CI, MK, Increased Eggs, Reduced Eggs) are active (True)
effect_labels = []
for _, row in df_final.iterrows():
    parts = []
    if row['cytoplasmic_incompatibility']:
        parts.append("CI")
    if row['male_killing']:
        parts.append("Male Killing")
    if 'increased_eggs' in row and row['increased_eggs']:
        parts.append("Increased Eggs")
    if 'reduced_eggs' in row and row['reduced_eggs']:
        parts.append("Reduced Eggs")
    if not parts:
        parts.append("None")
    effect_labels.append(" + ".join(parts))
df_final['effects_label'] = effect_labels

# Create a categorical column for exploration (Yes/No) for hue
df_final['exploration_flag'] = df_final['increased_exploration_rate'].map({True: 'Yes', False: 'No'})

# Plot boxplot for final population size across effect combinations, colored by exploration
plt.figure(figsize=(10, 6))
sns.boxplot(x='effects_label', y='Population Size', hue='exploration_flag',
            data=df_final, palette="colorblind")
plt.title('Final Population Size by Wolbachia Effect Combination')
plt.xlabel('Wolbachia Effects (excluding Exploration)')
plt.ylabel('Final Population Size')
plt.legend(title='Increased Exploration Rate')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("figures/boxplot_population_size.png")
plt.savefig("figures/boxplot_population_size.svg")
plt.close()

# Save CSV data for final population size boxplot
effect_columns = ['cytoplasmic_incompatibility', 'male_killing',
                  'increased_exploration_rate', 'increased_eggs']
if 'reduced_eggs' in df_final.columns:
    effect_columns.append('reduced_eggs')
out_pop_box_df = df_final[effect_columns + ['Population Size']].copy()
out_pop_box_df.rename(columns={'Population Size': 'Final Population Size'}, inplace=True)
out_pop_box_df.to_csv("figures/boxplot_population_size_data.csv", index=False)

# **4. Boxplots for Wolbachia Effects (Final Infection Rate)**  
# Plot boxplot for final infection rate across effect combinations, colored by exploration
plt.figure(figsize=(10, 6))
sns.boxplot(x='effects_label', y='Infection Rate', hue='exploration_flag',
            data=df_final, palette="colorblind")
plt.title('Final Infection Rate by Wolbachia Effect Combination')
plt.xlabel('Wolbachia Effects (excluding Exploration)')
plt.ylabel('Final Infection Rate')
plt.legend(title='Increased Exploration Rate')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("figures/boxplot_infection_rate.png")
plt.savefig("figures/boxplot_infection_rate.svg")
plt.close()

# Save CSV data for final infection rate boxplot
out_inf_box_df = df_final[effect_columns + ['Infection Rate']].copy()
out_inf_box_df.rename(columns={'Infection Rate': 'Final Infection Rate'}, inplace=True)
out_inf_box_df.to_csv("figures/boxplot_infection_rate_data.csv", index=False)

# **5. Time-to-Fixation Distributions**  
# Determine the distribution of Day when infection reaches 100% (fixation) for the two scenarios (CI vs CI+exploration)
# Extract the first day where infection rate is 1.0 for each replicate_id in each scenario
fix_times_ci = df_ci[df_ci['Infection Rate'] == 1.0].groupby('replicate_id')['Day'].min()
fix_times_er = df_ci_er[df_ci_er['Infection Rate'] == 1.0].groupby('replicate_id')['Day'].min()

# Prepare DataFrame of fixation times with exploration flag
times_df = pd.DataFrame({
    'Increased Exploration Rate': [False]*len(fix_times_ci) + [True]*len(fix_times_er),
    'Time to Fixation (Day)': np.concatenate([fix_times_ci.values, fix_times_er.values])
})

# Plot distribution (histogram) of time-to-fixation for scenarios with and without increased exploration
plt.figure(figsize=(8, 6))
sns.histplot(data=times_df, x='Time to Fixation (Day)', hue='Increased Exploration Rate',
             multiple='layer', stat='probability', common_norm=False, bins=20, alpha=0.5)
plt.title('Distribution of Time to Wolbachia Fixation')
plt.xlabel('Day to 100% Infection')
plt.ylabel('Probability Density')
plt.legend(title='Increased Exploration')
plt.tight_layout()
plt.savefig("figures/time_to_fixation.png")
plt.savefig("figures/time_to_fixation.svg")
plt.close()

# Save CSV data for time-to-fixation plot
times_df.to_csv("figures/time_to_fixation_data.csv", index=False)

# **6. Rate of Spread for Different Initial Infection Conditions**  
# (If data for different initial infection levels is available)
try:
    initial_df = pd.read_csv("initial_conditions.csv")
except FileNotFoundError:
    initial_df = None

if initial_df is not None:
    # Ensure 'Day' column exists
    if 'Day' not in initial_df.columns:
        initial_df.rename(columns={'Day': 'Day'}, inplace=True)
    # If initial infection rate is given as count or fraction, ensure it's a fraction column
    # Assume there's a column 'initial_infection_rate' or similar
    if 'initial_infection_rate' not in initial_df.columns:
        # Try to derive initial infection fraction from first day infection rate per simulation
        initial_df['initial_infection_rate'] = initial_df.groupby('simulation_id')['Infection Rate'].transform('first')  # example, adjust as needed

    # Compute median infection trajectory and 90% CI for each initial infection rate category
    initial_plot_list = []
    # Treat initial infection rate as categorical for grouping (each unique value)
    for init_val, group in initial_df.groupby('initial_infection_rate'):
        stats = group.groupby('Day')['Infection Rate'].agg(
            median='median', ci_lower=lambda x: np.quantile(x, 0.05), ci_upper=lambda x: np.quantile(x, 0.95)
        ).reset_index()
        stats['Initial Infection Rate'] = init_val
        initial_plot_list.append(stats)
    if initial_plot_list:
        initial_plot_df = pd.concat(initial_plot_list, ignore_index=True)
    else:
        initial_plot_df = pd.DataFrame(columns=['Initial Infection Rate', 'Day', 'median', 'ci_lower', 'ci_upper'])

    # Plot infection spread over time for different initial infection fractions
    plt.figure(figsize=(8, 6))
    # Convert initial infection rate to string percentage for legend labels
    unique_initials = sorted(initial_plot_df['Initial Infection Rate'].unique())
    for idx, init_val in enumerate(unique_initials):
        sub = initial_plot_df[initial_plot_df['Initial Infection Rate'] == init_val]
        color = palette[idx % len(palette)]
        plt.fill_between(sub['Day'], sub['ci_lower'], sub['ci_upper'], color=color, alpha=0.3)
        plt.plot(sub['Day'], sub['median'], color=color,
                 label=f'Initial {init_val*100:.0f}% infected (median)')
    plt.title('Infection Spread Over Time for Different Initial Infection Levels')
    plt.xlabel('Day')
    plt.ylabel('Infection Rate')
    plt.legend(title='Initial Infection Level')
    plt.tight_layout()
    plt.savefig("figures/rate_of_spread.png")
    plt.savefig("figures/rate_of_spread.svg")
    plt.close()

    # Save CSV data for rate of spread plot
    initial_plot_df.rename(columns={
        'Day': 'Day',
        'median': 'Infection Rate Median',
        'ci_lower': 'Infection Rate CI Lower',
        'ci_upper': 'Infection Rate CI Upper'
    }, inplace=True)
    initial_plot_df.to_csv("figures/rate_of_spread_data.csv", index=False)
else:
    print("Initial conditions data not found. Skipping rate-of-spread plot.")

# **7. Heatmaps of Infection Fixation Probability for Different Effect Combinations**  
# Calculate the probability of full infection (fixation) for each combination of effects, highlighting increased exploration
# We'll consider combinations of CI and Male Killing (with eggs effects off for clarity) under exploration on/off.
df_heat = df_final.copy()
# Filter to cases with no egg modifications to simplify (increased_eggs = False and reduced_eggs = False if present)
if 'increased_eggs' in df_heat.columns:
    df_heat = df_heat[df_heat['increased_eggs'] == False]
if 'reduced_eggs' in df_heat.columns:
    df_heat = df_heat[df_heat['reduced_eggs'] == False]

# Determine success (fixation) in each replicate_id_id: True if final infection rate is 100%
df_heat['success'] = (df_heat['Infection Rate'] >= 1.0)

# Compute fixation probability for each combination of CI, MK, and exploration
prob_df = df_heat.groupby(['cytoplasmic_incompatibility', 'male_killing', 'increased_exploration_rate'])['success'].mean().reset_index()
prob_df.rename(columns={'success': 'fix_probability'}, inplace=True)

# Pivot data for heatmap: rows = Male Killing (No/Yes), cols = CI (No/Yes)
# Create categorical Yes/No for labeling axes
prob_df['CI'] = prob_df['cytoplasmic_incompatibility'].map({False: 'No', True: 'Yes'})
prob_df['MK'] = prob_df['male_killing'].map({False: 'No', True: 'Yes'})
# Split by exploration status
prob_no_exp = prob_df[prob_df['increased_exploration_rate'] == False]
prob_yes_exp = prob_df[prob_df['increased_exploration_rate'] == True]
heat_data_no = prob_no_exp.pivot(index='MK', columns='CI', values='fix_probability')
heat_data_yes = prob_yes_exp.pivot(index='MK', columns='CI', values='fix_probability')

# Plot side-by-side heatmaps for exploration OFF vs ON
fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
# Heatmap for Exploration = No
sns.heatmap(heat_data_no, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="viridis",
            cbar=False, ax=axes[0])
axes[0].set_title('Exploration Rate: No')
axes[0].set_xlabel('Cytoplasmic Incompatibility')
axes[0].set_ylabel('Male Killing')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
# Heatmap for Exploration = Yes
sns.heatmap(heat_data_yes, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="viridis",
            cbar=True, ax=axes[1], cbar_kws={'label': 'Fixation Probability'})
axes[1].set_title('Exploration Rate: Yes')
axes[1].set_xlabel('Cytoplasmic Incompatibility')
axes[1].set_ylabel('')  # hide redundant y-label on second heatmap
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig("figures/infection_probability_heatmap.png")
plt.savefig("figures/infection_probability_heatmap.svg")
plt.close()

# Save CSV data for infection probability heatmap
# We include combination (CI, MK, exploration) and the computed fixation probability
heatmap_out_df = prob_df.copy()
# Drop auxiliary label columns
heatmap_out_df.drop(columns=['CI', 'MK'], inplace=True, errors='ignore')
heatmap_out_df.rename(columns={
    'cytoplasmic_incompatibility': 'cytoplasmic_incompatibility',
    'male_killing': 'male_killing',
    'increased_exploration_rate': 'increased_exploration_rate',
    'fix_probability': 'fixation_probability'
}, inplace=True)
heatmap_out_df.to_csv("figures/infection_probability_heatmap_data.csv", index=False)
