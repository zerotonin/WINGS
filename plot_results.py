import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure the figures directory exists
os.makedirs("figures", exist_ok=True)

# Load the combined dataset
data_path = "./data/combined_simulation_data.csv"
df = pd.read_csv(data_path)

# Set colorblind-friendly theme
sns.set_theme(style="whitegrid", palette="colorblind")

# **Function to Assign Experiment IDs**
def assign_experiment_ids(df):
    experiment_dict = {
        1: "No Effects",
        2: "Increased Eggs",
        3: "Increased Exploration",
        4: "Increased Eggs, Increased Exploration",
        5: "Male Killing",
        6: "Male Killing, Increased Eggs",
        7: "Male Killing, Increased Exploration",
        8: "Male Killing, Increased Eggs, Increased Exploration",
        9: "Cytoplasmic Incompatibility",
        10: "Cytoplasmic Incompatibility, Increased Eggs",
        11: "Cytoplasmic Incompatibility, Increased Exploration",
        12: "Cytoplasmic Incompatibility, Increased Eggs, Increased Exploration",
        13: "Cytoplasmic Incompatibility, Male Killing",
        14: "Cytoplasmic Incompatibility, Male Killing, Increased Eggs",
        15: "Cytoplasmic Incompatibility, Male Killing, Increased Exploration",
        16: "Cytoplasmic Incompatibility, Male Killing, Increased Eggs, Increased Exploration"
    }

    effect_combinations = [
        (False, False, False, False), (False, False, False, True),
        (False, False, True, False), (False, False, True, True),
        (False, True, False, False), (False, True, False, True),
        (False, True, True, False), (False, True, True, True),
        (True, False, False, False), (True, False, False, True),
        (True, False, True, False), (True, False, True, True),
        (True, True, False, False), (True, True, False, True),
        (True, True, True, False), (True, True, True, True),
    ]
    
    effect_to_experiment_id = {effect_combinations[i]: i + 1 for i in range(16)}
    
    df['experiment_id'] = df.apply(lambda row: effect_to_experiment_id[
        (row['Cytoplasmic Incompatibility'], row['Male Killing'], 
         row['Increased Exploration Rate'], row['Increased Eggs'])], axis=1)
    
    df['experiment_description'] = df['experiment_id'].map(experiment_dict)
    
    return df, experiment_dict

# Apply experiment ID assignment
df, experiment_dict = assign_experiment_ids(df)

# **1. Population Size Over Time**
pop_stats = df.groupby(['experiment_id', 'experiment_description', 'Day'])['Population Size'] \
              .agg(
                  mean='mean',
                  median='median',
                  q25=lambda x: x.quantile(0.25),
                  q75=lambda x: x.quantile(0.75),
              ).reset_index()

# Save the population summary
pop_stats.to_csv("figures/population_over_time.csv", index=False)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
palette = sns.color_palette("colorblind", n_colors=len(pop_stats['experiment_id'].unique()))

for (exp_id, exp_desc), subset in pop_stats.groupby(['experiment_id', 'experiment_description']):
    color = palette[exp_id % len(palette)]
    ax.fill_between(subset['Day'], subset['q25'], subset['q75'], color=color, alpha=0.3)
    ax.plot(subset['Day'], subset['median'], color=color, label=exp_desc)
    ax.plot(subset['Day'], subset['mean'], color=color, linestyle='--')

ax.set_title("Population Size Over Time")
ax.set_xlabel("Day")
ax.set_ylabel("Population Size")
ax.legend(title="Experiment")
fig.savefig("figures/population_over_time.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/population_over_time.svg", bbox_inches="tight")
plt.close(fig)

# **2. Infection Rate Over Time**
inf_stats = df.groupby(['experiment_id', 'experiment_description', 'Day'])['Infection Rate'] \
              .agg(
                  mean='mean',
                  median='median',
                  q25=lambda x: x.quantile(0.25),
                  q75=lambda x: x.quantile(0.75),
              ).reset_index()
inf_stats.to_csv("figures/infection_rate_over_time.csv", index=False)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for (exp_id, exp_desc), subset in inf_stats.groupby(['experiment_id', 'experiment_description']):
    color = palette[exp_id % len(palette)]
    ax.fill_between(subset['Day'], subset['q25'], subset['q75'], color=color, alpha=0.3)
    ax.plot(subset['Day'], subset['median'], color=color, label=exp_desc)
    ax.plot(subset['Day'], subset['mean'], color=color, linestyle='--')

ax.set_title("Infection Rate Over Time")
ax.set_xlabel("Day")
ax.set_ylabel("Infection Rate")
ax.legend(title="Experiment")
fig.savefig("figures/infection_rate_over_time.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/infection_rate_over_time.svg", bbox_inches="tight")
plt.close(fig)

# **3. Boxplots for Final Population Size**
final_df = df.groupby(['experiment_id', 'experiment_description', 'Replicate ID']).last().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x='experiment_description', y='Population Size', hue='experiment_description',
            data=final_df, ax=ax, palette="colorblind", legend=False)
ax.set_title("Final Population Size by Experiment")
ax.set_xlabel("Experiment")
ax.set_ylabel("Final Population Size")
plt.xticks(rotation=45, ha='right')
fig.savefig("figures/final_population_boxplot.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/final_population_boxplot.svg", bbox_inches="tight")
plt.close(fig)

final_df[['experiment_id', 'experiment_description', 'Replicate ID', 'Population Size']].to_csv(
    "figures/final_population_data.csv", index=False)

# **4. Boxplots for Final Infection Rate**
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x='experiment_description', y='Infection Rate', hue='experiment_description',
            data=final_df, ax=ax, palette="colorblind", legend=False)
ax.set_title("Final Infection Rate by Experiment")
ax.set_xlabel("Experiment")
ax.set_ylabel("Final Infection Rate")
plt.xticks(rotation=45, ha='right')
fig.savefig("figures/final_infection_boxplot.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/final_infection_boxplot.svg", bbox_inches="tight")
plt.close(fig)

final_df[['experiment_id', 'experiment_description', 'Replicate ID', 'Infection Rate']].to_csv(
    "figures/final_infection_data.csv", index=False)

# **5. Time-to-Fixation Distribution**
fixation_df = df[df['Infection Rate'] >= 0.49].groupby(
    ['experiment_id', 'experiment_description', 'Replicate ID']
)['Day'].min().reset_index()
fixation_df.rename(columns={'Day': 'Time to Fixation'}, inplace=True)

fig, ax = plt.subplots(figsize=(8, 6))
if len(fixation_df) == 0:
    ax.text(0.5, 0.5, "No experiments reached 49% infection",
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title("Time to Infection Fixation")
else:
    # Filter to groups with >= 2 data points (KDE needs at least 2)
    group_counts = fixation_df.groupby('experiment_description')['Time to Fixation'].count()
    valid_groups = group_counts[group_counts >= 2].index.tolist()
    single_groups = group_counts[group_counts == 1].index.tolist()

    if valid_groups:
        plot_df = fixation_df[fixation_df['experiment_description'].isin(valid_groups)]
        sns.histplot(data=plot_df, x='Time to Fixation', hue='experiment_description',
                     element="step", common_norm=False, kde=True, ax=ax)

    # Add single-point groups as vertical lines
    for grp_name in single_groups:
        val = fixation_df.loc[fixation_df['experiment_description'] == grp_name,
                              'Time to Fixation'].iloc[0]
        ax.axvline(val, linestyle='--', alpha=0.7, label=f"{grp_name} (n=1)")

    # Report combos that never reached fixation
    all_descs = set(experiment_dict.values())
    reached = set(fixation_df['experiment_description'].unique())
    never = all_descs - reached
    if never:
        print(f"  Note: {len(never)} experiment(s) never reached 49% infection:")
        for n in sorted(never):
            print(f"    - {n}")

    ax.set_title("Time to Infection Fixation")
    ax.set_xlabel("Days to Full Infection")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize='small')

fig.savefig("figures/time_to_fixation.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/time_to_fixation.svg", bbox_inches="tight")
plt.close(fig)

fixation_df.to_csv("figures/time_to_fixation.csv", index=False)
print("All figures saved to figures/")