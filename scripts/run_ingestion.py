import pandas as pd
import numpy as np
from glob import glob

# Directory containing CSV files
data_dir = "/home/geuba03p/PyProjects/WINGS/data/compare_spread_features/"
file_pattern = data_dir + "*.csv"

# 1. Load all CSV files efficiently
file_list = glob(file_pattern)
data_frames = []
for file in file_list:
    # Read CSV with specified dtypes to save memory&#8203;:contentReference[oaicite:6]{index=6}
    df = pd.read_csv(file, dtype={
        "Population Size": "float32",
        "Infection Rate": "float32"
    })
    # Add a 'Day' column (1-indexed day assuming each row is a day)
    df.insert(0, "Day", np.arange(1, len(df) + 1, dtype=np.int32))
    data_frames.append(df)

# Combine all data into one DataFrame
combined_df = pd.concat(data_frames, ignore_index=True)
# (Optional) Free up memory by deleting the list of smaller DataFrames
del data_frames

# 2. Compute daily summary statistics (mean, median, std)
daily_stats = combined_df.groupby("Day").agg({
    "Infection Rate": ["mean", "median", "std"],
    "Population Size": ["mean", "median", "std"]
})
# Flatten multi-level columns for convenience
daily_stats.columns = ["_".join(col) for col in daily_stats.columns]
daily_stats = daily_stats.reset_index()  # make 'Day' a column instead of index

# Function to bootstrap 95% CI for the mean of a given array
def bootstrap_ci(data_array, n_bootstrap=1000, ci=0.95):
    """Return the (lower, upper) CI bounds for the mean using bootstrapping."""
    # Generate bootstrap samples and compute their means&#8203;:contentReference[oaicite:7]{index=7}
    means = []
    n = len(data_array)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data_array, size=n, replace=True)
        means.append(sample.mean())
    # Calculate percentile bounds for the two-tailed CI&#8203;:contentReference[oaicite:8]{index=8}
    lower_bound = np.percentile(means, (1-ci)/2 * 100)   # e.g., 2.5th percentile
    upper_bound = np.percentile(means, (1-(1-ci)/2) * 100)  # e.g., 97.5th percentile
    return lower_bound, upper_bound

# Prepare lists to collect CI results
ci_days = []
ci_inf_lower = []
ci_inf_upper = []
ci_pop_lower = []
ci_pop_upper = []

# Compute bootstrapped CI for each day group
for day, group in combined_df.groupby("Day"):
    inf_rate_values = group["Infection Rate"].values
    pop_size_values = group["Population Size"].values
    # Bootstrap CIs for infection rate and population size
    inf_ci_lo, inf_ci_hi = bootstrap_ci(inf_rate_values, n_bootstrap=1000, ci=0.95)
    pop_ci_lo, pop_ci_hi = bootstrap_ci(pop_size_values, n_bootstrap=1000, ci=0.95)
    ci_days.append(day)
    ci_inf_lower.append(inf_ci_lo)
    ci_inf_upper.append(inf_ci_hi)
    ci_pop_lower.append(pop_ci_lo)
    ci_pop_upper.append(pop_ci_hi)

# Create a DataFrame for confidence intervals
ci_df = pd.DataFrame({
    "Day": ci_days,
    "InfectionRate_CI_lower": ci_inf_lower,
    "InfectionRate_CI_upper": ci_inf_upper,
    "PopulationSize_CI_lower": ci_pop_lower,
    "PopulationSize_CI_upper": ci_pop_upper
})

# Merge CI bounds with the daily_stats DataFrame
daily_summary_df = pd.merge(daily_stats, ci_df, on="Day")

# 3. Save the combined dataset and daily summary statistics
combined_df.to_csv("combined_dataset.csv", index=False)
daily_summary_df.to_csv("daily_summary_stats.csv", index=False)

# (The CSV files can later be loaded into pandas for Seaborn visualization)
