import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_stats(data_path, combinations, statistic='median', colorblind=True):
    """
    Plot population size and infection rate over time for specified combinations of Wolbachia effects.

    Args:
        data_path (str): Path to the CSV file with precomputed statistics.
        combinations (list of str): List of combination strings to plot.
        statistic (str): 'median' to plot median with CI, 'mean' to plot mean with SEM.
        colorblind (bool): If True, use a colorblind-friendly palette.
    """
    # Load the precomputed statistics
    df = pd.read_csv(data_path)
    print(df.combination.unique())
    # Set colorblind-friendly palette if requested
    if colorblind:
        palette = sns.color_palette("colorblind")

    figure_handle = plt.figure(figsize=(12, 6))

    # Population Size Plot
    ax1 = plt.subplot(1, 2, 1)
    plot_combination(ax1, df, combinations, 'pop_size', statistic)

    # Infection Rate Plot
    ax2 = plt.subplot(1, 2, 2)
    plot_combination(ax2, df, combinations, 'infection_rate', statistic)

    plt.tight_layout()
    return figure_handle

def plot_combination(ax, df, combinations, column_name, statistic):
    """
    Plots a specific statistic for a set of combinations on a given axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        df (pd.DataFrame): The DataFrame containing the data.
        combinations (list): The list of combinations to plot.
        column_name (str): The name of the column to plot ('Population Size' or 'Infection Rate').
        statistic (str): 'median' or 'mean'.
    """
    for comb in combinations:
        comb_data = df[df['combination'] == comb]

        if statistic == 'median':
            ax.fill_between(comb_data['day'], comb_data[f'{column_name.lower()}_ci_lower'], comb_data[f'{column_name.lower()}_ci_upper'], alpha=0.3)
            ax.plot(comb_data['day'], comb_data[f'{column_name.lower()}_median'], label=comb)
        elif statistic == 'mean':
            ax.fill_between(comb_data['day'], comb_data[f'{column_name.lower()}_mean'] - comb_data[f'{column_name.lower()}_sem'], comb_data[f'{column_name.lower()}_mean'] + comb_data[f'{column_name.lower()}_sem'], alpha=0.3)
            ax.plot(comb_data['day'], comb_data[f'{column_name.lower()}_mean'], label=comb)
    
    ax.set_title(f'{column_name} Over Time')
    ax.set_xlabel('Days')
    ax.set_ylabel(column_name)
    ax.legend()

def save_figure(figure_handle, combinations, save_path, file_format=['png', 'svg']):
    """
    Save the figure in specified formats using a filename derived from the combinations.

    Args:
        figure_handle (matplotlib.figure.Figure): The figure handle to save.
        combinations (list of str): The list of combination strings used in the plot.
        save_path (str): The directory path to save the file.
        file_format (list of str): List of formats to save the figure. Default is ['png', 'svg'].
    """
    # Create a filename based on the combinations
    filename_base = '_'.join(combinations)
    
    # Save in each specified file format
    for fmt in file_format:
        file_path = f'{save_path}/{filename_base}.{fmt}'
        figure_handle.savefig(file_path, format=fmt)

# Example usage
save_path = './figures'  # Update this to your save path
data_path = 'wolbachia_stats.csv'  # Path to your statistics CSV file

combinations_to_plot = [['er', 'ci', 'mk', 'eg', 'no_effects'],
                        ['er', 'ci', 'cier', 'no_effects'],
                        ['er', 'mk', 'mker', 'no_effects'],
                        ['er', 'eg', 'ereg', 'no_effects']
                       ]


for combi in combinations_to_plot:
    f_handle = plot_stats(data_path, combi, statistic='mean', colorblind=True)
    save_figure(f_handle, combi, save_path)

plt.show()