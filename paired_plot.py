"""
This code generates a paired plot. The plot uses the chick weight dataset for demonstration, and 
includes annotation for significance based on statistical testing. The appearance of the plot is 
customized and the final figure is saved.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import pypalettes

from preprocessing import apply_filter

# non-italic matplotlib greek characters
matplotlib.rcParams['mathtext.default'] = 'regular'


def get_p_value_string(p_value):
    """
    Determines text annotation for a given p-value based on thresholds.
    """
    p_value_string = None
    if p_value < 0.0001:
        p_value_string = "P < 0.0001"
    elif 0.0001 <= p_value <= 0.9999:
        p_value_string = "P = " + f"{p_value:.4f}"
    elif p_value > 0.9999:
        p_value_string = "P > 0.9999"
    if p_value_string is None:
        raise ValueError(r'The p-value string is None.')
    return p_value_string


def generate_plot(df,
                  subject_variable,
                  group_variable,
                  dependent_variable,
                  group_variable_order,
                  **plot_kwargs):
    """
    Generates a paired plot with an optional p-value annotation based on the provided dataframes 
    and parameters.
    """

    sns.set(style="whitegrid", font_scale=1.6)

    # default color palette assignment if not provided
    if 'palette_list' not in plot_kwargs:
        plot_kwargs['palette_list'] = sns.color_palette("Blues", len(group_variable_order))

    with plt.rc_context({'axes.edgecolor': 'black'}):

        # generate plot
        fig, ax = plt.subplots(figsize=(5.5, 8))
        # default x locations
        x_locations = [0.5, 1.5]
        x_span = x_locations[1] - x_locations[0]
        for _, current_subject in enumerate(df[subject_variable].unique()):
            y_locations = \
                [df.loc[(df[group_variable] == group_variable_order[0]) \
                & (df[subject_variable] == current_subject), dependent_variable].item(),
                 df.loc[(df[group_variable] == group_variable_order[1]) \
                     & (df[subject_variable] == current_subject), dependent_variable].item()]
            # dots for one-to-one data
            for q, x_location in enumerate(x_locations):
                ax.plot(x_location, y_locations[q], linestyle="",
                        marker='o',
                        markerfacecolor='none',
                        markeredgecolor=plot_kwargs['palette_list'][q],
                        markeredgewidth=2,
                        markersize=8,
                        zorder=1)
            # lines connecting dots for one-to-one data
            ax.plot(x_locations, y_locations, c='black', linewidth=2, zorder=2)

        # if statistical testing specified, obtain and show p-value
        if 'statistical_testing' in plot_kwargs and plot_kwargs['statistical_testing']:
            # annotation significance (using mannwhitneyu test on the data)
            filtered_df1 = df[df[group_variable] == group_variable_order[0]]
            filtered_df2 = df[df[group_variable] == group_variable_order[1]]
            data1 = filtered_df1[dependent_variable].to_numpy()
            data2 = filtered_df2[dependent_variable].to_numpy()
            _, p_value = mannwhitneyu(data1, data2)
            # add significance line
            dependent_variable_span = \
                plot_kwargs['dependent_variable_range'][1] \
                    - plot_kwargs['dependent_variable_range'][0]
            ax.plot([x_locations[0] + x_span * 0.2, x_locations[1] - x_span * 0.2],
                    [df[dependent_variable].max() + dependent_variable_span * 0.1] * 2,
                    c='black', linewidth=2, zorder=2)
            # add p=value
            significance_string = get_p_value_string(p_value)
            ax.annotate(significance_string,
                        xy=(np.mean(x_locations),
                            df[dependent_variable].max() + dependent_variable_span * 0.18),
                        color='black',
                        fontsize="small", weight='normal',
                        horizontalalignment='center',
                        verticalalignment='center')

        # set x and y limits
        ax.set(xlim=(x_locations[0] - x_span * 0.5,
                     x_locations[1] + x_span * 0.5))
        if 'dependent_variable_range' in plot_kwargs:
            ax.set(ylim=plot_kwargs['dependent_variable_range'])
            plt.yticks(np.linspace(plot_kwargs['dependent_variable_range'][0],
                                   plot_kwargs['dependent_variable_range'][1], num=5).tolist())

        # set x and y labels
        if 'group_variable_label' in plot_kwargs:
            ax.set_xlabel(plot_kwargs['group_variable_label'], labelpad=15)
        if 'dependent_variable_label' in plot_kwargs:
            ax.set_ylabel(plot_kwargs['dependent_variable_label'], labelpad=6)
        if 'specified_x_tick_labels' in plot_kwargs:
            ax.set_xticks(x_locations, labels=plot_kwargs['specified_x_tick_labels'])
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='both', colors='black')

        # hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # only show ticks on the left spines
        ax.get_yaxis().tick_left()
        # orient ticks outward
        ax.tick_params(axis='y', direction='out', width=2)

        # alter axis line size
        # change all spines
        for axis in ['bottom', 'left', 'top', 'right']:
            ax.spines[axis].set_linewidth(2)

        # set the color of the axis labels
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')

        # remove grid lines
        ax.grid(False)

        # adjust subplots spacing
        # if subplots are added, can include, for e.g., 'wspace=0.4, hspace=0.4'
        # to control padding between subplots
        plt.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8)

        # add global title
        if 'super_title' in plot_kwargs:
            fig.suptitle(plot_kwargs['super_title'], fontsize="large", color="black")


if __name__ == '__main__':

    # --- read data ---
    EXAMPLE_DATA_PATH = r'.\chickweight.csv'
    example_data_df = pd.read_csv(EXAMPLE_DATA_PATH)
    # specify preprocessing and plotting variables
    example_subject_variable = example_data_df.columns[2]
    example_dependent_variable = example_data_df.columns[0]
    example_condition_variable = example_data_df.columns[3]
    example_group_variable = example_data_df.columns[1]
    # apply filters
    condition_filter_value = [0]
    group_filter_range_values = [6, 10]
    example_data_df = apply_filter(example_data_df,
                                   example_condition_variable,
                                   condition_filter_value)
    example_data_df = apply_filter(example_data_df,
                                   example_group_variable,
                                   group_filter_range_values)
    # remove non-duplicated subjects
    example_data_df = \
        example_data_df[example_data_df.duplicated(subset=[example_subject_variable], keep=False)]
    # sort by group and subject variables
    example_data_df = \
        example_data_df.sort_values(by=[example_group_variable, example_subject_variable])
    # specify number of subjects per group
    example_data_df = example_data_df.groupby(example_group_variable).head(10)

    # specify data-related plotting parameters
    example_group_variable_order = \
        sorted(list(set(example_data_df[example_group_variable].tolist())))
    example_x_tick_labels = [str(s) + " days" for s in example_group_variable_order]
    # palette setup
    cmap = pypalettes.load_cmap("Chlorurus_microrhinos",
                                keep_first_n=len(example_group_variable_order))
    pypalettes_list = cmap.colors # return colors as a list of hexadecimal values

    # --- plot data ---
    generate_plot(example_data_df,
                  example_subject_variable,
                  example_group_variable,
                  example_dependent_variable,
                  example_group_variable_order,
                  statistical_testing=True,
                  group_variable_label='Age',
                  dependent_variable_label='Weight [g]',
                  dependent_variable_range=[0, 400],
                  specified_x_tick_labels=example_x_tick_labels,
                  palette_list=pypalettes_list)

    # save figure
    FILE_DESTINATION = r'.\figure'
    plt.savefig(os.path.join(FILE_DESTINATION + '.pdf').replace("\\", "/"), format="pdf")
    plt.savefig(os.path.join(FILE_DESTINATION + '.png').replace("\\", "/"), dpi=300)
    plt.close()
