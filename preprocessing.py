"""
Preprocessing function for handling data manipulation.
"""


def apply_filter(input_df, filter_variable, filter_list):
    """
    Applies a filter to the input dataset based on specified unique string indices for a given 
    variable.
    """
    filter_variable_unique_values = \
        sorted(list(set(input_df[filter_variable].tolist())))
    filter_values = [filter_variable_unique_values[i] for i in filter_list]
    input_df = input_df[input_df[filter_variable].isin(filter_values)]
    return input_df
