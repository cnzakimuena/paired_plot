"""
Preprocessing class for handling data manipulation and analysis.
"""
import pandas as pd

class Preprocessor:
    """
    This class provides methods for preprocessing a dataset, including filtering, handling 
    duplicates, and preparing variables for analysis and plotting.
    """

    def __init__(self, dataset_path):
        self.data_df = pd.read_csv(dataset_path)
        self.prep_variables = {
            "subject": self.data_df.columns[2],
            "dependent": self.data_df.columns[0],
            "condition": self.data_df.columns[3],
            "group": self.data_df.columns[1]
        }
        self.group_variable_order = None

    @staticmethod
    def apply_filter(input_df, filter_variable, filter_list):
        """
        Applies a filter to the input dataset based on specified unique string indices for a given 
        variable.
        """
        filter_variable_unique_values = \
            sorted(list(set(input_df[filter_variable].tolist())))
        filter_values = [filter_variable_unique_values[i] for i in filter_list]
        input_df = input_df[input_df[filter_variable].isin(filter_values)].copy()
        return input_df

    def apply_multiple_filters(self, condition_filters, group_filters):
        """ 
        Applies multiple filters to the input dataframe based on specified unique string indices 
        for condition and group variables.
        """
        # if condition filters specified, apply to preprocessing input dataframe
        if condition_filters:
            self.data_df = \
                self.apply_filter(self.data_df,
                                  self.prep_variables["condition"],
                                  condition_filters)
        # if group filters specified, apply to preprocessing input dataframe
        if group_filters:
            self.data_df = \
                self.apply_filter(self.data_df,
                                  self.prep_variables["group"],
                                  group_filters)

    def only_duplicates(self):
        """ 
        Keeps only duplicated subjects in the dataset, ensuring that each subject appears 
        more than once. 
        """
        duplicates = \
            self.data_df.duplicated(subset=[self.prep_variables["subject"]], keep=False)
        self.data_df = self.data_df[duplicates].copy()
        # sort by group and subject variables
        self.data_df = self.data_df.sort_values(by=[self.prep_variables["group"],
                                                    self.prep_variables["subject"]]).copy()

    def assign_group_variable_order(self):
        """
        Assigns the order of unique values for the group variable.
        """
        self.group_variable_order = \
            sorted(list(set(self.data_df[self.prep_variables["group"]].tolist())))

    def get_x_tick_labels(self):
        """ 
        Obtains x tick labels for the paired plot based on the unique values of the group variable,
        with special handling for a baseline value of 0.
        """
        x_tick_labels = None
        if 0 in self.group_variable_order:
            x_tick_labels = \
                ['hatching'] + [str(s) + " days" for s in self.group_variable_order[1:]]
        elif 0 not in self.group_variable_order:
            x_tick_labels = [str(s) + " days" for s in self.group_variable_order]
        if x_tick_labels is None:
            raise ValueError(r'The x tick labels list is None.')
        return x_tick_labels
