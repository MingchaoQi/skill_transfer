# Author: Minh Hua
# Date: 10/31/2022
# Purpose: This module contains classes to load transition data collected from reinforcement learning.

import pandas as pd

class TransitionDataLoader:
    """Loads a dataset of transitions partitioned by the action that caused the transition"""
    def __init__(self, file_path:str, current_state_cols:list, next_state_cols:list, action:int, action_col_name:str='Action') -> None:
        """
        Description:
            Initializes a one-layer neural network with linear output.

        Arguments:
            file_path: the path to the csv containing the transition data.
            current_state_cols: a list of columns that represent the current state variables, i.e. s.
            next_state_cols: a list of columns that represent the afterstate variables, i.e. s'.
            action: the action to partition the data by. Each dataset should only contain transition samples based on one action.
            action_col_name: the name of the column containing the current action. The default is 'Action'.

        Return:
            (None)
        """
        self.transition_df = pd.read_csv(file_path, index_col=False)
        # filter the dataframe by action
        self.transition_df = self.transition_df[self.transition_df[action_col_name] == action]
        # retain only the columns that we need in the data
        self.transition_df = self.transition_df[current_state_cols + next_state_cols]
        # reset the index of the dataframe
        self.transition_df = self.transition_df.reset_index(drop=True)
        # keep track of the current state and afterstate columns
        self.current_state_cols = current_state_cols
        self.next_state_cols = next_state_cols
        self.action = action
    
    def split_features_targets(self, target:str) -> pd.DataFrame:
        """
        Description:
            Returns the transition dataframe filtered by one single target.

        Arguments:
            target: the name of the target. Must be in self.next_state_cols.

        Return:
            (None)
        """
        return self.transition_df[self.current_state_cols + [target]]