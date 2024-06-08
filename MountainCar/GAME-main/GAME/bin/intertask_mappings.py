# Author: Minh Hua
# Date: 11/1/2022
# Purpose: This module contains code to support inter-task mappings.

import pandas as pd
import numpy as np
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle

class IntertaskMapping:
    """A class that represents an inter-task mapping consisting of a state mapping and action mapping."""
    def __init__(self, 
        state_mapping:list,
        action_mapping:list,
        src_state_var_names:list,
        src_action_names:list,
        target_state_var_names:list,
        target_action_names:list,
        fitness:float=None
    ):
        """
        Description:
            Initializes an inter-task mapping.

        Arguments:
            state_mapping: a list of values indicating the state mapping. Treat this as the state mapping chromosome.
            action_mapping: a list of values indicating the action mapping. Treat this as the action mapping chromsome.
            src_state_var_names: a list of the source task state variables in plain text. Used to decode the chromosome.
            src_actions: a list of the source task actions in plain text. Used to decode the chromosome.
            target_state_var_names: a list of the target task state variables in plain text. Used to decode the chromosome.
            target_actions: a list of the target actions in plain text. Used to decode the chromosome.
            fitness: the initial fitness of the mapping.

        Return:
            (None)
        """
        # intertask mapping attributes
        self.state_mapping = state_mapping
        self.action_mapping = action_mapping
        self.fitness = None

        # assign a unique ID to the offspring as a function of its state and action mapping
        self.ID = self.create_ID()

        # assign meaning to the state and action mappings
        # note that the map goes from the target to the source 注意映射是从目标域到源域的
        self.decoded_state_mapping = {target_state_var_names[i] : src_state_var_names[j] for i, j in zip(range(len(target_state_var_names)), state_mapping)}
        self.decoded_action_mapping = {target_action_names[i] : src_action_names[j] for i, j in zip(range(len(target_action_names)), action_mapping)}
        self.multiple_mapped_actions = [] # get a list of multiple target actions that a single src task action gets mapped to
        for src_action in range(len(src_action_names)):
            self.multiple_mapped_actions.append([idx for idx in range(len(action_mapping)) if action_mapping[idx] == src_action])

    def create_ID(self) -> str:
        """
        Description:
            Creates an ID for the mapping as a function of its state and action mapping.

        Arguments:
            None

        Return:
            (None)
        """
        return "".join(str(s) for s in self.state_mapping) + "".join(str(a) for a in self.action_mapping)

    def __str__(self) -> str:
        return 'ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}'.format(self.ID, self.state_mapping, self.action_mapping, self.fitness)
        # return 'State mapping: {}, Action mapping: {}'.format(self.decoded_state_mapping.values(), self.decoded_action_mapping.values())

class EvaluationNetworks:
    """This class contains all the neural networks that are used to evaluate an intertask mapping's transformed data."""
    def __init__(self, nn_folder_path:str) -> None:
        """
        Description:
            Initializes a class that holds all the neural networks for evaluating intertask mappings.

        Arguments:
            nn_folder_path: the path containing the pickled neural networks. Assumes that these networks ahve already been trained.

        Return:
            (None)
        """
        self.networks = {}
        # read in the neural networks
        for file in os.listdir(nn_folder_path):
            if file.endswith(".pickle"):
                # create a key for the network which is the action and the predicted state
                split_file = file.split('--')
                nn_key = split_file[0][1] + '_' + split_file[1][1:].split('.')[0]
                with open(os.path.join(nn_folder_path, file), 'rb') as f:
                    mlp = pickle.load(f)
                    self.networks[nn_key] = mlp

    def get_network(self, action:int, state:str) -> MLPRegressor:
        """
        Description:
            Returns the neural network that predicts the state given the action.

        Arguments:
            action: the action that the neural network approximates.
            state: the state that the neural network predicts.

        Return:
            (MLPRegressor) the neural network that predicts the state.
        """
        return self.networks[str(action) + '_' + state]

def transform_source_dataset(src_dataset:pd.DataFrame, intertask_map:IntertaskMapping, target_col_names:list, target_actions:list) -> pd.DataFrame:
    """
    Description:
        Transforms a dataset of transitions from the source task into a dataset of transitions in the target task.

    Arguments:
        src_dataset: the dataset containing transition samples in the source task.
        intertask_map: the intertask mapping to transform the samples.
        target_col_names: the names of the columns in the target task (transformed) dataset.
        target_actions: a list of actions in the target task.

    Return:
        (pd.DataFrame) a dataset of transformed transition samples from the source task.
    """
    # init empty dataframe for transformed src data
    transformed_src_df = pd.DataFrame()

    # transform state data first
    for col_name in target_col_names:
        # copy all state columns and ignore actions for now
        split_col_names = col_name.split('-')
        current_or_next = split_col_names[0]
        state_or_action = split_col_names[1]
        # check that we are not looking at the action column
        if not state_or_action == 'action':
            # we are looking at s or s'
            # construct the mapped state
            # reconstructed_col_name = "_".join(split_col_names[1:])
            # transform column name into source column using mapping
            src_task_col_name = intertask_map.decoded_state_mapping[state_or_action]
            # are we looking at s or s'?
            if current_or_next == 'Current' or current_or_next == 'Next':
                transformed_src_df[col_name] = src_dataset[current_or_next + '-' + src_task_col_name]
    
    # create dummy variables to mark multiple mapped actions
    action_dummy_data = np.zeros(shape=(len(target_actions), len(src_dataset)))
    for row_idx, row in src_dataset.iterrows():
        multiple_mapped_actions = intertask_map.multiple_mapped_actions[int(row['Current-action'])]
        for mapped_action in multiple_mapped_actions:
            action_dummy_data[mapped_action][row_idx] = 1
    for action_dummy_idx, target_action in enumerate(target_actions):
        transformed_src_df['Current-action-' + str(target_action)] = action_dummy_data[action_dummy_idx]
    
    return transformed_src_df.reset_index(drop=True)

def evaluate_mapping(
    mapping:IntertaskMapping,
    transformed_df:pd.DataFrame,
    eval_networks:EvaluationNetworks,
    current_state_cols:list,
    next_state_cols:list,
    actions:list,
    standard_features:bool=False,
    standard_targets:bool=False,
    standardizer_paths:str=None
) -> dict:
    """
    Description:
        Evaluates an intertask mapping using the transformed dataset and the evaluation networks.

    Arguments:
        mapping: the intertask mapping.
        transformed_df: the dataset of transformed source task transitions.
        eval_networks: the set of neural networks to be used for evaluation.
        current_state_cols: the names of the columns that represent the current state.
        next_state_cols: the names of the columns that represent the afterstate.
        actions: a list of actions in the target task.
        standard_features: whether to standardize the features.
        standard_targets: whether to standardize the targets.
        standardizer_paths: the path containing the standardizers.

    Return:
        (pd.DataFrame) a dataset of transformed transition samples from the source task.
    """
    # init a dictionary to hold the evaluation metrics
    eval_scores = {}
    for action in actions:
        # filter the dataset by the action
        action_col_name = 'Current-action-' + str(action)
        src_df_by_action = transformed_df[transformed_df[action_col_name] == 1]

        # evaluate the mapping's transformed df
        # we evaluate each afterstate variable independently
        for target_name in next_state_cols:
            if standard_features:
                with open(os.path.join(standardizer_paths, 'a{}--s{}--feature--scaler.pickle'.format(action, target_name)), 'rb') as f:
                    feature_scaler = pickle.load(f)
                features = feature_scaler.transform(src_df_by_action[current_state_cols])
            else:  
                features = src_df_by_action[current_state_cols]            
            if standard_targets:
                with open(os.path.join(standardizer_paths, 'a{}--s{}--target--scaler.pickle'.format(action, target_name)), 'rb') as f:
                    target_scaler = pickle.load(f)
                target = target_scaler.transform(np.array(src_df_by_action[target_name]).reshape(-1, 1)).reshape(len(src_df_by_action[target_name]), )
            else:                
                target = src_df_by_action[target_name]
            eval_mlp = eval_networks.get_network(action, target_name)
            # eval_score = eval_mlp.score(features, target)
            y_pred = eval_mlp.predict(features)
            eval_score = 1 - mean_squared_error(target, y_pred)
            eval_scores['{}--{}'.format(action, target_name)] = eval_score
    # return the dictionary of raw scores
    return eval_scores

def parse_mapping_eval_scores(eval_scores:dict, strategy:str='average') -> list:
    """
    Description:
        Convert the intertask mapping evaluation scores into a single score based on different strategies.

    Arguments:
        eval_scores: the dictionary of evaluation scores returned by evaluate_mapping().
        strategy: different ways to convert eval_scores into a single score.
            'average': average all the scores into one score.
            'by_state': return a list of scores aggregated by the next predicted state.
            'by_action': return a list of scores aggregated by the action.

    Return:
        (list) consolidated scores.
    """
    consolidated_scores = []
    if strategy == 'average': # just average all the scores
        consolidated_scores.append(sum(eval_scores.values()) / len(eval_scores.values()))
    
    return consolidated_scores

if __name__ == "__main__":
    from GAME.utils.config import config

    # load config data
    config_data = config()

    src_state_var_names = config_data['MC2D_state_names']
    src_action_names = config_data['MC2D_action_names']
    src_action_values = config_data['MC2D_action_values']
    target_state_var_names = config_data['MC3D_state_names']
    target_action_names = config_data['MC3D_action_names']
    target_action_values = config_data['MC3D_action_values']

    mapping = IntertaskMapping([1, 0, 1, 0], [1, 0, 2, 2, 2], src_state_var_names, src_action_names, target_state_var_names, target_action_names)

    src_data_path = config_data['output_path'] + "11032022 2DMC Sample Collection 100 Episodes with Training\\2DMC_100_episodes_sample_data.csv"
    src_data_df = pd.read_csv(src_data_path, index_col = False)
    transformed_df_col_names = config_data['3DMC_full_transition_df_col_names']

    transformed_df = transform_source_dataset(src_data_df, mapping, transformed_df_col_names, target_action_values)
    # transformed_df_out_path = "C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Evolutionary and Swarm Intelligence\\src\\GAME\\output\\11022022 Transformed Source Data 3DMC\\transformed_df.csv"
    # transformed_df.to_csv(transformed_df_out_path, index = False)

    network_folder_path = config_data['pickle_path'] + "11012022 3DMC Neural Nets\\"
    eval_networks = EvaluationNetworks(network_folder_path)
    # test_mlp = eval_networks.get_network(0, 'Next_x_position')

    transformed_df_current_state_cols = config_data['3DMC_current_state_transition_df_col_names']
    transformed_df_next_state_cols = config_data['3DMC_next_state_transition_df_col_names']

    eval_results = evaluate_mapping(mapping, transformed_df, eval_networks, transformed_df_current_state_cols, transformed_df_next_state_cols, target_action_values)
    print(parse_mapping_eval_scores(eval_results))