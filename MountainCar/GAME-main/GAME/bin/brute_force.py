# Author: Minh Hua
# Date: 11/3/2022
# Purpose: This module contains the brute force approach to generating an inter-task mapping.

from GAME.bin.intertask_mappings import *
from GAME.utils.config import config
from GAME.utils.stats_saver import StatisticsSaver, MappingSearchExperimentInfo
from sklearn.metrics import mean_squared_error

from itertools import product
import os
import pickle

class IntertaskMappingBf:
    """A class that represents an inter-task mapping consisting of a state mapping and action mapping. Used with the brute force approach."""
    def __init__(self, 
        state_mapping:list,
        src_state_var_names:list,
        src_action_names:list,
        target_state_var_names:list,
        target_action_names:list
    ):
        """
        Description:
            Initializes an inter-task mapping.

        Arguments:
            state_mapping: a list of values indicating the state mapping. Treat this as the state mapping chromosome.
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
        self.action_fitness = {}
        for target_action in range(len(target_action_names)):
            for src_action in range(len(src_action_names)):
                self.action_fitness[str(target_action) + '--' + str(src_action)] = None
        self.fitness = None

        # assign a unique ID to the offspring as a function of its state and action mapping
        self.ID = self.create_ID()

    def create_ID(self) -> str:
        """
        Description:
            Creates an ID for the mapping as a function of its state and action mapping.

        Arguments:
            None

        Return:
            (None)
        """
        return "".join(str(s) for s in self.state_mapping)

class MASTER:
    """
    Implementation of the MASTER algorithm presented by Matthew E. Taylor, Gregory Kuhlmann, and Peter Stone
    in Taylor, M. E., Kuhlmann, G., & Stone, P. (2008, May). Autonomous transfer for reinforcement learning. In AAMAS (1) (pp. 283-290).
    """
    def __init__(self,
        src_state_var_names:list,
        src_action_names:list,
        src_action_values:list,
        target_state_var_names:list,
        target_action_names:list,
        target_action_values:list,
        src_task_data_folder_and_filename:str,
        neural_networks_folder:str,
        keep_top_k:int=1,
        eval_metric:str='avg_fitness',
        print_debug:bool=False,
        save_output_path:str=None,
        save_every:int=None,
        stats_saver:StatisticsSaver=None,
        stats_folder_path:str=None,
        stats_filename:str=None,
        standard_features:bool=False,
        standard_targets:bool=False        
    ) -> None:
        """
        Description:
            Initializes an instance of the MASTER model.

        Arguments:
            src_state_var_names: the name of the state variables in the source task.
            src_action_names: the name of the actions in the source task.
            src_action_values: the numerical value of the actions in the source task.
            target_state_var_names: the name of the state variables in the target task.
            target_action_names: the name of the actions in the target task.
            target_action_values: the numerical value of the actions in the target task.
            src_task_data_folder_and_filename: the folder and filename of the transition samples in the source task.
            neural_networks_folder: the folder containing the neural networks being used for evaluation.
            keep_top_k: return the top k intertask mapping individuals.
            eval_metric: the metric to determine the fitness of individuals.
                'avg_fitness': average the fitness values across the predicted states and actions.
            print_debug: whether to print debug information.
            save_output_path: the path to save the results
            save_every: save output every mapping that gets processed.
            stats_saver: a StatisticsSaver object to log search data.
            stats_folder_path: the folder path to save the statistics data.
            stats_filename: the name of the statistics file.
            standard_features: whether to standardize the features.
            standard_targets: whether to standardize the targets.       

        Return:
            (None)
        """
        # save class variables
        self.src_state_var_names = src_state_var_names
        self.src_action_names = src_action_names
        self.src_action_values = src_action_values
        self.target_state_var_names = target_state_var_names
        self.target_action_names = target_action_names
        self.target_action_values = target_action_values
        self.keep_top_k = keep_top_k
        self.eval_metric = eval_metric

        # assign a unique code to each state variable and action for efficiency
        self.src_state_codes = {src_state_variable : id for src_state_variable, id in zip(src_state_var_names, range(len(src_state_var_names)))}
        self.src_action_codes = {src_action : id for src_action, id in zip(src_action_names, src_action_values)}
        self.target_state_codes = {target_state_variable : id for target_state_variable, id in zip(target_state_var_names, range(len(target_state_var_names)))}
        self.target_action_codes = {target_action : id for target_action, id in zip(target_action_names, target_action_values)}

        # load config data
        self.config_data = config()

        # transforming src data for mapping evaluation
        src_data_path = src_task_data_folder_and_filename
        self.src_data_df = pd.read_csv(src_data_path, index_col = False)
        self.transformed_df_col_names = self.config_data['3DMC_full_transition_df_col_names']
        self.transformed_df_current_state_cols = self.config_data['3DMC_current_state_transition_df_col_names']
        self.transformed_df_next_state_cols = self.config_data['3DMC_next_state_transition_df_col_names']

        # evaluation using networks
        network_folder_path = neural_networks_folder
        self.eval_networks = EvaluationNetworks(network_folder_path)

        self.print_debug = print_debug
        self.save_output_path = save_output_path
        self.save_every = save_every

        self.stats_saver = stats_saver
        self.stats_out_path = stats_folder_path
        self.stats_filename = stats_filename

        if standard_features:
            feature_scaler = MinMaxScaler()
            feature_names = ['Current-{}'.format(feature) for feature in self.src_state_var_names]
            feature_scaler.fit(self.src_data_df[feature_names])
            self.src_data_df[feature_names] = feature_scaler.transform(self.src_data_df[feature_names])

        if standard_targets:
            target_scaler = MinMaxScaler()
            target_names = ['Next-{}'.format(feature) for feature in self.src_state_var_names]
            target_scaler.fit(self.src_data_df[target_names])
            self.src_data_df[target_names] = target_scaler.transform(self.src_data_df[target_names])

        self.fitness_evaluations = 0

    def transform_source_dataset(self, src_dataset:pd.DataFrame, state_mapping:list, target_action:int, src_action:int, target_col_names:list) -> pd.DataFrame:
        """
        Description:
            Transforms a dataset of transitions from the source task into a dataset of transitions in the target task.

        Arguments:
            src_dataset: the dataset containing transition samples in the source task.
            state_mapping: the state mapping to transform the samples.
            target_action: the target action being mapped.
            src_action: the source action mapped to target_action.
            target_col_names: the names of the columns in the target task (transformed) dataset.

        Return:
            (pd.DataFrame) a dataset of transformed transition samples from the source task.
        """
        # init empty dataframe for transformed src data
        transformed_src_df = pd.DataFrame()

        decoded_state_mapping = {self.target_state_var_names[i] : self.src_state_var_names[j] for i, j in zip(range(len(self.target_state_var_names)), state_mapping)}

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
                src_task_col_name = decoded_state_mapping[state_or_action]
                # are we looking at s or s'?
                if current_or_next == 'Current' or current_or_next == 'Next':
                    transformed_src_df[col_name] = src_dataset[current_or_next + '-' + src_task_col_name]
        
        # create dummy variables to mark multiple mapped actions
        action_dummy_data = np.zeros(shape=(1, len(src_dataset)))
        for row_idx, row in src_dataset.iterrows():
            if int(row['Current-action']) == src_action:
                action_dummy_data[0][row_idx] = 1
        transformed_src_df['Current-action-' + str(target_action)] = action_dummy_data[0]
        
        return transformed_src_df.reset_index(drop=True)

    def evaluate_mapping_bf(
        self,
        target_action:int,
        transformed_df:pd.DataFrame,
        eval_networks:EvaluationNetworks,
        current_state_cols:list,
        next_state_cols:list
    ) -> dict:
        """
        Description:
            Evaluates an intertask mapping using the transformed dataset and the evaluation networks.

        Arguments:
            target_action: the target action being mapped.
            transformed_df: the dataset of transformed source task transitions.
            eval_networks: the set of neural networks to be used for evaluation.
            current_state_cols: the names of the columns that represent the current state.
            next_state_cols: the names of the columns that represent the afterstate.

        Return:
            (dict) a dictionary of scores.
        """
        # init a dictionary to hold the evaluation metrics
        eval_scores = {}
        # filter the dataset by the action
        action_col_name = 'Current-action-' + str(target_action)
        src_df_by_action = transformed_df[transformed_df[action_col_name] == 1]

        # evaluate the mapping's transformed df
        features = src_df_by_action[current_state_cols]
        # we evaluate each afterstate variable independently
        for target_name in next_state_cols:
            target = src_df_by_action[target_name]
            eval_mlp = eval_networks.get_network(target_action, target_name)
            # eval_score = eval_mlp.score(features, target)
            y_pred = eval_mlp.predict(features)
            eval_score = 1 - mean_squared_error(target, y_pred)
            eval_scores['{}--{}'.format(target_action, target_name)] = eval_score
        # count the number of fitness evaluations
        self.fitness_evaluations = self.fitness_evaluations + 1
        # return the dictionary of raw scores
        return eval_scores
        
    def evolve(self) -> list:
        """
        Description:
            Run the MASTER model and generate a list of intertask mappings.

        Arguments:
            (None)

        Return:
            (list) a list of the most fit intertask mappings.
        """
        # if we save data
        if self.save_output_path:
            with open(self.save_output_path, 'w') as f:
                f.write("Results\n")
            str_builder = ""
        i = 0

        population = []
        # enumerate all possible intertask mappings
        try:
            for state_mapping in product(self.src_state_codes.values(), repeat = len(self.target_state_var_names)): # all possible state mappings
            # for state_mapping in [[0, 1, 0, 1]]:
                mapping = IntertaskMappingBf(state_mapping, self.src_state_var_names, self.src_action_names, self.target_state_var_names, self.target_action_names)
                # enumerate all 1-to-1 action mappings
                for target_action_val in self.target_action_values:
                    for src_action_val in self.src_action_values:
                        # evaluate the mapping
                        # use mapping to create transformed df
                        transformed_df = self.transform_source_dataset(self.src_data_df, state_mapping, target_action_val, src_action_val, self.transformed_df_col_names)
                        # evaluate mapping using the transformed df
                        eval_scores = self.evaluate_mapping_bf(target_action_val, transformed_df, self.eval_networks, self.transformed_df_current_state_cols, self.transformed_df_next_state_cols)
                        # consolidate into one single score
                        fitness = parse_mapping_eval_scores(eval_scores)
                        mapping.action_fitness[str(target_action_val) + '--' + str(src_action_val)] = fitness
                        # print debug
                        if self.print_debug:
                            print("State mapping: {}, Target action: {}, Source action: {}, Action fitness: {}".format(state_mapping, target_action_val, src_action_val, fitness))
                        if self.save_output_path:
                            str_builder += "State mapping: {}, Target action: {}, Source action: {}, Action fitness: {}\n".format(state_mapping, target_action_val, src_action_val, fitness)
                        if i % self.save_every == 0:
                            with open(self.save_output_path, 'a') as f:
                                f.write(str_builder)
                                str_builder = ""
                        # if we want to save statistics
                        if self.stats_saver:
                            self.stats_saver.analyze_population_and_log_stats(population, i, 1, self.fitness_evaluations)
                        
                        i += 1

                # save state mapping fitness
                mapping.fitness = np.mean(list(mapping.action_fitness.values()))
                print("State mapping: {}, State fitness: {}".format(state_mapping, mapping.fitness))
                str_builder += "State mapping: {}, State fitness: {}\n".format(state_mapping, mapping.fitness)

                # save mapping to pop
                population.append(mapping)
            
            # return the k individuals with the highest fitness
            best_mapping = sorted(population, key=lambda agent: agent.fitness, reverse=True)[:self.keep_top_k]
            if self.save_output_path:
                if len(best_mapping) == 1:
                    str_builder += "Best mapping: {}, fitness: {}".format(best_mapping[0].state_mapping, best_mapping[0].fitness)
                with open(self.save_output_path, 'a') as f:
                    f.write(str_builder)
            if self.stats_saver:
                self.stats_saver.export_data(self.stats_out_path, self.stats_filename)

            return best_mapping

        except KeyboardInterrupt:
            if self.save_output_path:
                with open(self.save_output_path, 'a') as f:
                    f.write(str_builder)
            if self.stats_saver:
                self.stats_saver.export_data(self.stats_out_path, self.stats_filename)

class GAME_BF:
    """
    Brute force version of GAME.
    """
    def __init__(self,
        src_state_var_names:list,
        src_action_names:list,
        src_action_values:list,
        target_state_var_names:list,
        target_action_names:list,
        target_action_values:list,
        src_task_data_folder_and_filename:str,
        neural_networks_folder:str,
        keep_top_k:int=1,
        eval_metric:str='avg_fitness',
        print_debug:bool=False,
        save_output_path:str=None,
        save_every:int=None,
        stats_saver:StatisticsSaver=None,
        stats_folder_path:str=None,
        stats_filename:str=None,
        standard_features:bool=False,
        standard_targets:bool=False
    ) -> None:
        """
        Description:
            Initializes an instance of the MASTER model.
        Arguments:
            src_state_var_names: the name of the state variables in the source task.
            src_action_names: the name of the actions in the source task.
            src_action_values: the numerical value of the actions in the source task.
            target_state_var_names: the name of the state variables in the target task.
            target_action_names: the name of the actions in the target task.
            target_action_values: the numerical value of the actions in the target task.
            src_task_data_folder_and_filename: the folder and filename of the transition samples in the source task.
            neural_networks_folder: the folder containing the neural networks being used for evaluation.
            keep_top_k: return the top k intertask mapping individuals.
            eval_metric: the metric to determine the fitness of individuals.
                'avg_fitness': average the fitness values across the predicted states and actions.
            print_debug: whether to print debug information.
            save_output_path: the path to save the results
            save_every: save output every mapping that gets processed.
            stats_saver: a StatisticsSaver object to log search data.
            stats_folder_path: the folder path to save the statistics data.
            stats_filename: the name of the statistics file.
            standard_features: whether to standardize the features.
            standard_targets: whether to standardize the targets.  

        Return:
            (None)
        """
        # save class variables
        self.src_state_var_names = src_state_var_names
        self.src_action_names = src_action_names
        self.src_action_values = src_action_values
        self.target_state_var_names = target_state_var_names
        self.target_action_names = target_action_names
        self.target_action_values = target_action_values
        self.keep_top_k = keep_top_k
        self.eval_metric = eval_metric

        # assign a unique code to each state variable and action for efficiency
        self.src_state_codes = {src_state_variable : id for src_state_variable, id in zip(src_state_var_names, range(len(src_state_var_names)))}
        self.src_action_codes = {src_action : id for src_action, id in zip(src_action_names, src_action_values)}
        self.target_state_codes = {target_state_variable : id for target_state_variable, id in zip(target_state_var_names, range(len(target_state_var_names)))}
        self.target_action_codes = {target_action : id for target_action, id in zip(target_action_names, target_action_values)}

        # load config data
        self.config_data = config()

        # transforming src data for mapping evaluation
        src_data_path = src_task_data_folder_and_filename
        self.src_data_df = pd.read_csv(src_data_path, index_col = False)
        self.transformed_df_col_names = self.config_data['3DMC_full_transition_df_col_names']
        self.transformed_df_current_state_cols = self.config_data['3DMC_current_state_transition_df_col_names']
        self.transformed_df_next_state_cols = self.config_data['3DMC_next_state_transition_df_col_names']

        # evaluation using networks
        network_folder_path = neural_networks_folder
        self.eval_networks = EvaluationNetworks(network_folder_path)

        self.print_debug = print_debug
        self.save_output_path = save_output_path
        self.save_every = save_every

        self.stats_saver = stats_saver
        self.stats_out_path = stats_folder_path
        self.stats_filename = stats_filename

        if standard_features:
            feature_scaler = MinMaxScaler()
            feature_names = ['Current-{}'.format(feature) for feature in self.src_state_var_names]
            feature_scaler.fit(self.src_data_df[feature_names])
            self.src_data_df[feature_names] = feature_scaler.transform(self.src_data_df[feature_names])

        if standard_targets:
            target_scaler = MinMaxScaler()
            target_names = ['Next-{}'.format(feature) for feature in self.src_state_var_names]
            target_scaler.fit(self.src_data_df[target_names])
            self.src_data_df[target_names] = target_scaler.transform(self.src_data_df[target_names])

        self.fitness_evaluations = 0

    def evolve(self) -> list:
        """
        Description:
            Run the MASTER model and generate a list of intertask mappings.
        Arguments:
            
        Return:
            (list) a list of the most fit intertask mappings.
        """
        # if we save data
        if self.save_output_path:
            with open(self.save_output_path, 'w') as f:
                f.write("Results\n")
            str_builder = ""
        i = 0

        population = []
        try:
            # enumerate all possible intertask mappings
            for state_mapping in product(self.src_state_codes.values(), repeat = len(self.target_state_var_names)): # all possible state mappings
                for action_mapping in product(self.src_action_codes.values(), repeat = len(self.target_action_values)): # all possible action mappings
                    mapping = IntertaskMapping(state_mapping, action_mapping, self.src_state_var_names, self.src_action_names, self.target_state_var_names, self.target_action_names)
                    # evaluate the mapping
                    # use mapping to create transformed df
                    transformed_df = transform_source_dataset(self.src_data_df, mapping, self.transformed_df_col_names, self.target_action_values)
                    # evaluate mapping using the transformed df
                    eval_scores = evaluate_mapping(mapping, transformed_df, self.eval_networks, self.transformed_df_current_state_cols, self.transformed_df_next_state_cols, self.target_action_values)
                    # count the number of fitness evaluations
                    self.fitness_evaluations = self.fitness_evaluations + 1                
                    # consolidate into one single score
                    mapping.fitness = parse_mapping_eval_scores(eval_scores)[0]
                    population.append(mapping)
                    # print debug
                    if self.print_debug:
                        print('Individual ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}\n'.format(mapping.ID, mapping.state_mapping, mapping.action_mapping, mapping.fitness))
                    if self.save_output_path:
                        str_builder += 'Individual ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}\n'.format(mapping.ID, mapping.state_mapping, mapping.action_mapping, mapping.fitness)
                    if i % self.save_every == 0:
                        with open(self.save_output_path, 'a') as f:
                            f.write(str_builder)
                            str_builder = ""
                    # if we want to save statistics
                    if self.stats_saver:
                        self.stats_saver.analyze_population_and_log_stats(population, i, 1, self.fitness_evaluations)
                    
                    i += 1

            best_mapping = sorted(population, key=lambda agent: agent.fitness, reverse=True)[:self.keep_top_k]

            if self.save_output_path:
                if len(best_mapping) == 1:
                    str_builder += "Best mapping: {}, fitness: {}".format(best_mapping[0].ID, best_mapping[0].fitness)
                with open(self.save_output_path, 'a') as f:
                    f.write(str_builder)
            if self.stats_saver:
                self.stats_saver.export_data(self.stats_out_path, self.stats_filename)

            # return the k individuals with the highest fitness
            return best_mapping
        except KeyboardInterrupt:
            if self.save_output_path:
                with open(self.save_output_path, 'a') as f:
                    f.write(str_builder)
            if self.stats_saver:
                self.stats_saver.export_data(self.stats_out_path, self.stats_filename)

if __name__ == "__main__":
    from GAME.utils.config import config

    config_data = config()

    src_state_var_names = config_data['MC2D_state_names']
    src_action_names = config_data['MC2D_action_names']
    src_action_values = config_data['MC2D_action_values']
    target_state_var_names = config_data['MC3D_state_names']
    target_action_names = config_data['MC3D_action_names']
    target_action_values = config_data['MC3D_action_values']
    # src_task_data_folder_and_filename = os.path.join(config_data['data_path'], "mountain_car", "MC2D_transitions.csv")
    # src_task_data_folder_and_filename = os.path.join(config_data['data_path'], 'mountain_car', "MC2D_transitions_balanced.csv")
    src_task_data_folder_and_filename = os.path.join(config_data['output_path'], '12142022 2DMC Sample Collection 200 Episodes with Training', "2DMC_100_episodes_sample_data_small.csv")
    # neural_networks_folder = os.path.join(config_data['pickle_path'], 'neural_nets', 'mountain_car')
    neural_networks_folder = os.path.join(config_data['pickle_path'], "01072023 3DMC Transition Approx MSE")

    keep_top_k = 1
    eval_metric = 'avg_fitness'
    print_debug = True
    save_output_path  = os.path.join(config_data['output_path'], '01072023 MC GAME-BF', 'results.txt')
    save_every = 1

    search_exp_info = MappingSearchExperimentInfo('MC2D', 'MC3D', 'MASTER', None)
    stats_saver = StatisticsSaver(search_exp_info, 1, True)
    stats_folder_path = os.path.join(config_data['output_path'], '01072023 MC GAME-BF')
    stats_filename = 'stats.txt'
    stats_pickle = 'stats.pickle'

    standard_features = False
    standard_targets = False

    master = GAME_BF(src_state_var_names, src_action_names, src_action_values, target_state_var_names, target_action_names, target_action_values, 
    src_task_data_folder_and_filename, neural_networks_folder, keep_top_k, eval_metric, print_debug, save_output_path, save_every,
    stats_saver, stats_folder_path, stats_filename, standard_features, standard_targets)
    final_mappings = master.evolve()
    for mapping in final_mappings:
        print(mapping)

    with open(os.path.join(stats_folder_path, stats_pickle), 'wb') as f:
        pickle.dump(master.stats_saver, f)