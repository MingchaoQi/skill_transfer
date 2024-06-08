# Author: Minh Hua
# Date: 11/11/2022
# Purpose: This module contains the hill climbing algorithms that will be used to evolve inter-task mappings.

import random
import numpy as np
import os
from copy import deepcopy

from GAME.bin.intertask_mappings import *
from GAME.utils.config import config
from GAME.utils.stats_saver import StatisticsSaver, MappingSearchExperimentInfo

class GAME_HC:
    """Steepest-ascent hill climbing version of Genetic Algorithms for Mapping Evolution."""
    def __init__(self,
        target_task_name:str,
        src_state_var_names:list,
        src_action_names:list,
        src_action_values:list,
        target_state_var_names:list,
        target_action_names:list,
        target_action_values:list,
        src_task_data_folder_and_filename:str,
        neural_networks_folder:str,
        eval_metric:str='average',
        init_strat:str='random',
        max_fitness_evals:int=1000,
        early_stop:bool=False,
        early_stop_gen:int=5,
        early_stop_thresh:float=10**-4,
        print_debug:bool=False,
        save_output_path:str=None,
        save_every:int=None,
        stats_saver:StatisticsSaver=None,
        stats_folder_path:str=None,
        stats_filename:str=None,
        standard_features:bool=False,
        standard_targets:bool=False,
        count_comparisons:bool=False
    ) -> None:
        """
        Description:
            Initializes an instance of the Genetic Algorithms for Mapping Evolution model.

        Arguments:
            target_task_name: the name of the task. '3DMC' or '4v3'.
            src_state_var_names: the name of the state variables in the source task.
            src_action_names: the name of the actions in the source task.
            src_action_values: the numerical value of the actions in the source task.
            target_state_var_names: the name of the state variables in the target task.
            target_action_names: the name of the actions in the target task.
            target_action_values: the numerical value of the actions in the target task.
            src_task_data_folder_and_filename: the folder and filename of the transition samples in the source task.
            neural_networks_folder: the folder containing the neural networks being used for evaluation.
            eval_metric: the metric to determine the fitness of individuals.
                'average': average the fitness values across the predicted states and actions.
            init_strat: the initialization strategy.
                'random': uniformly initialize the individual chromosomes.
            max_fitness_evals: the maximum number of generations to evolve.
            early_stop: whether or not to stop evolving early.
            early_stop_gen: the number of generations to check for fitness improvement before stopping early.
            early_stop_thresh: the threshold to check whether the best fitness has changed.
            print_debug: whether or not to print debug information.
            save_output_path: the path to save the results
            save_every: save output every mapping that gets processed.
            stats_saver: a StatisticsSaver object to log search data.
            stats_folder_path: the folder path to save the statistics data.
            stats_filename: the name of the statistics file.
            standard_features: whether to standardize the features.
            standard_targets: whether to standardize the targets.
            count_comparisons: whether or not to count the number of comparisons.

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
        self.eval_metric = eval_metric

        # assign a unique code to each state variable and action for efficiency
        self.src_state_codes = {src_state_variable : id for src_state_variable, id in zip(src_state_var_names, range(len(src_state_var_names)))}
        self.src_action_codes = {src_action : id for src_action, id in zip(src_action_names, src_action_values)}
        self.target_state_codes = {target_state_variable : id for target_state_variable, id in zip(target_state_var_names, range(len(target_state_var_names)))}
        self.target_action_codes = {target_action : id for target_action, id in zip(target_action_names, target_action_values)}

        # evolutionary algorithm parameters
        self.pop_size = 1
        self.init_strat = init_strat
        self.max_fitness_evals = max_fitness_evals
        self.early_stop = early_stop
        self.early_stop_gen = early_stop_gen
        self.early_stop_thresh = early_stop_thresh

        # load config data
        self.config_data = config()

        # transforming src data for mapping evaluation
        src_data_path = src_task_data_folder_and_filename
        self.src_data_df = pd.read_csv(src_data_path, index_col = False)
        self.transformed_df_col_names = self.config_data['{}_full_transition_df_col_names'.format(target_task_name)]
        self.transformed_df_current_state_cols = self.config_data['{}_current_state_transition_df_col_names'.format(target_task_name)]
        self.transformed_df_next_state_cols = self.config_data['{}_next_state_transition_df_col_names'.format(target_task_name)]

        # evaluation using networks
        self.network_folder_path = neural_networks_folder
        self.eval_networks = EvaluationNetworks(self.network_folder_path)

        # create a cache that saves fitness evaluations to avoid repeat evaluations
        self.fitness_cache = {}

        self.print_debug = print_debug
        self.save_output_path = save_output_path
        self.save_every = save_every

        self.stats_saver = stats_saver
        self.stats_out_path = stats_folder_path
        self.stats_filename = stats_filename

        self.standard_features = standard_features
        self.standard_targets = standard_targets
        # if standard_features:
        #     feature_scaler = MinMaxScaler()
        #     feature_names = ['Current-{}'.format(feature) for feature in self.src_state_var_names]
        #     feature_scaler.fit(self.src_data_df[feature_names])
        #     self.src_data_df[feature_names] = feature_scaler.transform(self.src_data_df[feature_names])

        # if standard_targets:
        #     target_scaler = MinMaxScaler()
        #     target_names = ['Next-{}'.format(feature) for feature in self.src_state_var_names]
        #     target_scaler.fit(self.src_data_df[target_names])
        #     self.src_data_df[target_names] = target_scaler.transform(self.src_data_df[target_names])

        self.count_comparisons = count_comparisons
        if count_comparisons:
            self.comparisons = []

        self.fitness_evaluations = 0

    def init_pop(self) -> list:
        """
        Description:
            Initializes a population of inter-task mappings.

        Arguments:
            strategy: the strategy used to initialize the population.
                'random': uniformly initialize the individual chromosomes.

        Return:
            (list) the list of initial IntertaskMapping individuals.
        """
        population = []
        if self.init_strat == 'random': # randomly generate mappings. Each mapping is sampled uniformly
            for _ in range(self.pop_size):
                # first generate the state mapping
                state_mapping_chrom = [random.randint(0, len(self.src_state_var_names) - 1) for _ in range(len(self.target_state_var_names))]
                # then generate the action mapping
                action_mapping_chrom = np.random.choice(self.src_action_values, size = len(self.target_action_names), replace = True)
                intertask_mapping_individual = IntertaskMapping(state_mapping_chrom, action_mapping_chrom, self.src_state_var_names, self.src_action_names, self.target_state_var_names, self.target_action_names)
                # append individual to the population
                population.append(intertask_mapping_individual)
                # count comparisons if needed
                if self.count_comparisons:
                    self.comparisons[-1] = self.comparisons[-1] + 1
        # return initial population
        return population

    def evaluate_fitness(self, mapping:IntertaskMapping, set_fitness:float=True) -> float:
        """
        Description:
            Evaluate the fitness of the mapping.

        Arguments:
            mapping: the IntertaskMapping to evaluate the fitness for.
            set_fitness: wether or not we want to set the individual's fitness after evaluation.

        Return:
            (float) the fitness of the IntertaskMapping individual.
        """
        # count the number of fitness evaluations
        self.fitness_evaluations = self.fitness_evaluations + 1
        # if count comparisons
        if self.count_comparisons:
            self.comparisons[-1] = self.comparisons[-1] + 1
        # check to see if the fitness is already computed
        if mapping.ID in self.fitness_cache.keys():
            if self.print_debug:
                print("Fitness cache hit!")
            return self.fitness_cache[mapping.ID]
        else:
            # use mapping to create transformed df
            transformed_df = transform_source_dataset(self.src_data_df, mapping, self.transformed_df_col_names, self.target_action_values)
            # evaluate mapping using the transformed df
            eval_scores = evaluate_mapping(mapping, transformed_df, self.eval_networks, self.transformed_df_current_state_cols, self.transformed_df_next_state_cols, self.target_action_values, self.standard_features, self.standard_targets, self.network_folder_path)
            consolidated_score = parse_mapping_eval_scores(eval_scores, strategy = self.eval_metric)
            # evaluate the mapping's fitness depending on different strategies
            if self.eval_metric == 'average':
                consolidated_score = consolidated_score[0]
            # consolidate into one single score
            if set_fitness:
                mapping.fitness = consolidated_score
            # cache fitness to save computation
            self.fitness_cache[mapping.ID] = consolidated_score               
            return consolidated_score

class GAME_SAHC(GAME_HC):
    """Steepest-ascent hill climbing version of Genetic Algorithms for Mapping Evolution."""
    def __init__(self,
        target_task_name:str,
        src_state_var_names:list,
        src_action_names:list,
        src_action_values:list,
        target_state_var_names:list,
        target_action_names:list,
        target_action_values:list,
        src_task_data_folder_and_filename:str,
        neural_networks_folder:str,
        eval_metric:str='average',
        init_strat:str='random',
        max_fitness_evals:int=1000,
        early_stop:bool=False,
        early_stop_gen:int=5,
        early_stop_thresh:float=10**-4,
        print_debug:bool=False,
        save_output_path:str=None,
        save_every:int=None,
        stats_saver:StatisticsSaver=None,
        stats_folder_path:str=None,
        stats_filename:str=None,
        standard_features:bool=False,
        standard_targets:bool=False,
        count_comparisons:bool=False
    ) -> None:
        """
        Description:
            Initializes an instance of the Genetic Algorithms for Mapping Evolution model.

        Arguments:
            target_task_name: the name of the task. '3DMC' or '4v3'.
            src_state_var_names: the name of the state variables in the source task.
            src_action_names: the name of the actions in the source task.
            src_action_values: the numerical value of the actions in the source task.
            target_state_var_names: the name of the state variables in the target task.
            target_action_names: the name of the actions in the target task.
            target_action_values: the numerical value of the actions in the target task.
            src_task_data_folder_and_filename: the folder and filename of the transition samples in the source task.
            neural_networks_folder: the folder containing the neural networks being used for evaluation.
            eval_metric: the metric to determine the fitness of individuals.
                'average': average the fitness values across the predicted states and actions.
            init_strat: the initialization strategy.
                'random': uniformly initialize the individual chromosomes.
            max_fitness_evals: the maximum number of generations to evolve.
            early_stop: whether or not to stop evolving early.
            early_stop_gen: the number of generations to check for fitness improvement before stopping early.
            early_stop_thresh: the threshold to check whether the best fitness has changed.
            print_debug: whether or not to print debug information.
            save_output_path: the path to save the results
            save_every: save output every mapping that gets processed.
            stats_saver: a StatisticsSaver object to log search data.
            stats_folder_path: the folder path to save the statistics data.
            stats_filename: the name of the statistics file.
            standard_features: whether to standardize the features.
            standard_targets: whether to standardize the targets.
            count_comparisons: whether or not to count the number of comparisons.

        Return:
            (None)
        """
        # inherent from parent class
        super(GAME_SAHC, self).__init__(
            target_task_name,
            src_state_var_names,
            src_action_names,
            src_action_values,
            target_state_var_names,
            target_action_names,
            target_action_values,
            src_task_data_folder_and_filename,
            neural_networks_folder,
            eval_metric,
            init_strat,
            max_fitness_evals,
            early_stop,
            early_stop_gen,
            early_stop_thresh,
            print_debug,
            save_output_path,
            save_every,
            stats_saver,
            stats_folder_path,
            stats_filename,
            standard_features,
            standard_targets,
            count_comparisons
        )

    def sahc_mutation(self, parent_mapping:IntertaskMapping, str_builder:str=None) -> tuple:
        """
        Description:
            Try out all single bit flips and return the offspring with the highest fitness.

        Arguments:
            parent_mapping: the parent mapping to mutate.
            str_builder: the string builder to save the results.

        Return:
            (IntertaskMapping) the best offspring mapping.
        """
        offspring = []
        # try out all single mutation flips
        for state_mapping_idx in range(len(parent_mapping.state_mapping)): # mutate state mapping
            for possible_state_mappings in range(len(self.src_state_var_names)):
                if possible_state_mappings != parent_mapping.state_mapping[state_mapping_idx]:
                    mutated_state_mapping = deepcopy(parent_mapping.state_mapping)
                    mutated_state_mapping[state_mapping_idx] = possible_state_mappings
                    offspring_soln = IntertaskMapping(mutated_state_mapping, parent_mapping.action_mapping, self.src_state_var_names, self.src_action_names, self.target_state_var_names, self.target_action_names)
                    offspring_soln.fitness = self.evaluate_fitness(offspring_soln)
                    offspring.append(offspring_soln)
                    # print debug info
                    if self.print_debug:
                        print('Mutated offspring ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}'.format(offspring_soln.ID, offspring_soln.state_mapping, offspring_soln.action_mapping, offspring_soln.fitness))
                    if self.save_output_path:
                        str_builder += 'Mutated offspring ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}\n'.format(offspring_soln.ID, offspring_soln.state_mapping, offspring_soln.action_mapping, offspring_soln.fitness)                    
                    # if count comparisons
                    if self.count_comparisons:
                        self.comparisons[-1] = self.comparisons[-1] + 1
                    if self.stats_saver:
                        self.stats_saver.log_mapping(offspring_soln)
        for action_mapping_idx in range(len(parent_mapping.action_mapping)): # mutate action mapping
            for possible_action_mappings in self.src_action_values:
                if possible_action_mappings != parent_mapping.action_mapping[action_mapping_idx]:
                    mutated_action_mapping = deepcopy(parent_mapping.action_mapping)
                    mutated_action_mapping[action_mapping_idx] = possible_action_mappings
                    offspring_soln = IntertaskMapping(parent_mapping.state_mapping, mutated_action_mapping, self.src_state_var_names, self.src_action_names, self.target_state_var_names, self.target_action_names)
                    offspring_soln.fitness = self.evaluate_fitness(offspring_soln)
                    offspring.append(offspring_soln)
                    # print debug info
                    if self.print_debug:
                        print('Mutated offspring ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}'.format(offspring_soln.ID, offspring_soln.state_mapping, offspring_soln.action_mapping, offspring_soln.fitness))
                    if self.save_output_path:
                        str_builder += 'Mutated offspring ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}\n'.format(offspring_soln.ID, offspring_soln.state_mapping, offspring_soln.action_mapping, offspring_soln.fitness)
                    # if count comparisons
                    if self.count_comparisons:
                        self.comparisons[-1] = self.comparisons[-1] + 1
                    if self.stats_saver:
                        self.stats_saver.log_mapping(offspring_soln)                                     
        
        # determine the offspring with the best fitness
        return sorted(offspring, key=lambda agent: agent.fitness, reverse=True)[0], str_builder

    def evolve(self) -> list:
        """
        Description:
            Run the hill climber model and generate one intertask mapping.

        Arguments:
            (None)

        Return:
            (list) a list of the most fit intertask mappings.
        """
        try:
            # if we save data
            str_builder = None
            if self.save_output_path:
                with open(self.save_output_path, 'w') as f:
                    f.write("Results\n")
                str_builder = ""

            if self.count_comparisons:
                self.comparisons.append(0)

            # initialize initial population
            self.population = self.init_pop()
            # evaluate initial population's fitness
            for mapping in self.population:
                mapping.fitness = self.evaluate_fitness(mapping)
                # print debug info
                print('Initial Indivial ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}'.format(mapping.ID, mapping.state_mapping, mapping.action_mapping, mapping.fitness))
                if self.save_output_path:
                    str_builder += 'Initial Indivial ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}\n'.format(mapping.ID, mapping.state_mapping, mapping.action_mapping, mapping.fitness)

            # if we want to stop early, we have to keep track of the best fitness
            if self.early_stop:
                best_fitness = [self.population[0]]

            gen = 0

            # main evolution loop
            while self.fitness_evaluations < self.max_fitness_evals:
                best_mutated_offspring, str_builder = self.sahc_mutation(mapping, str_builder)
                if best_mutated_offspring.fitness > mapping.fitness:
                    mapping = best_mutated_offspring                    
                # print debug info
                if self.print_debug:
                    print('Best mutated offspring ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}\n'.format(mapping.ID, mapping.state_mapping, mapping.action_mapping, mapping.fitness))
                if self.save_output_path:
                    str_builder += 'Best mutated offspring ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}\n'.format(mapping.ID, mapping.state_mapping, mapping.action_mapping, mapping.fitness)                   

                # save info, print info, analyze search
                if gen % self.save_every == 0:
                    with open(self.save_output_path, 'a') as f:
                        f.write(str_builder)
                        str_builder = ""
                    if self.stats_saver:
                        self.stats_saver.export_data(self.stats_out_path, self.stats_filename)
                # if we want to save statistics
                if self.stats_saver:
                    self.stats_saver.analyze_population_and_log_stats(self.population, gen, self.comparisons[-1], self.fitness_evaluations)

                # determine early stop if needed
                if self.early_stop:
                    best_fitness.append(mapping)
                    if len(best_fitness) >= self.early_stop_gen:
                        # check to see if the best fitness has changed enough from the average of the past window
                        moving_average = sum(f.fitness for f in best_fitness[-self.early_stop_gen:]) / self.early_stop_gen
                        if self.print_debug:
                            print("Average best fitness of the past {} generations: {}".format(self.early_stop_gen, moving_average))
                        if abs(best_fitness[-1].fitness - moving_average) <= self.early_stop_thresh:
                            # we can stop early
                            if self.print_debug:
                                print("Stopping early.")
                            break

                # reset comparisons count
                if self.count_comparisons:
                    self.comparisons.append(0)
                
                # save mapping to population
                self.population[0] = mapping

                gen += 1

            if self.save_output_path:
                str_builder = "Final population:\n"
                str_builder += 'Offspring ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}\n'.format(mapping.ID, mapping.state_mapping, mapping.action_mapping, mapping.fitness)
                with open(self.save_output_path, 'a') as f:
                    f.write(str_builder)
            if self.stats_saver:
                self.stats_saver.export_data(self.stats_out_path, self.stats_filename)

        except KeyboardInterrupt:
            if self.save_output_path:
                with open(self.save_output_path, 'a') as f:
                    f.write(str_builder)
            if self.stats_saver:
                self.stats_saver.export_data(self.stats_out_path, self.stats_filename)

class GAME_RMHC(GAME_HC):
    """Random-mutation hill climbing version of Genetic Algorithms for Mapping Evolution."""
    def __init__(self,
        target_task_name:str,
        src_state_var_names:list,
        src_action_names:list,
        src_action_values:list,
        target_state_var_names:list,
        target_action_names:list,
        target_action_values:list,
        src_task_data_folder_and_filename:str,
        neural_networks_folder:str,
        eval_metric:str='average',
        init_strat:str='random',
        max_evol_gen:int=1000,
        early_stop:bool=False,
        early_stop_gen:int=5,
        early_stop_thresh:float=10**-4,
        print_debug:bool=False,
        save_output_path:str=None,
        save_every:int=None,
        stats_saver:StatisticsSaver=None,
        stats_folder_path:str=None,
        stats_filename:str=None,
        standard_features:bool=False,
        standard_targets:bool=False,
        count_comparisons:bool=False
    ) -> None:
        """
        Description:
            Initializes an instance of the Genetic Algorithms for Mapping Evolution model.

        Arguments:
            target_task_name: the name of the task. '3DMC' or '4v3'.
            src_state_var_names: the name of the state variables in the source task.
            src_action_names: the name of the actions in the source task.
            src_action_values: the numerical value of the actions in the source task.
            target_state_var_names: the name of the state variables in the target task.
            target_action_names: the name of the actions in the target task.
            target_action_values: the numerical value of the actions in the target task.
            src_task_data_folder_and_filename: the folder and filename of the transition samples in the source task.
            neural_networks_folder: the folder containing the neural networks being used for evaluation.
            eval_metric: the metric to determine the fitness of individuals.
                'average': average the fitness values across the predicted states and actions.
            init_strat: the initialization strategy.
                'random': uniformly initialize the individual chromosomes.
            max_evol_gen: the maximum number of generations to evolve.
            early_stop: whether or not to stop evolving early.
            early_stop_gen: the number of generations to check for fitness improvement before stopping early.
            early_stop_thresh: the threshold to check whether the best fitness has changed.
            print_debug: whether or not to print debug information.
            save_output_path: the path to save the results
            save_every: save output every mapping that gets processed.
            stats_saver: a StatisticsSaver object to log search data.
            stats_folder_path: the folder path to save the statistics data.
            stats_filename: the name of the statistics file.
            standard_features: whether to standardize the features.
            standard_targets: whether to standardize the targets.
            count_comparisons: whether or not to count the number of comparisons.

        Return:
            (None)
        """
        # inherent from parent class
        super(GAME_RMHC, self).__init__(
            target_task_name,
            src_state_var_names,
            src_action_names,
            src_action_values,
            target_state_var_names,
            target_action_names,
            target_action_values,
            src_task_data_folder_and_filename,
            neural_networks_folder,
            eval_metric,
            init_strat,
            max_evol_gen,
            early_stop,
            early_stop_gen,
            early_stop_thresh,
            print_debug,
            save_output_path,
            save_every,
            stats_saver,
            stats_folder_path,
            stats_filename,
            standard_features,
            standard_targets,
            count_comparisons
        )

    def rmhc_mutate(self, individual:IntertaskMapping, str_builder:str=None) -> IntertaskMapping:
        """
        Description:
           Mutate an IntertaskMapping individual.

        Arguments:
            individual: the IntertaskMapping individual to mutate.
            str_builder: string builder to save output.

        Return:
            (IntertaskMapping) the mutated IntertaskMapping individual.
        """
        mutated_state_mapping = deepcopy(individual.state_mapping)
        mutated_action_mapping = deepcopy(individual.action_mapping)
        # mutate a random state mapping
        random_state_idx = np.random.choice(len(mutated_state_mapping), size = 1)[0]
        random_action_idx = np.random.choice(len(mutated_action_mapping), size = 1)[0]
        mutated_state_mapping[random_state_idx] = np.random.choice(len(self.src_state_var_names), size = 1)[0]
        mutated_action_mapping[random_action_idx] = np.random.choice(self.src_action_values, size = 1)[0]
        mutated_offspring = IntertaskMapping(mutated_state_mapping, mutated_action_mapping, self.src_state_var_names, self.src_action_names, self.target_state_var_names, self.target_action_names)
        mutated_offspring.fitness = self.evaluate_fitness(mutated_offspring)

        # print debug info
        if self.print_debug:
            print('Mutated offspring ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}'.format(mutated_offspring.ID, mutated_offspring.state_mapping, mutated_offspring.action_mapping, mutated_offspring.fitness))
        if self.save_output_path:
            str_builder += 'Mutated offspring ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}\n'.format(mutated_offspring.ID, mutated_offspring.state_mapping, mutated_offspring.action_mapping, mutated_offspring.fitness)         
        # if count comparisons
        if self.count_comparisons:
            self.comparisons[-1] = self.comparisons[-1] + 1        
        if self.stats_saver:
            self.stats_saver.log_mapping(mutated_offspring)
        
        # return the mutated individual
        return mutated_offspring, str_builder

    def evolve(self) -> list:
        """
        Description:
            Run the hill climber model and generate one intertask mapping.

        Arguments:
            (None)

        Return:
            (list) a list of the most fit intertask mappings.
        """
        try:
            # if we save data
            str_builder = None
            if self.save_output_path:
                with open(self.save_output_path, 'w') as f:
                    f.write("Results\n")
                str_builder = ""

            if self.count_comparisons:
                self.comparisons.append(0)

            # initialize initial population
            self.population = self.init_pop()
            # evaluate initial population's fitness
            for mapping in self.population:
                mapping.fitness = self.evaluate_fitness(mapping)
                # print debug info
                print('Initial Indivial ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}'.format(mapping.ID, mapping.state_mapping, mapping.action_mapping, mapping.fitness))
                if self.save_output_path:
                    str_builder += 'Initial Indivial ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}\n'.format(mapping.ID, mapping.state_mapping, mapping.action_mapping, mapping.fitness)

            # if we want to stop early, we have to keep track of the best fitness
            if self.early_stop:
                best_fitness = [self.population[0]]

            gen = 0

            # main evolution loop
            while self.fitness_evaluations < self.max_fitness_evals:
                best_mutated_offspring, str_builder = self.rmhc_mutate(mapping, str_builder)
                if best_mutated_offspring.fitness > mapping.fitness:
                    mapping = best_mutated_offspring                    
                # print debug info
                if self.print_debug:
                    print('Best mutated offspring ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}\n'.format(mapping.ID, mapping.state_mapping, mapping.action_mapping, mapping.fitness))
                if self.save_output_path:
                    str_builder += 'Best mutated offspring ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}\n'.format(mapping.ID, mapping.state_mapping, mapping.action_mapping, mapping.fitness)                   

                # save info, print info, analyze search
                if gen % self.save_every == 0:
                    with open(self.save_output_path, 'a') as f:
                        f.write(str_builder)
                        str_builder = ""
                    if self.stats_saver:
                        self.stats_saver.export_data(self.stats_out_path, self.stats_filename)
                # if we want to save statistics
                if self.stats_saver:
                    self.stats_saver.analyze_population_and_log_stats(self.population, gen, self.comparisons[-1], self.fitness_evaluations)

                # determine early stop if needed
                if self.early_stop:
                    best_fitness.append(mapping)
                    if len(best_fitness) >= self.early_stop_gen:
                        # check to see if the best fitness has changed enough from the average of the past window
                        moving_average = sum(f.fitness for f in best_fitness[-self.early_stop_gen:]) / self.early_stop_gen
                        if self.print_debug:
                            print("Average best fitness of the past {} generations: {}".format(self.early_stop_gen, moving_average))
                        if abs(best_fitness[-1].fitness - moving_average) <= self.early_stop_thresh:
                            # we can stop early
                            if self.print_debug:
                                print("Stopping early.")
                            break

                # reset comparisons count
                if self.count_comparisons:
                    self.comparisons.append(0)
                
                # save mapping to population
                self.population[0] = mapping

                gen += 1

            if self.save_output_path:
                str_builder = "Final population:\n"
                str_builder += 'Offspring ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}\n'.format(mapping.ID, mapping.state_mapping, mapping.action_mapping, mapping.fitness)
                with open(self.save_output_path, 'a') as f:
                    f.write(str_builder)
            if self.stats_saver:
                self.stats_saver.export_data(self.stats_out_path, self.stats_filename)

        except KeyboardInterrupt:
            if self.save_output_path:
                with open(self.save_output_path, 'a') as f:
                    f.write(str_builder)
            if self.stats_saver:
                self.stats_saver.export_data(self.stats_out_path, self.stats_filename)

if __name__ == '__main__':
    # load the config data
    config_data = config()

    test = 0

    if test == 0: # 3DMC test
        # variables to identify the task
        target_task_name = '3DMC'
        src_state_var_names = config_data['MC2D_state_names']
        src_action_names = config_data['MC2D_action_names']
        src_action_values = config_data['MC2D_action_values']
        target_state_var_names = config_data['MC3D_state_names']
        target_action_names = config_data['MC3D_action_names']
        target_action_values = config_data['MC3D_action_values']

        # evolution parameters
        eval_metric = 'average'
        init_strat = 'random'
        max_fitness_evals = 1000
        early_stop = True
        early_stop_gen = 10
        early_stop_thresh = 10**-3
        print_debug = True

        save_output_path  = os.path.join(config_data['output_path'], '11112022 MC EA', 'MC_HC_results.txt')
        save_every = 1

        search_exp_info = MappingSearchExperimentInfo('2DMC', '3DMC', 'GAME', None)
        stats_saver = StatisticsSaver(search_exp_info, 1, True)
        stats_folder_path = os.path.join(config_data['output_path'], '11112022 MC EA')
        stats_filename = 'MC_HC_stats.txt'
        stats_pickle = 'MC_HC_stats.pickle'

        standard_features = False
        standard_targets = False

        count_comparisons = True

        # helper variables
        # transforming src data
        src_task_data_folder_and_filename = os.path.join("11032022 2DMC Sample Collection 100 Episodes with Training", "2DMC_100_episodes_sample_data.csv")
        neural_networks_folder = "11012022 3DMC Neural Nets"

        hc = GAME_SAHC(target_task_name, src_state_var_names, src_action_names, src_action_values, target_state_var_names, target_action_names, target_action_values, 
        src_task_data_folder_and_filename, neural_networks_folder, eval_metric, init_strat, max_fitness_evals, 
        early_stop, early_stop_gen, early_stop_thresh, print_debug, save_output_path, save_every, stats_saver, stats_folder_path, stats_filename, standard_features, standard_targets, count_comparisons)
        # run the GAME model
        hc.evolve()
        # print the final evolved population
        for mapping in hc.population:
            print(mapping)

        with open(os.path.join(stats_folder_path, stats_pickle), 'wb') as f:
            pickle.dump(hc.stats_saver, f)
    
    elif test == 1: # 4v3 Keepaway test
        # variables to identify the task
        target_task_name = '4v3'
        src_state_var_names = config_data['3v2_state_names']
        src_action_names = config_data['3v2_action_names']
        src_action_values = config_data['3v2_action_values']
        target_state_var_names = config_data['4v3_state_names']
        target_action_names = config_data['4v3_action_names']
        target_action_values = config_data['4v3_action_values']

        # evolution parameters
        eval_metric = 'average'
        init_strat = 'random'
        max_fitness_evals = 1000
        early_stop = True
        early_stop_gen = 10
        early_stop_thresh = 10**-4
        print_debug = True

        save_output_path  = os.path.join(config_data['output_path'], '11112022 MC EA', 'keepaway_HC_results.txt')
        save_every = 1

        search_exp_info = MappingSearchExperimentInfo('3v2', '4v3', 'GAME', None)
        stats_saver = StatisticsSaver(search_exp_info, 1, True)
        stats_folder_path = os.path.join(config_data['output_path'], '11112022 MC EA')
        stats_filename = 'keepaway_HC_stats.txt'
        stats_pickle = 'keepaway_HC_stats.pickle'

        standard_features = True
        standard_targets = True

        count_comparisons = True        

        # helper variables
        # transforming src data
        src_task_data_folder_and_filename = os.path.join('11102022 3v2 10x350 eps learned', "keepaway_3v2_transitions_v3.csv")
        neural_networks_folder = "11102022 4v3 Neural Nets"

        hc = GAME_RMHC(target_task_name, src_state_var_names, src_action_names, src_action_values, target_state_var_names, target_action_names, target_action_values, 
        src_task_data_folder_and_filename, neural_networks_folder, eval_metric, init_strat, max_fitness_evals, 
        early_stop, early_stop_gen, early_stop_thresh, print_debug, save_output_path, save_every, stats_saver, stats_folder_path, stats_filename, standard_features, standard_targets, count_comparisons)
        # run the GAME model
        hc.evolve()
        # print the final evolved population
        for mapping in hc.population:
            print(mapping)

        with open(os.path.join(stats_folder_path, stats_pickle), 'wb') as f:
            pickle.dump(hc.stats_saver, f)