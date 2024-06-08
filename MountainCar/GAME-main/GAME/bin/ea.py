# Author: Minh Hua
# Date: 11/1/2022
# Purpose: This module contains the main evolutionary algorithm that will be used to evolve inter-task mappings.

import random
import numpy as np
import os

from GAME.bin.intertask_mappings import *
from GAME.utils.config import config
from GAME.utils.stats_saver import StatisticsSaver, MappingSearchExperimentInfo

class GAME:
    """Genetic Algorithms for Mapping Evolution evolves a population of inter-task mapping for transfer learning in reinforcement learning."""
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
        pop_size:int=1000,
        crossover_rate:float=0.8,
        mutation_rate:float=0.2,
        init_strat:str='random',
        sel_strat:str='tournament',
        tournament_sel_k:int=5,
        crossover_strat:str='one-pt',
        mutation_strat:str='uniform',
        replace_strat:str='replace-all-parents',
        top_k_elitism:int=None,
        max_fitness_evals:int=10000,
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
            pop_size: the size of the population.
            crossover_rate: the crossover rate.
            mutation_rate: the mutation rate.
            init_strat: the initialization strategy.
                'random': uniformly initialize the individual chromosomes.
            sel_strat: the crossover strategy.
                'tournament': tournament selection.
            tournament_sel_k: the number of individuals to include in a tournament.
            crossover_strat: the crossover strategy.
                'one-pt': one-point crossover.
            mutation_strat: the mutation strategy.
                'uniform': uniform mutation.
            replace_strat: the population replacement strategy.
                'replace-all-parents':
            top_k_elitism: top-k elitism.
            max_fitness_evals: the maximum number of fitness evaluations.
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
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.init_strat = init_strat
        self.sel_strat = sel_strat
        self.tournament_sel_k = tournament_sel_k
        self.crossover_strat = crossover_strat
        self.mutation_strat = mutation_strat
        self.replace_strat = replace_strat
        self.top_k_elitism = top_k_elitism
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
        # count the number of fitness evaluations
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

    def select_parents(self) -> IntertaskMapping:
        """
        Description:
            Select parents for crossover according to some strategy.

        Arguments:
            strategy: the strategy used to crossover the parents.
                'tournament': tournament selection.
                'fitness-proportionate': fitness-proportionate selection.

        Return:
            (IntertaskMapping) a single IntertaskMapping parent chosen from a selection method.
        """
        if self.sel_strat == 'tournament':
            parents = random.choices(self.population, k = self.tournament_sel_k)
            parent = sorted(parents, key = lambda agent: agent.fitness, reverse=True)
            parent = parent[0]
        elif self.sel_strat == 'fitness-proportionate':
            r = random.random() * sum(mapping.fitness for mapping in self.population)
            x = 0
            for mapping in self.population:
                x += mapping.fitness
                if r <= x:
                    parent = mapping
                    break
        # if count comparisons
        if self.count_comparisons:
            self.comparisons[-1] = self.comparisons[-1] + self.pop_size
        return parent # return parent

    def crossover(self, parent_1:IntertaskMapping, parent_2:IntertaskMapping) -> list:
        """
        Description:
            Generate a number of offspring using certain crossover strategies.

        Arguments:
            parent_1: the first parent.
            parent_2: the second parent.
            strategy: the strategy used to crossover the parents.
                'one-pt': one-point crossover.
                'two-pt': two-point crossover.
                'fusion': select the bit from the higher-fitness parent.

        Return:
            (list) a list of IntertaskMapping offspring.
        """
        offspring = []
        if self.crossover_strat == 'one-pt':
            # randomly select a point to crossover the parents
            # crossover the state mapping first
            state_crossover_pt = np.random.choice(len(parent_1.state_mapping), size = 1)[0]
            offspring_1_state_mapping = list(parent_1.state_mapping[:state_crossover_pt]) + list(parent_2.state_mapping[state_crossover_pt:])
            offspring_2_state_mapping = list(parent_2.state_mapping[:state_crossover_pt]) + list(parent_1.state_mapping[state_crossover_pt:])
            # crossover the action mapping
            action_crossover_pt = np.random.choice(len(parent_1.action_mapping), size = 1)[0]
            offspring_1_action_mapping = list(parent_1.action_mapping[:action_crossover_pt]) + list(parent_2.action_mapping[action_crossover_pt:])
            offspring_2_action_mapping = list(parent_2.action_mapping[:action_crossover_pt]) + list(parent_1.action_mapping[action_crossover_pt:])
            # create and append the offspring 
            offspring_1 = IntertaskMapping(offspring_1_state_mapping, offspring_1_action_mapping, self.src_state_var_names, self.src_action_names, self.target_state_var_names, self.target_action_names)
            offspring.append(offspring_1)
            offspring_2 = IntertaskMapping(offspring_2_state_mapping, offspring_2_action_mapping, self.src_state_var_names, self.src_action_names, self.target_state_var_names, self.target_action_names)
            offspring.append(offspring_2)
        elif self.crossover_strat == 'two-pt':
            # randomly select points to crossover the parents
            # crossover the state mapping first
            state_crossover_pts = sorted(np.random.choice(len(parent_1.state_mapping), size = 2, replace = False))
            offspring_1_state_mapping = parent_1.state_mapping[:state_crossover_pts[0]] + parent_2.state_mapping[state_crossover_pts[0]:state_crossover_pts[1]] + parent_1.state_mapping[state_crossover_pts[1]:]
            offspring_2_state_mapping = parent_2.state_mapping[:state_crossover_pts[0]] + parent_1.state_mapping[state_crossover_pts[0]:state_crossover_pts[1]] + parent_2.state_mapping[state_crossover_pts[1]:]
            # crossover the action mapping
            action_crossover_pts = np.random.choice(len(parent_1.action_mapping), size = 2, replace = False)
            offspring_1_action_mapping = parent_1.state_mapping[:action_crossover_pts[0]] + parent_2.state_mapping[action_crossover_pts[0]:action_crossover_pts[1]] + parent_1.state_mapping[action_crossover_pts[1]:]
            offspring_2_action_mapping = parent_2.state_mapping[:action_crossover_pts[0]] + parent_1.state_mapping[action_crossover_pts[0]:action_crossover_pts[1]] + parent_2.state_mapping[action_crossover_pts[1]:]
            # create and append the offspring 
            offspring_1 = IntertaskMapping(offspring_1_state_mapping, offspring_1_action_mapping, self.src_state_var_names, self.src_action_names, self.target_state_var_names, self.target_action_names)
            offspring.append(offspring_1)
            offspring_2 = IntertaskMapping(offspring_2_state_mapping, offspring_2_action_mapping, self.src_state_var_names, self.src_action_names, self.target_state_var_names, self.target_action_names)
            offspring.append(offspring_2)
        elif self.crossover_strat == 'fusion':
            # set the child's bit to be the bit belonging to the parent with the higher fitness
            p = parent_1.fitness / (parent_1.fitness + parent_2.fitness) # probablistically select the parent with the higher fitness
            # copy the state mapping first
            state_mapping = []
            for parent_1_bit, parent_2_bit in zip(parent_1.state_mapping, parent_2.state_mapping):
                if parent_1_bit == parent_2_bit:
                    state_mapping.append(parent_1_bit)
                else:
                    if random.random() < p: # select P1
                        state_mapping.append(parent_1_bit)
                    else: # select P2
                        state_mapping.append(parent_2_bit)
            # copy the action mapping last
            action_mapping = []
            for parent_1_bit, parent_2_bit in zip(parent_1.action_mapping, parent_2.action_mapping):
                if parent_1_bit == parent_2_bit:
                    action_mapping.append(parent_1_bit)
                else:
                    if random.random() < p: # select P1
                        action_mapping.append(parent_1_bit)
                    else: # select P2
                        action_mapping.append(parent_2_bit)
            offspring.append(IntertaskMapping(state_mapping, action_mapping, self.src_state_var_names, self.src_action_names, self.target_state_var_names, self.target_action_names))
        # if count comparisons
        if self.count_comparisons:
            self.comparisons[-1] = self.comparisons[-1] + 1
        # return the children
        return offspring

    def mutate(self, individual:IntertaskMapping) -> IntertaskMapping:
        """
        Description:
           Mutate an IntertaskMapping individual.

        Arguments:
            individual: the IntertaskMapping individual to mutate.
            strategy: the strategy used to mutate the individual.
                'uniform': mutate each gene with equal probability.
                'weighted': 

        Return:
            (IntertaskMapping) the mutated IntertaskMapping individual.
        """
        mutated_state_mapping = []
        mutated_action_mapping = []
        if self.mutation_strat == 'uniform':
            for state_map in individual.state_mapping: # mutate state mapping
                if random.random() < self.mutation_rate:
                    mutated_state_mapping.append(random.randint(0, len(self.src_state_var_names) - 1))
                else:
                    mutated_state_mapping.append(state_map)
            for action_map in individual.action_mapping: # mutate action mapping
                if random.random() < self.mutation_rate:
                    mutated_action_mapping.append(random.randint(0, len(self.src_action_names) - 1))
                else:
                    mutated_action_mapping.append(action_map)
        # if count comparisons
        if self.count_comparisons:
            self.comparisons[-1] = self.comparisons[-1] + 1        
        # return the mutated individual
        return IntertaskMapping(mutated_state_mapping, mutated_action_mapping, self.src_state_var_names, self.src_action_names, self.target_state_var_names, self.target_action_names)

    def replace(self, offspring:list) -> list:
        """
        Description:
            Replace the current population with the new offspring population using different strategies.

        Arguments:
            offspring: a list of the current offspring.
            strategy: the replacement strategy.
                'replace-all-parents': canonical GA replacement strategy where the offspring replaces the parent population.

        Return:
            (list) a list of the new population.
        """
        if self.replace_strat == 'replace-all-parents':
            return offspring

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

    def determine_best_fit(self, population:list) -> IntertaskMapping:
        """
        Description:
            Determine the individual with the best fitness in the population.

        Arguments:
            population: the current population to evaluate the fitness for.

        Return:
            (IntertaskMapping) the most fit individual.
        """
        # if count comparisons
        if self.count_comparisons:
            self.comparisons[-1] = self.comparisons[-1] + self.pop_size
        return sorted(population, key = lambda agent: agent.fitness, reverse=True)[0]

    def evolve(self) -> list:
        """
        Description:
            Run the GAME model and generate a list of intertask mappings.

        Arguments:
            

        Return:
            (list) a list of the most fit intertask mappings.
        """
        try:
            # if we save data
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
                best_fitness = [self.determine_best_fit(self.population)]

            gen = 0

            # main evolution loop
            while self.fitness_evaluations < self.max_fitness_evals:
                if self.print_debug:
                    print("Generation {}".format(gen))

                # elitism
                if self.top_k_elitism:
                    # save top k individuals from parent population
                    top_k = sorted(self.population, key=lambda agent: agent.fitness, reverse=True)[:self.top_k_elitism]
                    # if count comparisons
                    if self.count_comparisons:
                        self.comparisons[-1] = self.comparisons[-1] + self.pop_size 
                        
                offspring = []
                # generate a number of offspring
                while len(offspring) < self.pop_size:
                    # select parents for crossover
                    parent_1 = self.select_parents()
                    parent_2 = self.select_parents()
                    # make sure we do not have identical parents
                    while parent_2.ID == parent_1.ID:
                        parent_2 = self.select_parents()
                    if random.random() < self.crossover_rate:
                        # generate offspring using crossover
                        new_offspring = self.crossover(parent_1, parent_2)
                    else: # offspring are exact copies of the parents
                        new_offspring = [parent_1, parent_2]
                    # mutate offspring
                    for offspring_idx in range(len(new_offspring)):
                        new_offspring[offspring_idx] = self.mutate(new_offspring[offspring_idx])
                    # evaluate offspring fitness
                    for offspring_soln in new_offspring:
                        offspring_soln.fitness = self.evaluate_fitness(offspring_soln)
                        # add offspring to temporary offspring array
                        offspring.append(offspring_soln)
                    # print debug info
                    if self.print_debug:
                        for offspring_soln in new_offspring:
                            print('Offspring ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}'.format(offspring_soln.ID, offspring_soln.state_mapping, offspring_soln.action_mapping, offspring_soln.fitness))
                    if self.save_output_path:
                        for offspring_soln in new_offspring:                    
                            str_builder += 'Offspring ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}\n'.format(offspring_soln.ID, offspring_soln.state_mapping, offspring_soln.action_mapping, offspring_soln.fitness)                   

                # replace population with offspring
                self.population = self.replace(offspring)

                # elitism
                if self.top_k_elitism:
                    self.population = sorted(self.population + top_k, key=lambda agent: agent.fitness, reverse=True)[:self.pop_size]
                    # if count comparisons
                    if self.count_comparisons:
                        self.comparisons[-1] = self.comparisons[-1] + self.pop_size                    

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
                    best_fitness.append(self.determine_best_fit(self.population))
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
                
                gen += 1

            if self.save_output_path:
                str_builder = "Final population:\n"
                for ind in self.population:
                    str_builder += 'Offspring ID: {}, State mapping: {}, Action mapping: {}, Fitness: {}\n'.format(ind.ID, ind.state_mapping, ind.action_mapping, ind.fitness)
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
        pop_size = 5
        crossover_rate = 0.8
        mutation_rate = 0.2
        init_strat = 'random'
        sel_strat = 'tournament'
        tournament_sel_k = int(0.25 * pop_size)
        crossover_strat = 'one-pt'
        mutation_strat = 'uniform'
        replace_strat = 'replace-all-parents'
        top_k_elitism = None
        max_fitness_evals = 2000
        early_stop = True
        early_stop_gen = 3
        early_stop_thresh = 10**-3
        print_debug = True

        save_output_path  = os.path.join(config_data['output_path'], '11112022 MC EA', 'MC_results.txt')
        save_every = 1

        search_exp_info = MappingSearchExperimentInfo('2DMC', '3DMC', 'GAME', None)
        stats_saver = StatisticsSaver(search_exp_info, 1, True)
        stats_folder_path = os.path.join(config_data['output_path'], '11112022 MC EA')
        stats_filename = 'MC_stats.txt'
        stats_pickle = 'MC_stats.pickle'

        standard_features = False
        standard_targets = False

        count_comparisons = True

        # helper variables
        # transforming src data
        src_task_data_folder_and_filename = os.path.join(config_data['output_path'], "11032022 2DMC Sample Collection 100 Episodes with Training", "2DMC_100_episodes_sample_data.csv")
        neural_networks_folder = os.path.join(config_data['pickle_path'], "11012022 3DMC Neural Nets")

        ea = GAME(target_task_name, src_state_var_names, src_action_names, src_action_values, target_state_var_names, target_action_names, target_action_values, 
        src_task_data_folder_and_filename, neural_networks_folder, eval_metric, pop_size, crossover_rate, 
        mutation_rate, init_strat, sel_strat, tournament_sel_k, crossover_strat, mutation_strat, replace_strat, top_k_elitism, max_fitness_evals, 
        early_stop, early_stop_gen, early_stop_thresh, print_debug, save_output_path, save_every, stats_saver, stats_folder_path, stats_filename, standard_features, standard_targets, count_comparisons)
        # run the GAME model
        ea.evolve()
        # print the final evolved population
        for mapping in ea.population:
            print(mapping)

        with open(os.path.join(stats_folder_path, stats_pickle), 'wb') as f:
            pickle.dump(ea.stats_saver, f)
    
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
        pop_size = 20
        crossover_rate = 0.8
        mutation_rate = 0.2
        init_strat = 'random'
        sel_strat = 'tournament'
        tournament_sel_k = int(0.25 * pop_size)
        crossover_strat = 'fusion'
        mutation_strat = 'uniform'
        replace_strat = 'replace-all-parents'
        top_k_elitism = 5
        max_fitness_evals = 2000
        early_stop = True
        early_stop_gen = 5
        early_stop_thresh = 10**-4
        print_debug = True

        save_output_path  = os.path.join(config_data['output_path'], '11112022 MC EA', 'keepaway_results.txt')
        save_every = 1

        search_exp_info = MappingSearchExperimentInfo('3v2', '4v3', 'GAME', None)
        stats_saver = StatisticsSaver(search_exp_info, 1, True)
        stats_folder_path = os.path.join(config_data['output_path'], '11112022 MC EA')
        stats_filename = 'stats.txt'
        stats_pickle = 'stats.pickle'

        standard_features = True
        standard_targets = True

        count_comparisons = True        

        # helper variables
        # transforming src data
        src_task_data_folder_and_filename = os.path.join('11102022 3v2 10x350 eps learned', "keepaway_3v2_transitions_v3.csv")
        neural_networks_folder = "11102022 4v3 Neural Nets"

        ea = GAME(target_task_name, src_state_var_names, src_action_names, src_action_values, target_state_var_names, target_action_names, target_action_values, 
        src_task_data_folder_and_filename, neural_networks_folder, eval_metric, pop_size, crossover_rate, 
        mutation_rate, init_strat, sel_strat, tournament_sel_k, crossover_strat, mutation_strat, replace_strat, top_k_elitism, max_fitness_evals, 
        early_stop, early_stop_gen, early_stop_thresh, print_debug, save_output_path, save_every, stats_saver, stats_folder_path, stats_filename, standard_features, standard_targets, count_comparisons)
        # run the GAME model
        ea.evolve()
        # print the final evolved population
        print("Best mappings: ")
        for mapping in ea.population:
            print(mapping)

        with open(os.path.join(stats_folder_path, stats_pickle), 'wb') as f:
            pickle.dump(ea.stats_saver, f)