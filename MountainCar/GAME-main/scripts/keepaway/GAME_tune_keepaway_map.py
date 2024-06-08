# Author: Minh Hua
# Date: 11/11/2022
# Purpose: Runs 10 trials of evolving mappings for Mountain Car.

import os

from GAME.bin.ea import GAME
from GAME.bin.intertask_mappings import *
from GAME.utils.config import config
from GAME.utils.stats_saver import StatisticsSaver, MappingSearchExperimentInfo

# load the config data
config_data = config()

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
pop_sizes = [10, 20, 40]
crossover_rates = [0.8, 0.75, 0.6]
mutation_rate = 0.2
init_strat = 'random'
sel_strat = 'tournament'
# tournament_sel_k = int(0.4 * pop_size)
tournament_sel_ks = [0.25, 0.5, 1]
crossover_strats = ['fusion', 'one-pt']
mutation_strat = 'uniform'
replace_strat = 'replace-all-parents'
top_k_elitisms = [1, 5, 10]
max_fitness_evals = 200
early_stop = False
early_stop_gen = 10
early_stop_thresh = 10**-3
print_debug = False

standard_features = True
standard_targets = True

count_comparisons = True

# helper variables
# transforming src data
src_task_data_folder_and_filename = os.path.join(config_data['data_path'], 'keepaway', "keepaway_3v2_transitions.csv")
neural_networks_folder = os.path.join(config_data['pickle_path'], 'neural_nets', "keepaway")

# run for 15 trials
trial = 1
try:
    for pop_size in pop_sizes:
        for crossover_rate in crossover_rates:
            mutation_rate = 1 - crossover_rate
            for tournament_sel_k in tournament_sel_ks:
                tournament_sel_k = int(tournament_sel_k * pop_size)
                for crossover_strat in crossover_strats:
                    for top_k_elitism in top_k_elitisms:
                        save_every = 1
                        agent_params = {
                            'eval_metric': eval_metric,
                            'pop_size': pop_size,
                            'crossover_rate': crossover_rate,
                            'mutation_rate': mutation_rate,
                            'init_strat': init_strat,
                            'sel_strat': sel_strat,
                            'tournament_sel_k': tournament_sel_k,
                            'crossover_strat': crossover_strat,
                            'mutation_strat': mutation_strat,
                            'replace_strat': replace_strat,
                            'top_k_elitism': top_k_elitism,
                            'max_fitness_evals': max_fitness_evals,
                            'early_stop': early_stop,
                            'early_stop_gen': early_stop_gen,
                            'early_stop_thresh': early_stop_thresh
                        }
                        search_exp_info = MappingSearchExperimentInfo('3v2', '4v3', 'GAME', agent_params)
                        print('Evaluating {}_{}_{}_{}_{}'.format(pop_size, crossover_rate, tournament_sel_k, crossover_strat, top_k_elitism))
                        
                        # trial-specific params
                        output_folder_name = '111622022 Tune keepaway Maps with GAME'
                        save_output_path  = os.path.join(config_data['output_path'], output_folder_name, '{}_{}_{}_{}_{}_population_results.txt'.format(pop_size, crossover_rate, tournament_sel_k, crossover_strat, top_k_elitism))
                        stats_saver = StatisticsSaver(search_exp_info, trial, True)
                        stats_folder_path = os.path.join(config_data['output_path'], output_folder_name)
                        stats_filename = '{}_{}_{}_{}_{}_stats.txt'.format(pop_size, crossover_rate, tournament_sel_k, crossover_strat, top_k_elitism)
                        stats_pickle = '{}_{}_{}_{}_{}_stats.pickle'.format(pop_size, crossover_rate, tournament_sel_k, crossover_strat, top_k_elitism)

                        # run the GAME model
                        ea = GAME(target_task_name, src_state_var_names, src_action_names, src_action_values, target_state_var_names, target_action_names, target_action_values, 
                        src_task_data_folder_and_filename, neural_networks_folder, eval_metric, pop_size, crossover_rate, 
                        mutation_rate, init_strat, sel_strat, tournament_sel_k, crossover_strat, mutation_strat, replace_strat, top_k_elitism, max_fitness_evals, 
                        early_stop, early_stop_gen, early_stop_thresh, print_debug, save_output_path, save_every, stats_saver, stats_folder_path, stats_filename, standard_features, standard_targets, count_comparisons)
                        ea.evolve()

                        # print best ind
                        with open('tuning_results', 'a') as f:
                            f.write('{}_{}_{}_{}_{}\n'.format(pop_size, crossover_rate, tournament_sel_k, crossover_strat, top_k_elitism))
                            f.write('{}\n'.format(sorted(ea.population, key = lambda agent: agent.fitness, reverse=True)[0].fitness))
                        
                        # print the final evolved population
                        # for mapping in ea.population:
                        #     print(mapping)

                        with open(os.path.join(stats_folder_path, stats_pickle), 'wb') as f:
                            pickle.dump(ea.stats_saver, f)
except:
    pass