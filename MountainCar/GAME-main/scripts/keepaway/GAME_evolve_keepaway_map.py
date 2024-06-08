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
pop_size = 100
crossover_rate = 0.8
mutation_rate = 0.043
init_strat = 'random'
sel_strat = 'tournament'
tournament_sel_k = int(0.1 * pop_size)
crossover_strat = 'one-pt'
mutation_strat = 'uniform'
replace_strat = 'replace-all-parents'
top_k_elitism = 5
max_fitness_evals = 2000
early_stop = False
early_stop_gen = 10
early_stop_thresh = 10**-3
print_debug = True

save_every = 10
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

standard_features = True
standard_targets = True

count_comparisons = True

# helper variables
# transforming src data
# src_task_data_folder_and_filename = os.path.join(config_data['data_path'], 'keepaway', "keepaway_3v2_transitions.csv")
src_task_data_folder_and_filename = os.path.join(config_data['logs_path'], '12142022_3v2_logs_random_explore', "keepaway_3v2_transitions.csv")
neural_networks_folder = os.path.join(config_data['pickle_path'], '01072023 4v3 Keepaway Transition Approx MSE')

# run for 15 trials
trials = 10
for trial in range(trials):
    print("Trial: {}".format(trial))
    
    # trial-specific params
    output_folder_name = '01072023 Evolve keepaway Maps with GAME 2000 FE One-pt CX 0.043 Mutation MSE'
    save_output_path  = os.path.join(config_data['output_path'], output_folder_name, 'trial{}_population_results.txt'.format(trial))
    stats_saver = StatisticsSaver(search_exp_info, trial, True)
    stats_folder_path = os.path.join(config_data['output_path'], output_folder_name)
    stats_filename = 'trial{}_stats.txt'.format(trial)
    stats_pickle = 'trial{}_stats.pickle'.format(trial)

    # run the GAME model
    ea = GAME(target_task_name, src_state_var_names, src_action_names, src_action_values, target_state_var_names, target_action_names, target_action_values, 
    src_task_data_folder_and_filename, neural_networks_folder, eval_metric, pop_size, crossover_rate, 
    mutation_rate, init_strat, sel_strat, tournament_sel_k, crossover_strat, mutation_strat, replace_strat, top_k_elitism, max_fitness_evals, 
    early_stop, early_stop_gen, early_stop_thresh, print_debug, save_output_path, save_every, stats_saver, stats_folder_path, stats_filename, standard_features, standard_targets, count_comparisons)
    ea.evolve()
    
    # print the final evolved population
    for mapping in ea.population:
        print(mapping)

    with open(os.path.join(stats_folder_path, stats_pickle), 'wb') as f:
        pickle.dump(ea.stats_saver, f)