# Author: Minh Hua
# Date: 11/11/2022
# Purpose: Runs 10 trials of evolving mappings for Mountain Car.

import os

from GAME.bin.hill_climber import GAME_SAHC
from GAME.bin.intertask_mappings import *
from GAME.utils.config import config
from GAME.utils.stats_saver import StatisticsSaver, MappingSearchExperimentInfo

# load the config data
config_data = config()

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
max_fitness_evals = 240
early_stop = True
early_stop_gen = 3
early_stop_thresh = 10**-3
print_debug = True

save_every = 1
agent_params = {
    'eval_metric': eval_metric,
    'init_strat': init_strat,
    'max_fitness_evals': max_fitness_evals,
    'early_stop': early_stop,
    'early_stop_gen': early_stop_gen,
    'early_stop_thresh': early_stop_thresh
}
search_exp_info = MappingSearchExperimentInfo('2DMC', '3DMC', 'GAME', agent_params)

standard_features = False
standard_targets = False

count_comparisons = True

# helper variables
# transforming src data
src_task_data_folder_and_filename = os.path.join(config_data['data_path'], 'mountain_car', "MC2D_transitions.csv")
neural_networks_folder = os.path.join(config_data['pickle_path'], 'neural_nets', "mountain_car")

# run for 15 trials
trials = 10
for trial in range(trials):
    print("Trial: {}".format(trial))
    
    # trial-specific params
    output_folder_name = '11112022 Evolve MC Maps with GAME_SAHC Best Early Stop'
    save_output_path  = os.path.join(config_data['output_path'], output_folder_name, 'trial{}_population_results.txt'.format(trial))
    stats_saver = StatisticsSaver(search_exp_info, trial, True)
    stats_folder_path = os.path.join(config_data['output_path'], output_folder_name)
    stats_filename = 'MC_trial{}_stats.txt'.format(trial)
    stats_pickle = 'MC_trial{}_stats.pickle'.format(trial)

    # run the GAME model
    hc = GAME_SAHC(target_task_name, src_state_var_names, src_action_names, src_action_values, target_state_var_names, target_action_names, target_action_values, 
    src_task_data_folder_and_filename, neural_networks_folder, eval_metric, init_strat, max_fitness_evals, 
    early_stop, early_stop_gen, early_stop_thresh, print_debug, save_output_path, save_every, stats_saver, stats_folder_path, stats_filename, standard_features, standard_targets, count_comparisons)
    hc.evolve()
    
    # print the final evolved population
    for mapping in hc.population:
        print(mapping)

    with open(os.path.join(stats_folder_path, stats_pickle), 'wb') as f:
        pickle.dump(hc.stats_saver, f)