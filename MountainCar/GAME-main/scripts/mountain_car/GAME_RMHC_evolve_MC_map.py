# Author: Minh Hua
# Date: 11/11/2022
# Purpose: Runs 10 trials of evolving mappings for Mountain Car.

import os

from GAME.bin.hill_climber import GAME_RMHC
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
early_stop = False
early_stop_gen = 20
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
search_exp_info = MappingSearchExperimentInfo('2DMC', '3DMC', 'GAME_RMHC', agent_params)

standard_features = False
standard_targets = False

count_comparisons = True

# helper variables
# transforming src data
src_task_data_folder_and_filename = os.path.join(config_data['output_path'], '12142022 2DMC Sample Collection 200 Episodes with Training', "2DMC_100_episodes_sample_data_small.csv")
neural_networks_folder = os.path.join(config_data['pickle_path'], "01072023 3DMC Transition Approx MSE")

# run for 15 trials
trials = 10
for trial in range(trials):
    print("Trial: {}".format(trial))
    
    # trial-specific params
    output_folder_name = '01072023 Evolve MC Maps with GAME-RMHC 240 FE'
    save_output_path  = os.path.join(config_data['output_path'], output_folder_name, 'trial{}_population_results.txt'.format(trial))
    stats_saver = StatisticsSaver(search_exp_info, trial, True)
    stats_folder_path = os.path.join(config_data['output_path'], output_folder_name)
    stats_filename = 'trial{}_stats.csv'.format(trial)
    stats_pickle = 'trial{}_stats.pickle'.format(trial)

    # run the GAME model
    hc = GAME_RMHC(target_task_name, src_state_var_names, src_action_names, src_action_values, target_state_var_names, target_action_names, target_action_values, 
    src_task_data_folder_and_filename, neural_networks_folder, eval_metric, init_strat, max_fitness_evals, 
    early_stop, early_stop_gen, early_stop_thresh, print_debug, save_output_path, save_every, stats_saver, stats_folder_path, stats_filename, standard_features, standard_targets, count_comparisons)
    hc.evolve()
    
    # print the final evolved population
    for mapping in hc.population:
        print(mapping)

    with open(os.path.join(stats_folder_path, stats_pickle), 'wb') as f:
        pickle.dump(hc.stats_saver, f)