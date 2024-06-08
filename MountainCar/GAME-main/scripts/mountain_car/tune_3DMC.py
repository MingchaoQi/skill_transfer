# Author: Minh Hua
# Date: 11/1/2022
# Purpose: This module tunes a Sarsa lambda agent for 3D Mountain Car.

import numpy as np

import GAME.envs.mountain_car
import gym
from GAME.agents.sarsa_lambda import SarsaLambdaCMAC3DMountainCar, SarsaLambdaCMAC3DMountainCarTransfer
from GAME.utils.helper_funcs import *
from GAME.utils.data_miners import *
from GAME.bin.intertask_mappings import IntertaskMapping
from GAME.bin.mountain_car_experiments import MountainCar3DExperiment
from GAME.utils.config import config

# load config data
config_data = config()

# Q-value reuse agent
mc2d_agent_folder_path = os.path.join(config_data['pickle_path'], '11122022 Train MC2D')
mc2d_agent_file_path = os.path.join(mc2d_agent_folder_path, 'trial0_agent_alpha_1.20_lamb_0.95_gam_1.00_eps_0.00_method_replacing_ntiles_8_max_size_2048.pickle')
with open(mc2d_agent_file_path, 'rb') as f:
    mc2d_agent = pickle.load(f)
src_state_var_names = config_data['MC2D_state_names']
src_action_names = config_data['MC2D_action_names']
src_action_values = config_data['MC2D_action_values']
target_state_var_names = config_data['MC3D_state_names']
target_action_names = config_data['MC3D_action_names']
target_action_values = config_data['MC3D_action_values']
state_mapping = [0, 1, 0, 1]
action_mapping = [1, 1, 2, 1, 2]
mapping = IntertaskMapping(state_mapping, action_mapping, src_state_var_names, src_action_names, target_state_var_names, target_action_names)
mapping_ID = mapping.ID

# tuning parameters
log = os.path.join(config_data['output_path'], "01092023 3DMC Tuning with 010111212", "tuning_3DMC.txt")

# agent hyperparams
agent_hyperparams = {
    'alpha' : np.arange(1, 6) / 4.0,
    'lamb' : [0.99, 0.95, 0.5, 0],
    'gamma' : [1],
    'method' : ['replacing', 'accumulating'],
    'epsilon': [0.01, 1],
    'num_of_tilings': [8, 14],
    'max_size' : [2048, 4096],
    'transfer_agent': [mc2d_agent],
    'mapping': [mapping]
}

decay_agent_eps = 1
base_agent_class = SarsaLambdaCMAC3DMountainCarTransfer

# experimental setup
env_name = 'MountainCar3D-v0'
env_max_steps = 5000
rd_seed = 420
max_episodes_per_trial = 10
num_trials = 1 
update_agent = True
start_learning_after = -1
print_debug = True

# whether to save sample transition data
experiment_info = ExperimentInfo(env_name, env_max_steps, rd_seed, max_episodes_per_trial, 'SarsaLambda')
save_sample_data = False
save_sample_every = None
sample_data_col_names = None
sample_data_column_dtypes = None
sample_data_folder = None
sample_data_filename = None
data_collector = None

# whether or not to save the agent's weights
save_agent = False
save_agent_every = None
save_agent_folder = None
save_agent_filename = None

# whether to evaluate the agent and save the evaluation data
eval_agent = True
save_eval_every = 100
eval_data_col_names = ['Trial', 'Episode', 'Reward']
eval_data_column_dtypes = ['int', 'int', 'int']
save_eval_folder = os.path.join(config_data['output_path'], "01092023 3DMC Tuning with 010111212")

min = True

# initialize log file if requested
try:
    if log:
        f = open(log, "w")
    dim = [] # holds the dimension of the hyperparameters so we can initialize an np array
    for param_vals in agent_hyperparams.values():
        dim.append(len(param_vals))
    # empty np array to hold the results corresponding to a parameter combination
    grid_search_results = np.empty((dim))
    i = 0
    # loop through every parameter combination and find the best one
    for idx, _ in np.ndenumerate(grid_search_results):
        # get the current combination of parameters
        cur_args_list = []
        for cur_param, param_key in zip(idx, agent_hyperparams.keys()):
            # print(param_key, hyperparams[param_key][cur_param])
            cur_args_list.append(agent_hyperparams[param_key][cur_param])
        print("Current args: {}".format(cur_args_list))
        if log:
            f.write("Current args: {}".format(cur_args_list))

        # instantiate the model with the current list of parameters
        base_agent = SarsaLambdaCMAC3DMountainCarTransfer(*cur_args_list)
        if base_agent.epsilon == 1:
            decay_agent_eps = 0.99

        # run experiment
        save_eval_filename = 'eval_3DMC_{}.csv'.format(cur_args_list[:-2])
        eval_data_collector = RLSamplesCollector(experiment_info, None, eval_data_col_names, eval_data_column_dtypes)
        average_steps_per_trial = MountainCar3DExperiment(base_agent_class, base_agent, decay_agent_eps, max_episodes_per_trial, 
        num_trials, update_agent, start_learning_after, print_debug, save_sample_data, save_sample_every, 
        sample_data_col_names, sample_data_folder, sample_data_filename, data_collector, save_agent, save_agent_every,
        save_agent_folder, save_agent_filename, eval_agent, save_eval_every, eval_data_col_names, save_eval_folder, save_eval_filename, 
        eval_data_collector, env_name, env_max_steps, rd_seed, 2)
        
        i += 1
        # append the model's average performance to a grid
        if len(average_steps_per_trial) > 0:
            print("{}: {}".format('Average Steps', sum(average_steps_per_trial) / len(average_steps_per_trial)))
            grid_search_results[idx] = sum(average_steps_per_trial) / len(average_steps_per_trial)
            if log:
                f.write("{}: {}".format('Average Steps', sum(average_steps_per_trial) / len(average_steps_per_trial)) + "\n")
                if i % 10 == 0:
                    f.close()
                    f = open(log, "a")
    # find the optimal combination of parameters and return it
    if not min:
        optimal_param_idx = np.unravel_index(grid_search_results.argmax(), grid_search_results.shape)
    else:
        optimal_param_idx = np.unravel_index(grid_search_results.argmin(), grid_search_results.shape)
    optimal_params = {}
    if log:
        f.write('Optimal params' + "\n")
    for cur_param, param_key in zip(optimal_param_idx, agent_hyperparams.keys()):
        # print("Optimal ", param_key, hyperparams[param_key][cur_param]) 
        optimal_params[param_key] = agent_hyperparams[param_key][cur_param]
        if log:
            f.write('{}: {}'.format(param_key, agent_hyperparams[param_key][cur_param]) + "\n")
    f.close()
# except KeyboardInterrupt:
#     if log:
#         f.close()
except:
    pass