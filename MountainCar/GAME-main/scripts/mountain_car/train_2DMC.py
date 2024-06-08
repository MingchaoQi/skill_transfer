# Author: Minh Hua
# Date: 10/31/2022
# Purpose: This module contains classes to load transition data collected from reinforcement learning.

import GAME.envs.mountain_car
import gym
import pickle
import os

from GAME.agents.sarsa_lambda import SarsaLambdaCMAC2DMountainCar, SarsaLambdaCMAC3DMountainCar
from GAME.utils.helper_funcs import *
from GAME.utils.data_miners import *
from GAME.utils.config import config
from GAME.bin.mountain_car_experiments import MountainCar2DExperiment
from GAME.bin.intertask_mappings import IntertaskMapping

# load config data
config_data = config()

# init agent
alpha = 1.2
lamb = 0.95
gamma = 1
method = 'replacing'
epsilon = 0
num_of_tilings = 8
max_size = 2048
decay_agent_eps = None
base_agent_class = SarsaLambdaCMAC2DMountainCar
base_agent = SarsaLambdaCMAC2DMountainCar(alpha, lamb, gamma, method, epsilon, num_of_tilings, max_size)

# experimental setup
env_name = 'MountainCar2D-v0'
env_max_steps = 5000
rd_seed = 42
max_episodes_per_trial = 2010
num_trials = 1 
update_agent = True
start_learning_after = 10
print_debug = True

# whether to save sample transition data
agent_info = SarsaLambdaAgentInfo(alpha, lamb, gamma, method, epsilon, num_of_tilings, max_size)
experiment_info = ExperimentInfo(env_name, env_max_steps, rd_seed, max_episodes_per_trial, 'SarsaLambda')
save_sample_data = False
save_sample_every = None
sample_data_col_names = ['Trial', 'Episode', 'Step']
sample_data_col_names = sample_data_col_names + config_data['3DMC_current_state_transition_df_col_names']
sample_data_col_names = sample_data_col_names + [config_data['action_transition_df_col_name'], 'Reward']
sample_data_col_names = sample_data_col_names + config_data['3DMC_next_state_transition_df_col_names']
sample_data_col_names = sample_data_col_names + [config_data['next_action_transition_df_col_name']]
sample_data_column_dtypes = ['int', 'int', 'int', 'float', 'float', 'float', 'float', 'int', 'int', 'float', 'float', 'float', 'float', 'int']
sample_data_folder = None
sample_data_filename = None
data_collector = None

experiment_name = "11122022 Train MC2D"

# whether or not to save the agent's weights
save_agent = True
save_agent_every = 25
save_agent_folder = os.path.join(config_data["pickle_path"], experiment_name)
save_agent_filename = 'agent_alpha_{:.2f}_lamb_{:.2f}_gam_{:.2f}_eps_{:.2f}_method_{}_ntiles_{}_max_size_{}.pickle'.format(alpha, lamb, gamma, epsilon, method, num_of_tilings, max_size)

# whether to evaluate the agent and save the evaluation data
eval_agent = False
save_eval_every = 25
eval_data_col_names = ['Trial', 'Episode', 'Reward']
eval_data_column_dtypes = ['int', 'int', 'int']
save_eval_folder = os.path.join(config_data["output_path"], experiment_name)
save_eval_filename = 'eval_3DMC_a{}_l{}_e{}_nt{}.csv'.format(alpha, lamb, epsilon, num_of_tilings)
eval_data_collector = RLSamplesCollector(experiment_info, agent_info, eval_data_col_names, eval_data_column_dtypes)

transfer = 2

# run experiment
average_steps_per_trial = MountainCar2DExperiment(base_agent_class, base_agent, decay_agent_eps, max_episodes_per_trial, 
num_trials, update_agent, start_learning_after, print_debug, save_sample_data, save_sample_every, 
sample_data_col_names, sample_data_folder, sample_data_filename, data_collector, save_agent, save_agent_every,
save_agent_folder, save_agent_filename, eval_agent, save_eval_every, eval_data_col_names, save_eval_folder, save_eval_filename, 
eval_data_collector, env_name, env_max_steps, rd_seed)