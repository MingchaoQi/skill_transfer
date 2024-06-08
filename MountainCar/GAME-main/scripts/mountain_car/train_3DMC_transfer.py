# Author: Minh Hua
# Date: 10/31/2022
# Purpose: This module contains classes to load transition data collected from reinforcement learning.

import GAME.envs.mountain_car
import gym
import pickle
import os

from GAME.agents.sarsa_lambda import SarsaLambdaCMAC3DMountainCarTransfer, SarsaLambdaCMAC3DMountainCar
from GAME.utils.helper_funcs import *
from GAME.utils.data_miners import *
from GAME.utils.config import config
from GAME.bin.mountain_car_experiments import MountainCar3DExperiment
from GAME.bin.intertask_mappings import IntertaskMapping

# load config data
config_data = config()

# init agent
alpha = 0.5
lamb = 0.5
gamma = 1
method = 'replacing'
epsilon = 0.01
num_of_tilings = 8
max_size = 4096
decay_agent_eps = None
base_agent_class = SarsaLambdaCMAC3DMountainCarTransfer
# mapping_ID = '010110202'
# agent_filename = '3DMC_with_transfer_a{}_l{}_e{}_nt{}_{}.pickle'.format(alpha, lamb, epsilon, num_of_tilings, mapping_ID)
# trained_agent_path = os.path.join(config_data['pickle_path'], 'agents', 'mountain_car', '11112022 3DMC Agent Initialized with Transfer Mapping 010100000', agent_filename)
# with open(trained_agent_path, 'rb') as f:
#     base_agent = pickle.load(f)
#     base_agent.epsilon = 0

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
base_agent = SarsaLambdaCMAC3DMountainCarTransfer(alpha, lamb, gamma, method, epsilon, num_of_tilings, max_size, mc2d_agent, mapping)
# base_agent.epsilon = 0

# experimental setup
env_name = 'MountainCar3D-v0'
env_max_steps = 5000
rd_seed = 42
max_episodes_per_trial = 2010
num_trials = 10
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

experiment_name = "01092023 Train MC3D With QValue Transfer Mapping {} 2".format(mapping_ID)

# whether or not to save the agent's weights
save_agent = True
save_agent_every = 25
save_agent_folder = os.path.join(config_data["pickle_path"], experiment_name)
save_agent_filename = 'a_{:.2f}_l_{:.2f}_g_{:.2f}_eps_{:.2f}_method_{}_nt_{}_size_{}.pickle'.format(alpha, lamb, gamma, epsilon, method, num_of_tilings, max_size)

# whether to evaluate the agent and save the evaluation data
eval_agent = True
save_eval_every = 25
eval_data_col_names = ['Trial', 'Episode', 'Reward']
eval_data_column_dtypes = ['int', 'int', 'int']
save_eval_folder = os.path.join(config_data["output_path"], experiment_name)
save_eval_filename = 'eval_3DMC_a{}_l{}_e{}_nt{}.csv'.format(alpha, lamb, epsilon, num_of_tilings)
eval_data_collector = RLSamplesCollector(experiment_info, agent_info, eval_data_col_names, eval_data_column_dtypes)

transfer = 2

# run experiment
average_steps_per_trial = MountainCar3DExperiment(base_agent_class, base_agent, decay_agent_eps, max_episodes_per_trial, 
num_trials, update_agent, start_learning_after, print_debug, save_sample_data, save_sample_every, 
sample_data_col_names, sample_data_folder, sample_data_filename, data_collector, save_agent, save_agent_every,
save_agent_folder, save_agent_filename, eval_agent, save_eval_every, eval_data_col_names, save_eval_folder, save_eval_filename, 
eval_data_collector, env_name, env_max_steps, rd_seed, transfer)