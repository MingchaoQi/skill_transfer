# Author: Minh Hua
# Date: 11/1/2022
# Purpose: This script collects transition data from 3D Mountain Car.

from GAME.agents.sarsa_lambda import SarsaLambdaCMAC3DMountainCar
from GAME.utils.data_miners import *
from GAME.bin.MountainCarExperiments import MountainCar3DExperiment
from GAME.utils.config import config

# load config data
config_data = config()

# init agent
alpha = 1.2
lamb = 0.95
gamma = 1
method = 'replacing'
epsilon = 1
num_of_tilings = 8
max_size = 2048
decay_agent_eps = None
base_agent_class = SarsaLambdaCMAC3DMountainCar
base_agent = SarsaLambdaCMAC3DMountainCar(alpha, lamb, gamma, method, epsilon, num_of_tilings, max_size)

# experimental setup
env_name = 'MountainCar3D-v0'
env_max_steps = 5000
rd_seed = 42
max_episodes_per_trial = 50
num_trials = 1 
update_agent = False
start_learning_after = -1
print_debug = True

# whether to save sample transition data
agent_info = SarsaLambdaAgentInfo(alpha, lamb, gamma, method, epsilon, num_of_tilings, max_size)
experiment_info = ExperimentInfo(env_name, env_max_steps, 42, rd_seed, 'SarsaLambda')
save_sample_data = True
save_sample_every = 25
sample_data_col_names = ['Trial', 'Episode', 'Step']
sample_data_col_names = sample_data_col_names + config_data['3DMC_current_state_transition_df_col_names']
sample_data_col_names = sample_data_col_names + [config_data['Current-action'], 'Reward']
sample_data_col_names = sample_data_col_names + config_data['3DMC_next_state_transition_df_col_names']
sample_data_col_names = sample_data_col_names + [config_data['next_action_transition_df_col_name']]
sample_data_column_dtypes = ['int', 'int', 'int', 'float', 'float', 'float', 'float', 'int', 'int', 'float', 'float', 'float', 'float', 'int']
sample_data_folder = config_data["output_path"] + "11012022 3DMC Sample Collection 50 Episodes Full Explore\\"
sample_data_filename = '3DMC_50_episodes_sample_data.csv'
data_collector = RLSamplesCollector(experiment_info, agent_info, sample_data_col_names, sample_data_column_dtypes)

# whether or not to save the agent's weights
save_agent = False
save_agent_every = None
save_agent_folder = None
save_agent_filename = None

# whether to evaluate the agent and save the evaluation data
eval_agent = False
save_eval_every = None
eval_data_col_names = None
eval_data_column_dtypes = None
save_eval_folder = None
save_eval_filename = None
eval_data_collector = None

# run experiment
average_steps_per_trial = MountainCar3DExperiment(base_agent_class, base_agent, decay_agent_eps, max_episodes_per_trial, 
num_trials, update_agent, start_learning_after, print_debug, save_sample_data, save_sample_every, 
sample_data_col_names, sample_data_folder, sample_data_filename, data_collector, save_agent, save_agent_every,
save_agent_folder, save_agent_filename, eval_agent, save_eval_every, eval_data_col_names, save_eval_folder, save_eval_filename, 
eval_data_collector, env_name, env_max_steps, rd_seed)

