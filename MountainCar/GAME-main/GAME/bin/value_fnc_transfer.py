# Author: Minh Hua
# Date: 11/11/2022
# Purpose: This module transfer knowledge using inter-task mappings for a variety of different tasks.

from GAME.utils.config import config
from GAME.agents.sarsa_lambda import SarsaLambdaCMAC2DMountainCar, SarsaLambdaCMAC3DMountainCar
from GAME.agents.TileCoding import *
from GAME.bin.intertask_mappings import IntertaskMapping
import pickle
import numpy as np
import os

def mountain_car_transfer(src_agent:SarsaLambdaCMAC2DMountainCar, dest_agent:SarsaLambdaCMAC3DMountainCar, mapping:IntertaskMapping, len_src_state_vals:int):
    # copy weights from source agent to destination agent
    for src_w_idx, src_weight in enumerate(src_agent.weights):
        # find source state and action values corresponding to weight i in hash table
        for k, v in src_agent.hash_table.dictionary.items():
            if v == src_w_idx:
                src_coord = k
                break
        # extract the source state and action values from the coordinates
        tiling_no = src_coord[0]
        src_state_vals = src_coord[1:len_src_state_vals+1]
        src_action_vals = src_coord[-1]
        # get same target state and actions
        target_state_val = [src_state_vals[i] for i in mapping.state_mapping]
        target_action_vals = [i for i in range(len(mapping.action_mapping)) if mapping.action_mapping[i] == src_action_vals]
        r2_values = [[0.9929816541449503, 0.9863922534862493, 0.9827540818142966],
        [0.9898554363903732, 0.9807461375299111, 0.9727148311728532], [0.992743856876952, 0.9865786993775076, 0.9873256083696892],
        [0.9938926191874691, 0.9863977078527576, 0.9832993967828477], [0.9813449416585194, 0.9650493513874667, 0.9363257347643477]
        ]
        r2_sum = 0
        for target_action_val in [0, 1, 2, 3, 4]:
            r2_sum += r2_values[target_action_val][src_action_vals]
        for target_action_val in [0, 1, 2, 3, 4]:
        # for target_action_val in target_action_vals:
            coordinates = (tiling_no, *target_state_val, target_action_val)
            target_w_idx = hashcoords(coordinates, dest_agent.hash_table)
            # set target weight
            dest_agent.weights[target_w_idx] = src_weight * (r2_values[target_action_val][src_action_vals] / r2_sum)

    # set all empty weights in target CMAC to be the average value of all non-zero weights in the target CMAC
    nonzero_weights_idx = dest_agent.weights.nonzero()
    average_weight = np.mean(dest_agent.weights[nonzero_weights_idx])
    for w_idx in range(len(dest_agent.weights)):
        if dest_agent.weights[w_idx] == 0:
            dest_agent.weights[w_idx] = average_weight

    return dest_agent

if __name__ == "__main__":
    config_data = config()

    src_state_var_names = config_data['MC2D_state_names']
    src_action_names = config_data['MC2D_action_names']
    src_action_values = config_data['MC2D_action_values']
    target_state_var_names = config_data['MC3D_state_names']
    target_action_names = config_data['MC3D_action_names']
    target_action_values = config_data['MC3D_action_values']

    mc2d_agent_folder_path = os.path.join(config_data['pickle_path'], '11032022 2DMC Sample Collection 100 Episodes with Training')
    mc2d_agent_file_path = os.path.join(mc2d_agent_folder_path, '2DMC_100_ep_a1.2_l0.95_e0_nt8.pickle')

    mc3d_agent_folder_path = os.path.join(config_data['pickle_path'], '11072022 Train MC3D No Transfer')
    mc3d_agent_file_path = os.path.join(mc3d_agent_folder_path, 'trial0_agent_alpha_0.75_lamb_0.99_gam_1.00_eps_0.01_method_replacing_ntiles_8_max_size_4096.pickle')    

    with open(mc2d_agent_file_path, 'rb') as f:
        mc2d_agent = pickle.load(f)

    with open(mc3d_agent_file_path, 'rb') as f:
        mc3d_agent = pickle.load(f)

    # agent
    alpha = 0.75
    lamb = 0.99
    gamma = 1
    method = 'replacing'
    epsilon = 0.01
    num_of_tilings = 8
    max_size = 4096
    mc3d_agent = SarsaLambdaCMAC3DMountainCar(alpha, lamb, gamma, method, epsilon, num_of_tilings, max_size)

    state_mapping = [0, 1, 0, 1]
    action_mapping = [1, 0, 2, 0, 2]
    mapping = IntertaskMapping(state_mapping, action_mapping, src_state_var_names, src_action_names, target_state_var_names, target_action_names)

    # transfer knowledge
    mc3d_agent = mountain_car_transfer(mc2d_agent, mc3d_agent, mapping, len(src_state_var_names))

    dest_agent_filename = '3DMC_with_transfer_a{}_l{}_e{}_nt{}_{}.pickle'.format(mc3d_agent.alpha, mc3d_agent.lamb, mc3d_agent.epsilon, mc3d_agent.num_of_tilings, mapping.ID)
    save_agent_path = os.path.join(config_data['pickle_path'], 'agents', 'mountain_car', '12142022 New Experiments', dest_agent_filename)
    with open(save_agent_path, 'wb') as f:
        pickle.dump(mc3d_agent, f)