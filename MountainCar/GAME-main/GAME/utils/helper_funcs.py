# Author: Minh Hua
# Date: 10/24/2022
# Purpose: This module contains helper functions for the entire project.

import pandas as pd
import numpy as np
import pickle
import os

def distance(p1:tuple, p2:tuple) -> float:
    """
    Description:
        Returns the Euclidean distance between two points.

    Arguments:
        p1: the first point.
        p2: the second point.

    Return:
        (float) the Euclidean distance between two points.
    """
    return np.sum(np.square(np.array(p1) - np.array(p2)))

def angle(a:tuple, b:tuple, c:tuple) -> float:
    """
    Description:
        Returns the angle between two vectors, as defined by three points.

    Arguments:
        a: the first point.
        b: the second point. Usually the middle point that the two vectors share.
        c: the third point.

    Return:
        (float) the angle in degrees between the three poitns.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def save_agent_to_file(agent, path:str, agent_filename:str, agent_weights_filename:bool=True, agent_hash_filename:bool=True, agent_z_filename:bool=True) -> None:
    """
    Description:
        Saves a Sarsa lambda agent.

    Arguments:
        agent: the agent to be saved.
        path: the path containing the folder to save agent data.
        agent_filename: the filename for the agent.
        agent_weights_filename: the filename for the agent's weights.
        agent_hash_filename: the filename for the agent's hash table.
        agent_z_filename: the filename for the agent's trace.

    Return:
        (None) 
    """
    with open(os.path.join(path, agent_filename), 'wb') as f:
        pickle.dump(agent, f)
    if agent_weights_filename:
        agent_weights_filename = ".".join(agent_filename.split('.')[:-1]) + '_weights.pickle'
        with open(os.path.join(path, agent_weights_filename), 'wb') as f:
            pickle.dump(agent.weights, f)
    if agent_hash_filename:
        agent_hash_filename = ".".join(agent_filename.split('.')[:-1]) + '_hash.pickle'
        with open(os.path.join(path, agent_hash_filename), 'wb') as f:
            pickle.dump(agent.hash_table, f)
    if agent_z_filename:
        agent_z_filename = ".".join(agent_filename.split('.')[:-1]) + '_z.pickle'
        with open(os.path.join(path, agent_z_filename), 'wb') as f:
            pickle.dump(agent.z, f)

def generate_keepaway_learning_curves(kwy_paths:list, window_size:int=900, alpha:float=0.01, coarse:int=30) -> tuple:
    """
    Description:
        Graphs keepaway learning curves using .kwy log files.

    Arguments:
        kwy_paths: a list of paths to the .kwy files. Used to splice together separate .kwy files.
        window_size: the size of the sliding window for averaging the number of steps.
        alpha: alpha value used in the low-pass filter to smooth the curve.
        coarse: every coarse-th point gets outputted to cut down on file size.

    Return:
        (tuple) a list of the training time and a list of the average episode durations.
    """
    # init variables
    ep_dur_sum = 0 # Window sum of episode durations
    start = 0
    q = np.zeros(shape = (1, window_size))
    output_X = []
    output_Y = []
    output_eps = []

    # read the first kwy file. There might be more that results from interrupted training
    initial_kwy_file_path = kwy_paths[0]

    # read the first window_size entries
    with open(initial_kwy_file_path, 'r') as f:
        # ignore header
        line = f.readline()
        while line[0] == '#':
            line = f.readline()
        # fill window
        for i in range(len(q[0])):
            if line: # while we still have non-empty lines
                parsed_line = line.split()
                ep_dur = int(parsed_line[3])
                q[0][i] = ep_dur
                ep_dur_sum += q[0][i]
                line = f.readline()
            else:
                raise ValueError("Not enough data to fill window.")

        # read the rest of the data
        i = 0
        ep = 0
        ccount = 0
        prev = ep_dur_sum
        while line:
            if ccount % coarse == 0:
                output_X.append(start / 10.0 / 3600)
                output_Y.append(prev / 10.0 / window_size)
                output_eps.append(ep)
            
            ep_dur_sum -= q[0][i]
            parsed_line = line.split()
            ep_dur = int(parsed_line[3])
            q[0][i] = ep_dur
            ep_dur_sum += q[0][i]
            start += q[0][i]

            prev = (1 - alpha) * prev + alpha * ep_dur_sum
            i = (i + 1) % window_size
            ccount += 1
            line = f.readline()

            ep += 1
    
    # continue to parse keepaway files if more are supplied. Appends this data to the previously computed data
    if len(kwy_paths) > 1:
        for kwy_path_idx in range(1, len(kwy_paths)):
            with open(kwy_paths[kwy_path_idx], 'r') as f:
                # ignore header
                line = f.readline()
                while line[0] == '#':
                    line = f.readline()

                # read the rest of the data
                while line:
                    if ccount % coarse == 0:
                        output_X.append(start / 10.0 / 3600)
                        output_Y.append(prev / 10.0 / window_size)
                        output_eps.append(ep)
                    
                    ep_dur_sum -= q[0][i]
                    parsed_line = line.split()
                    ep_dur = int(parsed_line[3])
                    q[0][i] = ep_dur
                    ep_dur_sum += q[0][i]
                    start += q[0][i]

                    prev = (1 - alpha) * prev + alpha * ep_dur_sum
                    i = (i + 1) % window_size
                    ccount += 1
                    line = f.readline()
                    
                    ep += 1
    
    # return the parsed kwy data
    return output_X, output_Y, output_eps

def generate_MC_learning_curves(training_data:pd.DataFrame, window_size:int=10, alpha:float=0.01, coarse:int=30) -> tuple:
    """
    Description:
        Graphs Mountain Car learning curves using .csv training output files.

    Arguments:
        training_data: a pandas DataFrame containing the trial number, the episode, and the reward for that episode.
        window_size: the size of the sliding window for averaging the number of steps.
        alpha: alpha value used in the low-pass filter to smooth the curve.
        coarse: every coarse-th point gets outputted to cut down on file size.

    Return:
        (tuple) a list of the training time and a list of the average episode durations.
    """
    # init variables
    ep_reward_sum = 0 # Window sum of episode durations
    start = 0
    q = np.zeros(shape = (1, window_size))
    output_X = []
    output_Y = []
    output_eps = []
    eps = 0

    # read the first window_size entries
    i = 0
    for ind, row in training_data.iterrows():
        # fill window
        if ind < window_size:
            ep_reward = row['Reward']
            q[0][i] = ep_reward
            ep_reward_sum += q[0][i]
            i += 1
        else:
            break

    # read the rest of the data
    i = 0
    ccount = 0
    prev = ep_reward_sum
    for ind, row in training_data.iterrows():
        if ind >= window_size:
            if ccount % coarse == 0:
                output_X.append(start)
                output_Y.append(prev / window_size)
                output_eps.append(eps)
            
            ep_reward_sum -= q[0][i]
            ep_reward = row['Reward']
            q[0][i] = ep_reward
            ep_reward_sum += q[0][i]
            start = row['Episode']
            eps += 1

            prev = (1 - alpha) * prev + alpha * ep_reward_sum
            i = (i + 1) % window_size
            ccount += 1
    
    # return the parsed kwy data
    return output_X, output_Y, output_eps