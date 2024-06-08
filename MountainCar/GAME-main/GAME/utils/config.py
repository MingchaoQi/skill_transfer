# Author: Minh Hua
# Date: 11/3/2022
# Purpose: This module contains a config file that can be changed to contain metadata to be used in various places.

import os

def config() -> dict:
    """
    Description:
        Returns a set of config variables that can be used. For example, the path to the various different folders in the project.

    Arguments:
        (None)

    Return:
        (None)
    """
    config = {}
    # paths
    # path to the main repository
    config['GAME_path'] = os.path.join('E:\\VAULT 419', 'Files', 'School', 'JHU Archive', 'Fall 2022', 'Evolutionary and Swarm Intelligence', 'src', 'GAME')
    # path to the logs
    config['logs_path'] = os.path.join(config['GAME_path'], 'keepaway_logs')
    # path to the outputs
    config['output_path'] = os.path.join(config['GAME_path'], 'output')
    # path to the pickled objects
    config['pickle_path'] = os.path.join(config['GAME_path'], 'pickle')
    # path to data folder
    config['data_path'] = os.path.join(config['GAME_path'], 'data')

    # task and environment variables
    # 2D Mountain Car
    config['MC2D_state_names'] = ['x_position', 'x_velocity']
    config['MC2D_action_names'] = ['Left', 'Neutral', 'Right']
    config['MC2D_action_values'] = [0, 1, 2]
    # 3D Mountain Car
    config['MC3D_state_names'] = ['x_position', 'x_velocity', 'y_position', 'y_velocity']
    config['MC3D_action_names'] = ['Neutral', 'West', 'East', 'South', 'North']
    config['MC3D_action_values'] = [0, 1, 2, 3, 4]
    # 3v2 Keepaway
    config['3v2_state_names'] = [
        'dist(K1,C)',
        'dist(K1,K2)',
        'dist(K1,K3)',
        'dist(K1,T1)',
        'dist(K1,T2)',
        'dist(K2,C)',
        'dist(K3,C)',
        'dist(T1,C)',
        'dist(T2,C)',
        'Min(dist(K2,T1),dist(K2,T2))',
        'Min(dist(K3,T1),dist(K3,T2))',
        'Min(ang(K2,K1,T1),ang(K2,K1,T2))',
        'Min(ang(K3,K1,T1),ang(K3,K1,T2))',
    ]
    config['3v2_action_names'] = ['Hold', 'Pass1', 'Pass2']
    config['3v2_action_values'] = [0, 1, 2]
    # 4v3 Keepaway
    config['4v3_state_names'] = [
        'dist(K1,C)',
        'dist(K1,K2)',
        'dist(K1,K3)',
        'dist(K1,K4)',
        'dist(K1,T1)',
        'dist(K1,T2)',
        'dist(K1,T3)',
        'dist(K2,C)',
        'dist(K3,C)',
        'dist(K4,C)',
        'dist(T1,C)',
        'dist(T2,C)',
        'dist(T3,C)',
        'Min(dist(K2,T1),dist(K2,T2),dist(K2,T3))',
        'Min(dist(K3,T1),dist(K3,T2),dist(K3,T3))',
        'Min(dist(K4,T1),dist(K4,T2),dist(K4,T3))',
        'Min(ang(K2,K1,T1),ang(K2,K1,T2),ang(K2,K1,T3))',
        'Min(ang(K3,K1,T1),ang(K3,K1,T2),ang(K3,K1,T3))',
        'Min(ang(K4,K1,T1),ang(K4,K1,T2),ang(K4,K1,T3))'
    ]
    config['4v3_action_names'] = ['Hold', 'Pass1', 'Pass2', 'Pass3']
    config['4v3_action_values'] = [0, 1, 2, 3]
    # 3D Mountain Car
    config['pendulum_state_names'] = ['cos(th)', 'sin(th)', 'thdot']
    config['pendulum_action_names'] = ['-1.0', '0', '1.0']
    config['pendulum_action_values'] = [-1.0, 0, 1.0]    

    # column names for transition data
    config['action_transition_df_col_name'] = 'Current-action'
    config['next_action_transition_df_col_name'] = 'Next-action'
    # 2DMC
    config['2DMC_current_state_transition_df_col_names'] = ['Current-x_position', 'Current-x_velocity']
    config['2DMC_next_state_transition_df_col_names'] = ['Next-x_position', 'Next-x_velocity']
    config['2DMC_full_transition_df_col_names'] = config['2DMC_current_state_transition_df_col_names'] + [config['action_transition_df_col_name']] + config['2DMC_next_state_transition_df_col_names']
    
    # 3DMC
    config['3DMC_current_state_transition_df_col_names'] = ['Current-x_position', 'Current-x_velocity', 'Current-y_position', "Current-y_velocity"]
    config['3DMC_next_state_transition_df_col_names'] = ['Next-x_position', 'Next-x_velocity', 'Next-y_position', 'Next-y_velocity']
    config['3DMC_full_transition_df_col_names'] = config['3DMC_current_state_transition_df_col_names'] + [config['action_transition_df_col_name']] + config['3DMC_next_state_transition_df_col_names']
    
    # 3v2
    config['3v2_current_state_transition_df_col_names'] = [
        'Current-dist(K1,C)',
        'Current-dist(K1,K2)',
        'Current-dist(K1,K3)',
        'Current-dist(K1,T1)',
        'Current-dist(K1,T2)',
        'Current-dist(K2,C)',
        'Current-dist(K3,C)',
        'Current-dist(T1,C)',
        'Current-dist(T2,C)',
        'Current-Min(dist(K2,T1),dist(K2,T2))',
        'Current-Min(dist(K3,T1),dist(K3,T2))',
        'Current-Min(ang(K2,K1,T1),ang(K2,K1,T2))',
        'Current-Min(ang(K3,K1,T1),ang(K3,K1,T2))'
    ]
    config['3v2_next_state_transition_df_col_names'] = [
        'Next-dist(K1,C)',
        'Next-dist(K1,K2)',
        'Next-dist(K1,K3)',
        'Next-dist(K1,T1)',
        'Next-dist(K1,T2)',
        'Next-dist(K2,C)',
        'Next-dist(K3,C)',
        'Next-dist(T1,C)',
        'Next-dist(T2,C)',
        'Next-Min(dist(K2,T1),dist(K2,T2))',
        'Next-Min(dist(K3,T1),dist(K3,T2))',
        'Next-Min(ang(K2,K1,T1),ang(K2,K1,T2))',
        'Next-Min(ang(K3,K1,T1),ang(K3,K1,T2))'
    ]
    config['3v2_full_transition_df_col_names'] = config['3v2_current_state_transition_df_col_names'] + [config['action_transition_df_col_name']] + config['3v2_next_state_transition_df_col_names']

    # 4v3
    config['4v3_current_state_transition_df_col_names'] = [
        'Current-dist(K1,C)',
        'Current-dist(K1,K2)',
        'Current-dist(K1,K3)',
        'Current-dist(K1,K4)',
        'Current-dist(K1,T1)',
        'Current-dist(K1,T2)',
        'Current-dist(K1,T3)',
        'Current-dist(K2,C)',
        'Current-dist(K3,C)',
        'Current-dist(K4,C)',
        'Current-dist(T1,C)',
        'Current-dist(T2,C)',
        'Current-dist(T3,C)',
        'Current-Min(dist(K2,T1),dist(K2,T2),dist(K2,T3))',
        'Current-Min(dist(K3,T1),dist(K3,T2),dist(K3,T3))',
        'Current-Min(dist(K4,T1),dist(K4,T2),dist(K4,T3))',
        'Current-Min(ang(K2,K1,T1),ang(K2,K1,T2),ang(K2,K1,T3))',
        'Current-Min(ang(K3,K1,T1),ang(K3,K1,T2),ang(K3,K1,T3))',
        'Current-Min(ang(K4,K1,T1),ang(K4,K1,T2),ang(K4,K1,T3))'
    ]
    config['4v3_next_state_transition_df_col_names'] = [
        'Next-dist(K1,C)',
        'Next-dist(K1,K2)',
        'Next-dist(K1,K3)',
        'Next-dist(K1,K4)',
        'Next-dist(K1,T1)',
        'Next-dist(K1,T2)',
        'Next-dist(K1,T3)',
        'Next-dist(K2,C)',
        'Next-dist(K3,C)',
        'Next-dist(K4,C)',
        'Next-dist(T1,C)',
        'Next-dist(T2,C)',
        'Next-dist(T3,C)',
        'Next-Min(dist(K2,T1),dist(K2,T2),dist(K2,T3))',
        'Next-Min(dist(K3,T1),dist(K3,T2),dist(K3,T3))',
        'Next-Min(dist(K4,T1),dist(K4,T2),dist(K4,T3))',
        'Next-Min(ang(K2,K1,T1),ang(K2,K1,T2),ang(K2,K1,T3))',
        'Next-Min(ang(K3,K1,T1),ang(K3,K1,T2),ang(K3,K1,T3))',
        'Next-Min(ang(K4,K1,T1),ang(K4,K1,T2),ang(K4,K1,T3))'
    ]
    config['4v3_full_transition_df_col_names'] = config['4v3_current_state_transition_df_col_names'] + [config['action_transition_df_col_name']] + config['4v3_next_state_transition_df_col_names']

    return config