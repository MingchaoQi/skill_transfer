from GAME.utils.config import config
from GAME.utils.data_miners import RCSLogMiner
import os
import pandas as pd

config_data = config()

test = 3
if test == 0:
    all_logs_folderpath = os.path.join(config_data['logs_path'], "logs_3v2_10x350eps_learned")
    experiments = []
    for file in os.listdir(all_logs_folderpath):
        if file.endswith('.kwy') and file.split('.')[0] not in experiments:
            experiments.append(file.split('.')[0])

    for experiment_name in experiments:
        print(experiment_name)
        rcg_csv_filepath = os.path.join(all_logs_folderpath, "{}.rcg.csv".format(experiment_name))
        logs_folderpath = os.path.join(all_logs_folderpath, experiment_name)
        num_keepers = 3
        num_takers = 2
        num_state_vars = 13
        num_actions = 3
        transition_df_col_names = ['Cycle', 'ID_kp_w_ball']
        transition_df_col_names = transition_df_col_names + config_data['3v2_full_transition_df_col_names']
        transition_df_col_dtypes = [
            'int', 'int', 'float', 'float', 'float', 'float', 'float',
            'float', 'float', 'float', 'float', 'float',
            'float', 'float', 'float',
            'int', 'float', 'float', 'float', 'float', 'float',
            'float', 'float', 'float', 'float', 'float',
            'float', 'float', 'float'
        ]
        try:
            log_miner = RCSLogMiner(rcg_csv_filepath, logs_folderpath, transition_df_col_names, transition_df_col_dtypes, num_keepers, num_takers, num_state_vars, num_actions)
            csv_out_path = all_logs_folderpath
            csv_out_name = 'keepaway_3v2_transitions_{}.csv'.format(experiment_name)
            log_miner.export_data(csv_out_path, csv_out_name)
        except:
            pass
elif test == 1:
    all_logs_folderpath = os.path.join(config_data['logs_path'], "logs_4v3_6x350eps_random")
    experiments = []
    for file in os.listdir(all_logs_folderpath):
        if file.endswith('.kwy') and file.split('.')[0] not in experiments:
            experiments.append(file.split('.')[0])

    for experiment_name in experiments:
        print(experiment_name)
        rcg_csv_filepath = os.path.join(all_logs_folderpath, "{}.rcg.csv".format(experiment_name))
        logs_folderpath = os.path.join(all_logs_folderpath, experiment_name)
        num_keepers = 4
        num_takers = 3
        num_state_vars = 19
        num_actions = 4
        transition_df_col_names = ['Cycle', 'ID_kp_w_ball']
        transition_df_col_names = transition_df_col_names + config_data['4v3_full_transition_df_col_names']
        transition_df_col_dtypes = [
            'int', 'int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'int',
                'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float'
        ]
        log_miner = RCSLogMiner(rcg_csv_filepath, logs_folderpath, transition_df_col_names, transition_df_col_dtypes, num_keepers, num_takers, num_state_vars, num_actions)
        csv_out_path = all_logs_folderpath
        csv_out_name = 'keepaway_4v3_transitions_{}.csv'.format(experiment_name)
        log_miner.export_data(csv_out_path, csv_out_name)
elif test == 2:
    all_logs_folderpath = os.path.join(config_data['logs_path'], "logs_4v3_6x350eps_random")
    experiments = []
    for file in os.listdir(all_logs_folderpath):
        if file.endswith('.kwy') and file.split('.')[0] not in experiments:
            experiments.append(file.split('.')[0])

    # read in the first dataframe as the starter
    trans_csv_filename = os.path.join(all_logs_folderpath, 'keepaway_4v3_transitions_{}.csv'.format(experiments[0]))
    trans_df = pd.read_csv(trans_csv_filename, index_col = False)
    for exp_idx in range(1, len(experiments)):
        trans_csv_filename = os.path.join(all_logs_folderpath, 'keepaway_4v3_transitions_{}.csv'.format(experiments[exp_idx]))
        temp_df = pd.read_csv(trans_csv_filename, index_col = False)
        # get the last cycle in the total dataframe
        last_cycle = trans_df['Cycle'].to_list()[-1]
        temp_df['Cycle'] = temp_df['Cycle'] + last_cycle
        # merge the two dataframes
        trans_df = pd.concat([trans_df, temp_df], ignore_index=True, sort=False)
    
    # write df to file
    csv_out_path = all_logs_folderpath
    csv_out_name = 'keepaway_4v3_transitions_v3.csv'
    trans_df.to_csv(os.path.join(csv_out_path, csv_out_name), index = False)
elif test == 3:
    all_logs_folderpath = os.path.join(config_data['logs_path'], "12142022_3v2_logs_random_explore")
    experiments = []
    for file in os.listdir(all_logs_folderpath):
        if file.endswith('.kwy') and file.split('.')[0] not in experiments:
            experiments.append(file.split('.')[0])

    # read in the first dataframe as the starter
    trans_csv_filename = os.path.join(all_logs_folderpath, 'keepaway_3v2_transitions_{}.csv'.format(experiments[0]))
    trans_df = pd.read_csv(trans_csv_filename, index_col = False)
    for exp_idx in range(1, len(experiments)):
        try:
            trans_csv_filename = os.path.join(all_logs_folderpath, 'keepaway_3v2_transitions_{}.csv'.format(experiments[exp_idx]))
            temp_df = pd.read_csv(trans_csv_filename, index_col = False)
            # get the last cycle in the total dataframe
            last_cycle = trans_df['Cycle'].to_list()[-1]
            temp_df['Cycle'] = temp_df['Cycle'] + last_cycle
            # merge the two dataframes
            trans_df = pd.concat([trans_df, temp_df], ignore_index=True, sort=False)
        except:
            pass
    
    # write df to file
    csv_out_path = all_logs_folderpath
    csv_out_name = 'keepaway_3v2_transitions.csv'
    trans_df.to_csv(os.path.join(csv_out_path, csv_out_name), index = False)

