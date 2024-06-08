# Author: Minh Hua
# Date: 10/29/2022
# Purpose: This module contains data miners for saving data from RL runs or mining data from saved logs.

from collections import namedtuple
import pandas as pd
import numpy as np
import sys
import math
import os

from GAME.utils.helper_funcs import *

# Named tuple that stores metadata about an experiment
ExperimentInfo = namedtuple('ExperimentInfo', [
    'env_name', 'env_max_steps', 'env_seed',
    'env_max_episodes', 'agent_name'
])

# Named tuple that stores metadata about an agent
SarsaLambdaAgentInfo = namedtuple('SarsaLambdaAgentInfo', [
    'alpha', 'lamb', 'gamma', 'method',
    'epsilon', 'num_of_tilings', 'max_size'
])

class RLSamplesCollector:
    """This class collects samples from RL training episodes and saves them to a csv"""
    def __init__(
        self,
        experiment_info:ExperimentInfo,
        agent_info:SarsaLambdaAgentInfo,
        data_column_names:list,
        data_column_dtypes:list
    ) -> None:
        """
        Description:
            Initializes a Sarsa(lambda) agent using CMAC tile coding.

        Arguments:
            experiment_info: metadata about the experiment.
            agent_info: metadata about the agent.
            data_column_names: a list of column names for the data.
            data_column_dtypes: a list of dtypes for the columns.

        Return:
            (None)
        """
        # save metadata
        self.experiment_info = experiment_info
        self.agent_info = agent_info

        # build pandas DataFrame using data_column_names
        self.data_column_names = data_column_names
        self.data_column_dtypes = data_column_dtypes
        # data_column_info = {col_name : col_dtype for col_name, col_dtype in zip(data_column_names, data_column_dtypes)}
        self.data = []
        # self.data_df = pd.DataFrame({c: pd.Series(dtype=dt) for c, dt in data_column_info.items()})

    def log_data(self, data_pt:dict) -> None:
        """
        Description:
            Appends a data point to the dataframe.

        Arguments:
            data_pt: a dictionary where the keys are the column names and the items are the data values.

        Return:
            (None)
        """
        self.data.append(data_pt)

    def export_data(self, path:str, file_name:str) -> None:
        """
        Description:
            Saves the dataframe to an external file.

        Arguments:
            path: the path to save the dataframe.
            file_name: the name for the file.

        Return:
            (None)
        """
        data_df = pd.DataFrame(self.data, columns = self.data_column_names)
        data_df.to_csv(os.path.join(path, file_name), index = False)

    def write_metadata(self, path:str, file_name:str) -> None:
        """
        Description:
            Saves the metadata to an external file.

        Arguments:
            path: the path to save the metadata.
            file_name: the name for the file.

        Return:
            (None)
        """
        with open(path + file_name, 'w') as f:
            sep = '-------------------------------------------------------------'
            f.write('Experiment Info \n' + sep + '\n')
            for k, v in self.experiment_info._asdict().items():
                f.write('{} = {} \n'.format(k, v))
            f.write('\nAgent Info \n' + sep + '\n')
            for k, v in self.agent_info._asdict().items():
                f.write('{} = {} \n'.format(k, v))

class RCSLogMiner:
    """This class mines data from .rcg and .rcl logs from Robocup Soccer and creates datasets for training neural networks"""
    def __init__(self,
        rcg_csv_filepath:str,
        logs_folderpath:str,
        transition_df_col_names:list,
        transition_df_col_dtypes:list,
        num_keepers:int=3,
        num_takers:int=2,
        num_state_vars:int=13,
        num_actions:int=3
    ) -> None:
        """
        Description:
            Initializes a data miner to extract information from RCS logs.

        Arguments:
            

        Return:
            (None)
        """
        # save class variables
        self.rcg_csv_filepath = rcg_csv_filepath
        self.logs_folderpath = logs_folderpath
        self.num_keepers = num_keepers
        self.num_takers = num_takers
        self.num_state_vars = num_state_vars
        self.num_actions = num_actions

        # helper constants
        self.MAX_INT = sys.maxsize * 2 + 1
        self.center_of_field = (0, 0)
        self.ball_kickable_dist = 0.03 + 0.085 + 0.7
        self.keeper_ids = [i for i in range(1, num_keepers + 1)]
        self.taker_ids = [i for i in range(num_keepers + 1, num_keepers + num_takers + 1)]

        # create a dataframe for the rcg file
        rcg_usable_cols = ['show_time', 'ball_x', 'ball_y', 'ball_vx', 'ball_vy']
        for i in range(1, num_takers + num_keepers + 1):
            rcg_usable_cols.append('player_l{}_state'.format(i))
            rcg_usable_cols.append('player_l{}_x'.format(i))
            rcg_usable_cols.append('player_l{}_y'.format(i))
            rcg_usable_cols.append('player_l{}_vx'.format(i))
            rcg_usable_cols.append('player_l{}_vy'.format(i))
        self.rcg_df = pd.read_csv(rcg_csv_filepath, index_col=False, usecols = rcg_usable_cols)
        self.rcg_df = self.rcg_df.dropna(axis=1) # drop empty columns
        self.rcg_df_len = len(self.rcg_df)

        # load the draw log files which holds the actions
        # because there are mismatches in the keeper naming convention between the draw.log and rcg files, we must resolve them
        self.draw_logs = {}
        for i in range(1, num_keepers + 1):
            dl_df, kp_idx = self.parse_draw_log(os.path.join(logs_folderpath, 'k{}-draw.log'.format(i)))
            print(kp_idx)
            if kp_idx == -1:
                raise ValueError("I could not identify the keeper in the k{}-draw.log file.".format(i))
            self.draw_logs[kp_idx] = dl_df
        if len(self.draw_logs) != num_keepers:
            raise ValueError("One of the draw logs were overriden with the same keeper id.")

        # create an empty output dataframe that contains the (s, a, s') triples
        transition_df_col_info = {col_name : col_dtype for col_name, col_dtype in zip(transition_df_col_names, transition_df_col_dtypes)}
        self.transition_df = pd.DataFrame({c: pd.Series(dtype=dt) for c, dt in transition_df_col_info.items()})
        self.transition_df_col_names = transition_df_col_names
        # fill dataframe with data parsed from the rcg file
        self.parse_rcg_and_fill_trans_df()

    def get_sorted_dist_between_cur_and_others(self, cur_pos:tuple, other_pos:list) -> list:
        """
        Description:
            Compute and sort the distances between a reference point and other points.

        Arguments:
            cur_pos: the current position in (x, y) format.
            other_pos: a list of the other positions also in (x, y) format.

        Return:
            (list) sorted list of the distances between a reference point and other points.
        """
        distances = [distance(cur_pos, o_pos) for o_pos in other_pos]
        return sorted(distances)

    def get_sorted_angle_between_cur_and_others(self, cur_tm_pos:tuple, kp_w_ball_pos:tuple, other_pos:list) -> list:
        """
        Description:
            Compute and sort the angles between a reference point and other points.

        Arguments:
            cur_tm_pos: the current position in (x, y) format.
            kp_w_ball_pos: the position of the keeper with the ball in (x, y) format.
            other_pos: a list of the other positions also in (x, y) format.

        Return:
            (list) sorted list of the angles between a reference point and other points.
        """
        angles = [angle(cur_tm_pos, kp_w_ball_pos, o_pos) for o_pos in other_pos]
        return sorted(angles)

    def get_sorted_others_idx_to_cur(self, cur_pos:tuple, other_pos:list) -> list:
        """
        Description:
            Rank other players according to their distance to a reference point.

        Arguments:
            cur_pos: the current position in (x, y) format.
            other_pos: a list of the other positions also in (x, y) format.

        Return:
            (list) a sorted list of indices that are closest to a reference point.
        """
        distances = [distance(cur_pos, o_pos) for o_pos in other_pos]
        return np.argsort(distances)
    
    def get_sorted_tm_idx_to_cur(self, cur_pos:tuple, player_positions:list) -> list:
        """
        Description:
            Rank the closest teammates to a player.

        Arguments:
            cur_pos: the current position in (x, y) format.
            player_positions: a list of the players' positions in (x, y) format. Players usually denotes keepers.

        Return:
            (list) sorted list of the indexes of the closest teammates to the current player.
        """
        distances = [distance(cur_pos, other_pos) for other_pos in player_positions]
        return np.argsort(distances)[1:] # drop the first index as it counts the current keeper since he is the closest to himself

    def get_closest_to_ball(self, player_positions:list, ball_loc:tuple) -> tuple:
        """
        Description:
            Get the index of the closest player to a ball.

        Arguments:
            player_positions: a list of the players' positions in (x, y) format. Players usually denotes keepers.
            ball_loc: the location of the ball in (x, y) format.

        Return:
            (tuple) the index of the closest player to a ball and the distance.
        """
        min_dist = self.MAX_INT
        min_player_id = -1
        for player_idx, player_pos in enumerate(player_positions):
            distance_to_ball = distance(player_pos, ball_loc)
            if distance_to_ball < min_dist:
                min_dist = distance_to_ball
                min_player_id = player_idx
        return min_player_id, min_dist

    # def parse_rcg_and_fill_trans_df(self) -> None:
    #     """
    #     Description:
    #         Fills the transition dataframe with data from the rcg file.

    #     Arguments:
    #         None

    #     Return:
    #         (None)
    #     """
    #     transition_data = []
    #     for index, row in self.rcg_df.iterrows():
    #         # get the keeper and taker positions
    #         keeper_positions = [(row['player_l{}_x'.format(id)], row['player_l{}_y'.format(id)]) for id in self.keeper_ids]
    #         taker_positions = [(row['player_l{}_x'.format(id)], row['player_l{}_y'.format(id)]) for id in self.taker_ids]
    #         # get the ball position
    #         ball_loc = (row['ball_x'], row['ball_y'])
    #         # find the closest keeper to the ball
    #         closest_keeper_idx, closest_distance_to_ball = self.get_closest_to_ball(keeper_positions, ball_loc)
    #         ball_velocity = math.sqrt(row['ball_vx']**2 + row['ball_vy']**2)
    #         # mark the cycle
    #         cycle = row['show_time']
    #         if closest_distance_to_ball <= self.ball_kickable_dist and ball_velocity <= 2 and int(cycle) < self.rcg_df_len - 1: # we only fill information if a keeper actually has the ball
    #             # save the current state variables
    #             current_state = self.build_state_data_from_rcg_row(keeper_positions, taker_positions, closest_keeper_idx)
    #             # get a list of teammate idxes sorted according to distance to the keeper with the ball
    #             sorted_tm_idxes = self.get_sorted_tm_idx_to_cur(keeper_positions[closest_keeper_idx], keeper_positions)
    #             action = self.parse_action(closest_keeper_idx + 1, cycle, sorted_tm_idxes)
    #             # print('Cycle: {}, Keeper {}, Action: {}'.format(cycle, closest_keeper_idx + 1, action))
    #             # get next state preemptively
    #             next_keeper_positions =  [(self.rcg_df.at[index + 1, 'player_l{}_x'.format(id)], self.rcg_df.at[index + 1, 'player_l{}_y'.format(id)]) for id in self.keeper_ids]
    #             next_taker_positions = [(self.rcg_df.at[index + 1, 'player_l{}_x'.format(id)], self.rcg_df.at[index + 1, 'player_l{}_y'.format(id)]) for id in self.taker_ids]
    #             next_state = self.build_state_data_from_rcg_row(next_keeper_positions, next_taker_positions, closest_keeper_idx)
                
    #             # append (s, a, s') to dataframe
    #             concatenated_data = [cycle, closest_keeper_idx + 1] + current_state + [action] + next_state
    #             transition_data.append(concatenated_data)
    #             # data_dict = {data_col : data for data_col, data in zip(self.transition_df.columns, concatenated_data)}
    #             # self.transition_df.append(data_dict, ignore_index = True)
    #     self.transition_df = pd.DataFrame(transition_data, columns = self.transition_df_col_names)

    def parse_rcg_and_fill_trans_df(self) -> None:
        """
        Description:
            Fills the transition dataframe with data from the rcg file.

        Arguments:
            None

        Return:
            (None)
        """
        transition_data = []
        previous_keeper = -1
        for index, row in self.rcg_df.iterrows():
            # get the keeper and taker positions
            keeper_positions = [(row['player_l{}_x'.format(id)], row['player_l{}_y'.format(id)]) for id in self.keeper_ids]
            taker_positions = [(row['player_l{}_x'.format(id)], row['player_l{}_y'.format(id)]) for id in self.taker_ids]
            # get the ball position
            ball_loc = (row['ball_x'], row['ball_y'])
            # find the closest keeper to the ball
            closest_keeper_idx, closest_distance_to_ball = self.get_closest_to_ball(keeper_positions, ball_loc)
            ball_velocity = math.sqrt(row['ball_vx']**2 + row['ball_vy']**2)
            # mark the cycle
            cycle = row['show_time']
            # save the current closest keeper
            if closest_distance_to_ball <= self.ball_kickable_dist and ball_velocity <= 2:
                previous_keeper = closest_keeper_idx
            # reset the previous_keeper if the ball velocity is zero, indicating that the episode has ended
            if row['ball_vx'] == 0 and row['ball_vy'] == 0:
                previous_keeper = -1
            if previous_keeper != -1 and int(cycle) < self.rcg_df_len - 1: # we only fill information if a keeper actually has the ball
                # save the current state variables
                current_state = self.build_state_data_from_rcg_row(keeper_positions, taker_positions, previous_keeper)
                # get a list of teammate idxes sorted according to distance to the keeper with the ball
                sorted_tm_idxes = self.get_sorted_tm_idx_to_cur(keeper_positions[previous_keeper], keeper_positions)
                action = self.parse_action(previous_keeper + 1, cycle, sorted_tm_idxes)
                # if previous_action == -1 or (action != -1 and action != previous_action): # save the previous action of the previous keeper
                #     previous_action = action
                # print('Cycle: {}, Keeper {}, Action: {}'.format(cycle, closest_keeper_idx + 1, action))
                # get next state preemptively
                next_keeper_positions =  [(self.rcg_df.at[index + 1, 'player_l{}_x'.format(id)], self.rcg_df.at[index + 1, 'player_l{}_y'.format(id)]) for id in self.keeper_ids]
                next_taker_positions = [(self.rcg_df.at[index + 1, 'player_l{}_x'.format(id)], self.rcg_df.at[index + 1, 'player_l{}_y'.format(id)]) for id in self.taker_ids]
                next_state = self.build_state_data_from_rcg_row(next_keeper_positions, next_taker_positions, previous_keeper)
                
                # append (s, a, s') to dataframe
                concatenated_data = [cycle, previous_keeper + 1] + current_state + [action] + next_state
                transition_data.append(concatenated_data)
                # data_dict = {data_col : data for data_col, data in zip(self.transition_df.columns, concatenated_data)}
                # self.transition_df.append(data_dict, ignore_index = True)
        self.transition_df = pd.DataFrame(transition_data, columns = self.transition_df_col_names)

    def parse_action(self, kp_id:int, cycle:int, closest_tms:list) -> int:
        """
        Description:
            Determines the action that keeper with kp_id took by parsing the corresponding draw log.

        Arguments:
            kp_id: the ID of the keeper who took the action.
            cycle: the cycle that the action executed.
            closest_tms: array of teammate ids ranked by distance.

        Return:
            (int) the id of the executed action.
        """
        kp_dl_df = self.draw_logs[kp_id]
        kp_dl_at_cycle = kp_dl_df[kp_dl_df['Cycle'] == cycle]['State'].to_list()
        for state in kp_dl_at_cycle:
            # if the action is "holding", action = 0
            if state == "holding":
                return 0
            # else if the action is in the form "(p l j)", then action = index of keeper j in the closest_tms list
            elif "(p" in state:
                idx_of_target = int(state[1:-1].split(" ")[2]) - 1
                # find target_id in closest_tms array to determine rank of target keeper
                return np.where(closest_tms == idx_of_target)[0][0] + 1
        return -1 # error

    def build_state_data_from_rcg_row(self, keeper_positions:list, taker_positions:list, kp_w_ball_idx:int) -> list:
        """
        Description:
            Reads a row from the rcg dataframe and extracts the relevant state information.

        Arguments:
            keeper_positions: the positions of the keepers in (x, y) format.
            taker_positions: the positions of the takers in (x, y) format.
            closest_keeper_idx: the index of the keeper with the ball.

        Return:
            (list) a list of state variables for the transition dataframe.
        """
        data_for_trans_df = [] # initialize a list to store data to store to the transition df
        kp_w_ball_loc = keeper_positions[kp_w_ball_idx]
        # get a list of teammate idxes sorted according to distance to the keeper with the ball
        sorted_tm_idxes = self.get_sorted_tm_idx_to_cur(kp_w_ball_loc, keeper_positions)
        # get a list of taker idxes sorted according to distance to the current keeper
        sorted_tk_idxes = self.get_sorted_others_idx_to_cur(kp_w_ball_loc, taker_positions)
        # save 'dist(K1,C)'
        data_for_trans_df.append(distance(kp_w_ball_loc, self.center_of_field))
        # distance between current keeper and the teammates, sorted according to distance
        tm_dist_to_cur = [] # distance between keeper i and the keeper with the ball
        tm_dist_to_c = [] # distance between keeper i and center of field
        min_tm_dist_tk = [] # minimum distance between keeper i and the other takers
        min_tm_ang_tk = [] # minimum angle between keeper i and the other takers
        for tm_idx in sorted_tm_idxes:
            tm_loc = keeper_positions[tm_idx]
            tm_dist_to_cur.append(distance(tm_loc, kp_w_ball_loc))
            tm_dist_to_c.append(distance(tm_loc, self.center_of_field))
            # compute min distance between teammates and opponents
            min_tm_dist_tk.append(self.get_sorted_dist_between_cur_and_others(tm_loc, taker_positions)[0])
            # compute min angle between teammates and opponents
            min_tm_ang_tk.append(self.get_sorted_angle_between_cur_and_others(tm_loc, kp_w_ball_loc, taker_positions)[0])
        # distance between current keeper and takers, sorted according to distance
        tk_dist_to_cur = [] # distance between taker i and the keeper with the ball 
        tk_dist_to_c = [] # distance between taker i and center of field
        for tk_idx in sorted_tk_idxes:
            tk_loc = taker_positions[tk_idx]
            tk_dist_to_cur.append(distance(tk_loc, kp_w_ball_loc))
            tk_dist_to_c.append(distance(tk_loc, self.center_of_field))
        # save distances that were just calculated
        concat_dist_list = tm_dist_to_cur + tk_dist_to_cur + tm_dist_to_c + tk_dist_to_c + min_tm_dist_tk + min_tm_ang_tk
        for d in concat_dist_list:
            data_for_trans_df.append(d)
        
        return data_for_trans_df

    def parse_draw_log(self, draw_log_path:str, only_save_state:bool=True) -> tuple:
        """
        Description:
            Parses a draw.log file and converts it into a dataframe.

        Arguments:
            draw_log_path: the path to the draw.log file.
            only_save_actions: only save the data instances that relate to the state or the action of the keeper.

        Return:
            (pd.DataFrame, int) the parsed draw.log file as a pandas DataFrame and the id of the keeper that owns the draw log.
        """
        data_column_names = ['Cycle', 'State']
        data_column_dtypes = ['int', 'str']
        data_column_info = {col_name : col_dtype for col_name, col_dtype in zip(data_column_names, data_column_dtypes)}
        df = pd.DataFrame({c: pd.Series(dtype=dt) for c, dt in data_column_info.items()})
        # keep track of the first cycle we see an action so we can identify the keeper
        cycle_of_first_action = -1
        seen_first_action = False
        kp_id = -1
        with open(draw_log_path, 'r') as f:
            # lines = f.readlines()
            for line in f:
                line_split = line.split(' ')
                if len(line_split) > 1 and line_split[1] == '\"state\"':
                    try:
                        cycle = int(line_split[0][:-1])
                        state = line.split('\"')[3]
                        # check to see if we can id the keeper
                        if '(' in state and not seen_first_action:
                            cycle_of_first_action = cycle + 1
                            seen_first_action = True
                            # check the rcg dataframe for the id of the keeper who also took an action at the specified time
                            rcg_at_time_slice = self.rcg_df[self.rcg_df['show_time'] == cycle_of_first_action]
                            for col in rcg_at_time_slice.columns:
                                if rcg_at_time_slice[col].to_list()[0] == '0x3':
                                    kp_id = int(col.split('_')[1][1:])
                        df = df.append({'Cycle': cycle, 'State': state}, ignore_index=True)
                    except IndexError:
                        pass
        return df, kp_id

    def export_data(self, path:str, file_name:str) -> None:
        """
        Description:
            Saves the dataframe to an external file.

        Arguments:
            path: the path to save the dataframe.
            file_name: the name for the file.

        Return:
            (None)
        """
        self.transition_df.to_csv(os.path.join(path, file_name), index = False)

if __name__ == '__main__':
    from GAME.utils.config import config
    config_data = config()

    test = 1
    if test == 0:
        experiment_name = "202211071939-UbuntuXenialSmall"
        rcg_csv_filepath = config_data['logs_path'] + "{}\\{}.rcg.csv".format(experiment_name, experiment_name)
        logs_folderpath = config_data['logs_path'] + experiment_name
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
        num_keepers = 3
        num_takers = 2
        num_state_vars = 13
        num_actions = 3
        log_miner = RCSLogMiner(rcg_csv_filepath, logs_folderpath, transition_df_col_names, transition_df_col_dtypes, num_keepers, num_takers, num_state_vars, num_actions)
        csv_out_path = config_data['output_path'] + "11072022 3v2 RCS Sample Data Collection"
        csv_out_name = 'keepaway_3v2_transitions.csv'
        log_miner.export_data(csv_out_path, csv_out_name)
    elif test == 1:
        experiment_name = "202211071930-UbuntuXenialSmall"
        rcg_csv_filepath = config_data['logs_path'] + "{}\\{}.rcg.csv".format(experiment_name, experiment_name)
        logs_folderpath = config_data['logs_path'] + experiment_name
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
        csv_out_path = config_data['output_path'] + "11072022 4v3 RCS Sample Data Collection"
        csv_out_name = 'keepaway_4v3_transitions_v2.csv'
        log_miner.export_data(csv_out_path, csv_out_name)