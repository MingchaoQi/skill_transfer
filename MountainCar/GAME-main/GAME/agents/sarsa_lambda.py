# Author: Minh Hua
# Date: 10/24/2022
# Purpose: This module contains code for a Sarsa Lambda reinforcement learning agent using CMAC tile coding.
# Code obtained from Jeremy Zhang (https://github.com/MJeremy2017/reinforcement-learning-implementation) and modified.

from GAME.agents.TileCoding import *
from GAME.bin.intertask_mappings import IntertaskMapping
from GAME.utils.config import config
import numpy as np

class SarsaLambdaCMAC:
    """Base class for Sarsa lambda with CMAC"""
    def __init__(self,
        alpha:float,
        lamb:float,
        gamma:float,
        method:str,
        num_of_tilings:int,
        max_size:int
    ) -> None:
        """
        Description:
            Initializes a Sarsa(lambda) agent using CMAC tile coding.

        Arguments:
            alpha: the step size parameter.
            lambda: the trace decay rate.
            gamma: the discount rate.
            method: 'replacing' or 'accumulating'.
            num_of_tilings: the number of tilings used in CMAC.
            max_size: the maximum size of the weights and trace vectors.

        Return:
            (None)
        """
        # save class variables
        self.method = method # whether we are accumulating or replace traces
        self.alpha = alpha # global step size
        self.alpha_per_tile = alpha / num_of_tilings # step size for each tile
        self.lamb = lamb # trace decay rate
        self.gamma = gamma # discount rate
        self.max_size = max_size # the maximum size of the weights and trace vectors.
        self.num_of_tilings = num_of_tilings

        # initialize a hash table for use in tile coding
        self.hash_table = IHT(max_size)
        # initialize a set of weights for each tile
        self.weights = np.zeros(max_size)
        # trace vector
        self.z = np.zeros(max_size)

        # init config
        self.config_data = config()
    
    def update(self, active_tiles:list, target:float) -> None:
        """
        Description:
            Update the weights using the current state, action, and target value.

        Arguments:
            active_tiles: the tiles activated by the state and action. Returned by tiles from TileCoding.
            target: the target value.

        Return:
            (None)
        """
        # update the traces
        if self.method == 'accumulating':
            self.z *= self.gamma * self.lamb
            self.z[active_tiles] += 1
        elif self.method == 'replacing':
            active = np.isin(range(len(self.z)), active_tiles)
            self.z[active] = 1
            self.z[~active] *= self.gamma * self.lamb

        # update the weights
        estimate = np.sum(self.weights[active_tiles])
        delta = self.alpha_per_tile * (target - estimate)
        self.weights += self.alpha_per_tile * delta * self.z

class SarsaLambdaCMAC2DMountainCar(SarsaLambdaCMAC):
    """Class for Sarsa lambda with CMAC to learn 2D Mountain Car"""
    def __init__(self,
        alpha:float,
        lamb:float,
        gamma:float,
        method:str,
        epsilon:float,
        num_of_tilings:int,
        max_size:int) -> None:
        """
        Description:
            Initializes a Sarsa(lambda) agent using CMAC tile coding for 2D Mountain Car.

        Arguments:
            alpha: the step size parameter.
            lambda: the trace decay rate.
            gamma: the discount rate.
            method: 'replacing' or 'accumulating'.
            epsilon: the epsilon parameter for epsilon-greedy strategy of choosing actions.
            num_of_tilings: the number of tilings used in CMAC.
            max_size: the maximum size of the weights and trace vectors.

        Return:
            (None)
        """
        # inherent from parent class
        super(SarsaLambdaCMAC2DMountainCar, self).__init__(alpha, lamb, gamma, method, num_of_tilings, max_size)
        self.epsilon = epsilon
        self.actions = self.config_data['MC2D_action_values']

        # initialize variables specific to 2D mountain car
        # scale position and velocity for tile coding software
        self.min_position = -1.2
        self.max_position = 0.6
        self.min_speed = -0.07
        self.max_speed = 0.07
        self.position_scale = self.num_of_tilings / (self.max_position - self.min_position)
        self.velocity_scale = self.num_of_tilings / (self.max_speed - self.min_speed)
        self.goal_position = 0.5
        self.goal_velocity = 0.0

    def get_active_tiles(self, state:list, action:int) -> list:
        """
        Description:
            Get the indices of the active tiles for a given state and action.

        Arguments:
            state: a list of the current state variables in 2D Mountain Car.
            action: the current action.

        Return:
            (list) a list of the indices of the active tiles
        """
        # unpack state variables
        position, velocity = state
        normalized_state = [position * self.position_scale, velocity * self.velocity_scale]
        actions = [action]

        # use TileCoding module to get active tiles
        active_tiles = tiles(self.hash_table, self.num_of_tilings, normalized_state, actions)
        return active_tiles

    def get_value(self, state:list, action:int) -> float:
        """
        Description:
            Estimate the value of a given state and action.

        Arguments:
            state: a list of the current state variables in 2D Mountain Car.
            action: the current action.

        Return:
            (float) the value of the current state and action
        """
        # remember to return a 0.0 if we have already reached the goal state
        terminated = bool(
            # state[0] >= self.goal_position and state[1] >= self.goal_velocity
            state[0] >= self.goal_position
        )
        if terminated:
            return 0.0
        # else
        active_tiles = self.get_active_tiles(state, action)
        return np.sum(self.weights[active_tiles])

    def choose_action_eps_greedy(self, state:list) -> int:
        """
        Description:
            Chooses an action according to the epsilon-greedy strategy

        Arguments:
            state: a list of the current state variables in 2D Mountain Car.

        Return:
            (int) an integer representing the chosen action
        """
        # choose a random action
        if np.random.uniform(0, 1) <= self.epsilon:
            return np.random.choice(self.actions)
        # else, choose a greedy action
        else:
            # assign value to each action
            values = {}
            for a in self.actions:
                value_of_a = self.get_value(state, a)
                values[a] = value_of_a
            return np.random.choice([k for k, v in values.items() if v==max(values.values())])

class SarsaLambdaCMAC3DMountainCar(SarsaLambdaCMAC2DMountainCar):
    """Class for Sarsa lambda with CMAC to learn 3D Mountain Car"""
    def __init__(self,
        alpha:float,
        lamb:float,
        gamma:float,
        method:str,
        epsilon:float,
        num_of_tilings:int,
        max_size:int) -> None:
        """
        Description:
            Initializes a Sarsa(lambda) agent using CMAC tile coding for 3D Mountain Car.

        Arguments:
            alpha: the step size parameter.
            lambda: the trace decay rate.
            gamma: the discount rate.
            method: 'replacing' or 'accumulating'.
            epsilon: the epsilon parameter for epsilon-greedy strategy of choosing actions.
            num_of_tilings: the number of tilings used in CMAC.
            max_size: the maximum size of the weights and trace vectors.

        Return:
            (None)
        """
        # inherent from parent class
        super(SarsaLambdaCMAC3DMountainCar, self).__init__(alpha, lamb, gamma, method, epsilon, num_of_tilings, max_size)
        self.actions = self.config_data['MC3D_action_values']

        # initialize variables specific to 3D mountain car
        # scale position and velocity for tile coding software
        self.min_x_position = -1.2
        self.max_x_position = 0.6
        self.min_y_position = -1.2
        self.max_y_position = 0.6
        self.max_x_speed = 0.07
        self.min_x_speed = -0.07
        self.max_y_speed = 0.07
        self.min_y_speed = -0.07
        self.goal_x_position = 0.5
        self.goal_y_position = 0.5
        self.x_position_scale = self.num_of_tilings / (self.max_x_position - self.min_x_position)
        self.x_velocity_scale = self.num_of_tilings / (self.max_x_speed - self.min_x_speed)
        self.y_position_scale = self.num_of_tilings / (self.max_y_position - self.min_y_position)
        self.y_velocity_scale = self.num_of_tilings / (self.max_y_speed - self.min_y_speed)
        self.goal_velocity = 0.0

    def get_active_tiles(self, state:list, action:int) -> list:
        """
        Description:
            Get the indices of the active tiles for a given state and action.

        Arguments:
            state: a list of the current state variables in 3D Mountain Car.
            action: the current action.

        Return:
            (list) a list of the indices of the active tiles
        """
        # unpack state variables
        x_position, x_velocity, y_position, y_velocity = state
        normalized_state = [x_position * self.x_position_scale, x_velocity * self.x_velocity_scale,
                            y_position * self.y_position_scale, y_velocity * self.y_velocity_scale]
        actions = [action]

        # use TileCoding module to get active tiles
        active_tiles = tiles(self.hash_table, self.num_of_tilings, normalized_state, actions)
        return active_tiles

    def get_value(self, state:list, action:int) -> float:
        """
        Description:
            Estimate the value of a given state and action.

        Arguments:
            state: a list of the current state variables in 3D Mountain Car.
            action: the current action.

        Return:
            (float) the value of the current state and action
        """
        # remember to return a 0.0 if we have already reached the goal state
        terminated = bool(
            # state[0] >= self.goal_position and state[1] >= self.goal_velocity
            state[0] >= self.goal_x_position and state[2] >= self.goal_y_position
        )
        if terminated:
            return 0.0
        # else
        active_tiles = self.get_active_tiles(state, action)
        return np.sum(self.weights[active_tiles])

class SarsaLambdaCMAC3DMountainCarTransfer(SarsaLambdaCMAC3DMountainCar):
    """Class for Sarsa lambda with CMAC to learn 3D Mountain Car with transfer"""
    def __init__(self,
        alpha:float,
        lamb:float,
        gamma:float,
        method:str,
        epsilon:float,
        num_of_tilings:int,
        max_size:int,
        transfer_agent:SarsaLambdaCMAC2DMountainCar=None,
        mapping:IntertaskMapping=None) -> None:
        """
        Description:
            Initializes a Sarsa(lambda) agent using CMAC tile coding for 3D Mountain Car.

        Arguments:
            alpha: the step size parameter.
            lambda: the trace decay rate.
            gamma: the discount rate.
            method: 'replacing' or 'accumulating'.
            epsilon: the epsilon parameter for epsilon-greedy strategy of choosing actions.
            num_of_tilings: the number of tilings used in CMAC.
            max_size: the maximum size of the weights and trace vectors.
            transfer_agent: the agent to transfer from.

        Return:
            (None)
        """
        # inherent from parent class
        super(SarsaLambdaCMAC3DMountainCarTransfer, self).__init__(alpha, lamb, gamma, method, epsilon, num_of_tilings, max_size)
        self.transfer_agent = transfer_agent
        self.mapping = mapping

    def get_value(self, state:list, action:int) -> float:
        """
        Description:
            Estimate the value of a given state and action.

        Arguments:
            state: a list of the current state variables in 3D Mountain Car.
            action: the current action.

        Return:
            (float) the value of the current state and action
        """
        # remember to return a 0.0 if we have already reached the goal state
        terminated = bool(
            # state[0] >= self.goal_position and state[1] >= self.goal_velocity
            state[0] >= self.goal_x_position and state[2] >= self.goal_y_position
        )
        if terminated:
            return 0.0
        # else
        active_tiles = self.get_active_tiles(state, action)
        base_value = np.sum(self.weights[active_tiles])
        # add value from transfer agent
        # build mapped state variables and actions
        x, x_dot, y, y_dot = state
        src_action = self.mapping.action_mapping[action]
        # r2_values = [[0.9929816541449503, 0.9863922534862493, 0.9827540818142966],
        # [0.9898554363903732, 0.9807461375299111, 0.9727148311728532], [0.992743856876952, 0.9865786993775076, 0.9873256083696892],
        # [0.9938926191874691, 0.9863977078527576, 0.9832993967828477], [0.9813449416585194, 0.9650493513874667, 0.9363257347643477]
        # ]
        # r2_values = [[0.0118, 0.0079, 0.0103],
        # [0.0095, 0.0088, 0.0127], [0.0144, 0.0095, 0.0089],
        # [0.0099, 0.0093, 0.0135], [0.0136, 0.01, 0.01]
        # ]        
        # r2_sum = sum(1/val for val in r2_values[action])
        # # print("Base value: {}".format(base_value))
        # for src_action in [0, 1, 2]:
        #     # print("Source action: {}, Value: {}".format(src_action, self.transfer_agent.get_value([x, x_dot], src_action)))
        #     src_active_tiles = self.transfer_agent.get_active_tiles([x, x_dot], src_action)
        #     base_value += np.sum(self.transfer_agent.weights[src_active_tiles]) * (1 / r2_values[action][src_action]) * (1 / r2_sum)
        #     src_active_tiles = self.transfer_agent.get_active_tiles([y, y_dot], src_action)
        #     base_value += np.sum(self.transfer_agent.weights[src_active_tiles]) * (1 / r2_values[action][src_action]) * (1 / r2_sum)
        src_active_tiles = self.transfer_agent.get_active_tiles([x, x_dot], src_action)
        base_value += np.sum(self.transfer_agent.weights[src_active_tiles])
        src_active_tiles = self.transfer_agent.get_active_tiles([y, y_dot], src_action)
        base_value += np.sum(self.transfer_agent.weights[src_active_tiles])        
        return base_value

    def update(self, active_tiles:list, target:float, estimate:float) -> None:
        """
        Description:
            Update the weights using the current state, action, and target value.

        Arguments:
            active_tiles: the tiles activated by the state and action. Returned by tiles from TileCoding.
            target: the target value.

        Return:
            (None)
        """
        # update the traces
        if self.method == 'accumulating':
            self.z *= self.gamma * self.lamb
            self.z[active_tiles] += 1
        elif self.method == 'replacing':
            active = np.isin(range(len(self.z)), active_tiles)
            self.z[active] = 1
            self.z[~active] *= self.gamma * self.lamb

        # update the weights
        # estimate = np.sum(self.weights[active_tiles])
        delta = self.alpha_per_tile * (target - estimate)
        self.weights += self.alpha_per_tile * delta * self.z

class SarsaLambdaCMACPendulum(SarsaLambdaCMAC):
    """Class for Sarsa lambda with CMAC to learn Pendulum"""
    def __init__(self,
        alpha:float,
        lamb:float,
        gamma:float,
        method:str,
        epsilon:float,
        num_of_tilings:int,
        max_size:int) -> None:
        """
        Description:
            Initializes a Sarsa(lambda) agent using CMAC tile coding for Pendulum.

        Arguments:
            alpha: the step size parameter.
            lambda: the trace decay rate.
            gamma: the discount rate.
            method: 'replacing' or 'accumulating'.
            epsilon: the epsilon parameter for epsilon-greedy strategy of choosing actions.
            num_of_tilings: the number of tilings used in CMAC.
            max_size: the maximum size of the weights and trace vectors.

        Return:
            (None)
        """
        # inherent from parent class
        super(SarsaLambdaCMACPendulum, self).__init__(alpha, lamb, gamma, method, num_of_tilings, max_size)
        self.epsilon = epsilon
        self.actions = self.config_data['pendulum_action_values']

        # initialize variables specific to Pendulum
        # scale angle and velocity for tile coding software
        self.min_angle = -1.0
        self.max_angle = 1.0
        self.min_speed = -8.0
        self.max_speed = 8.0
        self.cos_th_scale = self.num_of_tilings / (self.max_angle - self.min_angle)
        self.sin_th_scale = self.num_of_tilings / (self.max_angle - self.min_angle)
        self.thdot_scale = self.num_of_tilings / (self.max_speed - self.min_speed)

    def get_active_tiles(self, state:list, action:int) -> list:
        """
        Description:
            Get the indices of the active tiles for a given state and action.

        Arguments:
            state: a list of the current state variables in Pendulum.
            action: the current action.

        Return:
            (list) a list of the indices of the active tiles
        """
        # unpack state variables
        cos_th, sin_th, thdot = state
        normalized_state = [cos_th * self.cos_th_scale, sin_th * self.sin_th_scale, thdot * self.thdot_scale]
        actions = [action]

        # use TileCoding module to get active tiles
        active_tiles = tiles(self.hash_table, self.num_of_tilings, normalized_state, actions)
        return active_tiles

    def get_value(self, state:list, action:int) -> float:
        """
        Description:
            Estimate the value of a given state and action.

        Arguments:
            state: a list of the current state variables in Pendulum.
            action: the current action.

        Return:
            (float) the value of the current state and action
        """
        # remember to return a 0.0 if we have already reached the goal state
        # terminated = bool(
        #     # state[0] >= self.goal_position and state[1] >= self.goal_velocity
        #     state[0] >= self.goal_position
        # )
        # if terminated:
        #     return 0.0
        # else
        active_tiles = self.get_active_tiles(state, action)
        return np.sum(self.weights[active_tiles])

    def choose_action_eps_greedy(self, state:list) -> int:
        """
        Description:
            Chooses an action according to the epsilon-greedy strategy

        Arguments:
            state: a list of the current state variables in Pendulum.

        Return:
            (int) an integer representing the chosen action
        """
        # choose a random action
        if np.random.uniform(0, 1) <= self.epsilon:
            return np.random.choice(self.actions)
        # else, choose a greedy action
        else:
            # assign value to each action
            values = {}
            for a in self.actions:
                value_of_a = self.get_value(state, a)
                values[a] = value_of_a
            return np.random.choice([k for k, v in values.items() if v==max(values.values())])        