"""
http://incompleteideas.net/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import math

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gym.utils import seeding


class MountainCar3DEnv(gym.Env):
    """
    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the east, west, north, south
        or cease any acceleration.

    Observation:
        Type: Box(2)
        Num    Observation                          Min            Max
        0      Car horizonal Position              -1.2           0.6
        1      Car horizonal Velocity              -0.07          0.07
        2      Car vertical Position               -1.2           0.6
        3      Car vertical Velocity               -0.07          0.07

    Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the west
        1      Accelerate to the east
        2      Don't accelerate
        3      Accelerate to the south
        4      Accelerate to the north

        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.

    Reward:
         Reward of 0 is awarded if the agent reached the flag (position = (0.5,0.5))
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than (0.5,0.5).

    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.

    Episode Termination:
         The car position is more than (0.5,0.5)
         Episode length is greater than 500
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, goal_velocity=0):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity

        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed, self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed, self.max_position, self.max_speed], dtype=np.float32)

        self.viewer = None

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        position_x, velocity_x, position_y, velocity_y = self.state
        if action == 0:
            velocity_x += (action - 1) * self.force + math.cos(3 * position_x) * (-self.gravity)
            velocity_y += math.cos(3 * position_y) * (-self.gravity)
        elif action == 1:
            velocity_x += action * self.force + math.cos(3 * position_x) * (-self.gravity)
            velocity_y += math.cos(3 * position_y) * (-self.gravity)
        elif action == 2:
            velocity_x += math.cos(3 * position_x) * (-self.gravity)
            velocity_y += math.cos(3 * position_y) * (-self.gravity)
        elif action == 3:
            velocity_x += math.cos(3 * position_x) * (-self.gravity)
            velocity_y += (action - 4) * self.force + math.cos(3 * position_y) * (-self.gravity)
        elif action == 4:
            velocity_x += math.cos(3 * position_x) * (-self.gravity)
            velocity_y += (action - 3) * self.force + math.cos(3 * position_y) * (-self.gravity)
        else:
            velocity_x += math.cos(3 * position_x) * (-self.gravity)
            velocity_y += math.cos(3 * position_y) * (-self.gravity)
        velocity_x = np.clip(velocity_x, -self.max_speed, self.max_speed)
        velocity_y = np.clip(velocity_y, -self.max_speed, self.max_speed)
        position_x += velocity_x
        position_y += velocity_y
        position_x = np.clip(position_x, self.min_position, self.max_position)
        position_y = np.clip(position_y, self.min_position, self.max_position)
        if position_x == self.min_position and velocity_x < 0:
            velocity_x = 0
        if position_y == self.min_position and velocity_y < 0:
            velocity_y = 0

        done = bool(position_x >= self.goal_position and velocity_x >= self.goal_velocity
                    and position_y >= self.goal_position and velocity_y >= self.goal_velocity)
        reward = -1.0

        self.state = (position_x, velocity_x, position_y, velocity_y)
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = np.array(
            [self.np_random.uniform(low=-0.51, high=-0.5), 0, self.np_random.uniform(low=-0.51, high=-0.5), 0])
        return np.array(self.state, dtype=np.float32)

    def reset_manual(self, start):
        self.state = np.array(
            [self.np_random.uniform(low=-0.51, high=-0.51), 0, self.np_random.uniform(low=-0.7, high=-0.3), 0])
        return np.array(self.state, dtype=np.float32)

    def _height(self, xs, ys):
        return np.sin(3 * xs) * 0.45 + np.sin(3 * ys) * 0.45 + 1.1
        # return np.sin(3 * xs)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 600

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = np.linspace(self.min_position, self.max_position, 100)
            zs = self._height(xs, ys)
            # xys = list(zip((xs - self.min_position) * scale, ys * scale))
            #
            # self.track = rendering.make_polyline(xys)
            # self.track.set_linewidth(4)
            # self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = (self.goal_position - self.min_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

        pos_x = self.state[0]
        pos_y = self.state[2]
        self.cartrans.set_translation(
            (pos_x - self.min_position) * scale, (pos_y - self.min_position) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos_x))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


