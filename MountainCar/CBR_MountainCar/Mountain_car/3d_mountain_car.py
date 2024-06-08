# Author: Minh Hua
# Date: 10/18/2022
# Purpose: This module simulates the three-dimensional Mountain Car environment. Adapted from gym's implementation of two-dimensional Mountain Car.

import math
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

class MountainCar3DEnv(gym.Env):
    """
    ### Description

    The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
    at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
    that can be applied to the car in either direction. The goal of the MDP is to strategically
    accelerate the car to reach the goal state on top of the right hill. 

    This implementation extends the Mountain Car MDP in gym to a three-dimensional variant. The two-dimensional
    curve is extended to the surface sin(3x) + sin(3y). We add two state variables, the y position and the y velocity.
    We also add actions that move the car in the y direction. The goal state is once again on top of the hill, at 
    positions where x >= 0.5 and y >= 0.5.

    ### Observation Space

    The observation is a `ndarray` with shape `(4,)` where the elements correspond to the following:

    | Num | Observation                          | Min  | Max | Unit         |
    |-----|--------------------------------------|------|-----|--------------|
    | 0   | position of the car along the x-axis | -Inf | Inf | position (m) |
    | 1   | velocity of the car along the x-axis | -Inf | Inf | position (m) |
    | 2   | position of the car along the y-axis | -Inf | Inf | position (m) |
    | 4   | velocity of the car along the y-axis | -Inf | Inf | position (m) |   

    ### Action Space

    There are 5 discrete deterministic actions:

    | Num | Action             | Value | Unit         |
    |-----|-------------------------|-------|--------------|
    | 0   | Do not accelerate       | Inf   | position (m) |
    | 1   | Accelerate West         | Inf   | position (m) |
    | 2   | Accelerate East         | Inf   | position (m) |
    | 3   | Accelerate South        | Inf   | position (m) |
    | 4   | Accelerate North        | Inf   | position (m) |    

    ### Transition Dynamics:

    Given an action, the mountain car follows the following transition dynamics:

    *x_velocity<sub>t+1</sub> = x_velocity<sub>t</sub> + (action - 1) * force - cos(3 * x_position<sub>t</sub>) * gravity*

    *x_position<sub>t+1</sub> = x_position<sub>t</sub> + x_velocity<sub>t+1</sub>*

    *y_velocity<sub>t+1</sub> = y_velocity<sub>t</sub> + (action - 1) * force - cos(3 * y_position<sub>t</sub>) * gravity*

    *y_position<sub>t+1</sub> = y_position<sub>t</sub> + y_velocity<sub>t+1</sub>*

    where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity set to 0
    upon collision with the wall. The positions are clipped to the range `[-1.2, 0.6]` and
    velocities are clipped to the range `[-0.07, 0.07]`.


    ### Reward:

    The goal is to reach the flag placed on top of the right hill as quickly as possible, as such the agent is
    penalised with a reward of -1 for each timestep.

    ### Starting State

    The x and y positions of the car is assigned a uniform random value in *[-0.6 , -0.4]*.
    The starting velocity of the car is always assigned to 0.

    ### Episode End

    The episode ends if either of the following happens:
    1. Termination: The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
    2. Truncation: The length of the episode is 200.


    ### Arguments

    ```
    gym.make('MountainCar3D-v0')
    ```

    ### Version History

    * v0: Initial versions release (1.0.0)
    """
    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": 30
    }

    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0):
        self.min_x_position = -1.2
        self.max_x_position = 0.6
        self.min_y_position = -1.2
        self.max_y_position = 0.6
        self.max_x_speed = 0.07
        self.max_y_speed = 0.07
        self.goal_x_position = 0.5
        self.goal_y_position = 0.5
        self.goal_velocity = goal_velocity

        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.min_x_position, -self.max_x_speed, self.min_y_position, -self.max_y_speed], dtype=np.float32)
        self.high = np.array([self.max_x_position, self.max_x_speed, self.max_y_position, self.max_y_speed], dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.isopen = True

        # We have 5 actions, corresponding to "Neutral", "West", "East", "South", "North"
        self.action_space = spaces.Discrete(5)
        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the change in velocity if that action is taken.
        (以下字典将抽象动作从“self.action_space”映射到采取该动作时的速度变化。)
        i.e.
        Neutral does not modify the velocity
        West modifies the x velocity by -0.001
        East modifies the x velocity by 0.001
        South modifies the y velocity by -0.001
        North modifies the y velocity by 0.001
        """
        self._action_to_velocity_change = {
            0: np.array([1, 1]), # Neutral
            1: np.array([0, 1]), # West
            2: np.array([2, 1]), # East
            3: np.array([1, 0]), # South
            4: np.array([1, 2]) # North
        }

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, action: int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        # 获取状态变量
        x_position, x_velocity, y_position, y_velocity = self.state
        # 将x和y方向的动作映射为速度变化
        action_x, action_y = self._action_to_velocity_change[action]
        # x方向速度改变
        x_velocity += (action_x - 1) * self.force + math.cos(3 * x_position) * (-self.gravity)
        x_velocity = np.clip(x_velocity, -self.max_x_speed, self.max_x_speed)
        # y方向速度改变
        y_velocity += (action_y - 1) * self.force + math.cos(3 * y_position) * (-self.gravity)
        y_velocity = np.clip(y_velocity, -self.max_y_speed, self.max_y_speed)

        # change x position
        x_position += x_velocity
        x_position = np.clip(x_position, self.min_x_position, self.max_x_position)
        # change y position
        y_position += y_velocity
        y_position = np.clip(y_position, self.min_y_position, self.max_y_position)
        # reset velocity if we hit the edges that are not the goal state
        if x_position == self.min_x_position and x_velocity < 0: # this would move us out of the x = -1.2 edge
            x_velocity = 0
        if x_position == self.max_x_position and x_velocity > 0: # this moves us out of the x = 0.6 edge (before the goal state)
            x_velocity = 0
        if y_position == self.min_y_position and y_velocity < 0: # this moves us out of the y = -1.2 edge
            y_velocity = 0
        if y_position == self.max_y_position and y_velocity > 0: # this moves us out of the y = 0.6 edge (before the goal state is reached)
            y_velocity = 0

        # check termination
        terminated = bool(
            # x_position >= self.goal_x_position and x_velocity >= self.goal_velocity and y_position >= self.goal_y_position and y_velocity >= self.goal_velocity
            x_position >= self.goal_x_position and y_position >= self.goal_y_position
        )
        reward = -1.0
        if terminated:
            reward = 0.0
            x_velocity = 0
            y_velocity = 0
        info = self._get_info()

        # next state
        self.state = (x_position, x_velocity, y_position, y_velocity)
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, info
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        self.state = np.array([self.np_random.uniform(low=low, high=high), 0, self.np_random.uniform(low=low, high=high), 0]) # x, dot_x, y, dot_y

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}
    
    def _get_info(self):
        return {"z_position": math.sin(3 * self.state[0]) + math.sin(3 * self.state[2])}

    def render(self):
        # imports and initializations
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        # main render loop
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.max_x_position - self.min_x_position
        world_height = self.max_y_position - self.min_y_position
        x_scale = self.screen_width / world_width
        y_scale = self.screen_height / world_height
        carwidth = 20
        carheight = 20

        # background
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))
        # add contour plot
        # contour_plot = pygame.image.load("C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Evolutionary and Swarm Intelligence\\src\\GAME\\notebooks\\3d_mountain_car_surf.png")
        # charImage = pygame.transform.scale(charImage, charRect.size)
        # contour_plot = contour_plot.convert()

        x_position = self.state[0]
        y_position = self.state[2]

        # xs = np.linspace(self.min_x_position, self.max_x_position, 100)
        # ys = self._height(xs)
        # xys = list(zip((xs - self.min_x_position) * scale, ys * scale))

        # pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            # c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * x_position))
            coords.append(
                (
                    c[0] + (x_position - self.min_x_position) * x_scale,
                    c[1] + (y_position - self.min_y_position) * y_scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        # for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
        #     c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * x_position))
        #     wheel = (
        #         int(c[0] + (x_position - self.min_x_position) * scale),
        #         int(c[1] + clearance + self._height(x_position) * scale),
        #     )

        #     gfxdraw.aacircle(
        #         self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
        #     )
        #     gfxdraw.filled_circle(
        #         self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
        #     )

        flagx1 = int((self.goal_x_position - self.min_x_position) * x_scale)
        flagx2 = flagx1 + 50
        flagy1 = int((self.goal_y_position - self.min_y_position) * y_scale)
        # flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        # gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        # gfxdraw.aapolygon(
        #     self.surf,
        #     [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
        #     (204, 204, 0),
        # )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx1, flagy1), (flagx1, flagy2), (flagx2, flagy2), (flagx2, flagy1)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        # self.screen.blit(contour_plot, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False