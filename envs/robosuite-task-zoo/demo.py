import numpy as np

import robosuite
import robosuite_task_zoo

robots = "IIWA"
env = robosuite_task_zoo.environments.manipulation.HammerPlaceEnv(
    robots,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
)

# reset env
env.reset()
env.viewer.set_camera(camera_id=0)

# get action limits
low, high = env.action_spec

for i in range(10000000000):
    action = np.random.uniform(low, high)
    obs, reward, done, _ = env.step(action)
    env.render()

env.close()