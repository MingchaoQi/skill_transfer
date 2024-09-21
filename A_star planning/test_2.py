import numpy as np
import robosuite as suite
from scipy.spatial.transform import Rotation as R
import robosuite_task_zoo
import matplotlib.pyplot as plt
from robosuite.utils.control_utils import *
import robosuite.utils.transform_utils as T
from motion_planning_v2 import Motion_planning

robots = "IIWA"
env = robosuite_task_zoo.environments.manipulation.HammerPlaceEnv(
    robots,
    has_renderer=True,

    # gripper_types="RobotiqThreeFingerGripper",
    has_offscreen_renderer=True,
    use_camera_obs=True,
    render_camera="frontview",
    control_freq=20,
    controller_configs=suite.load_controller_config(
        default_controller="OSC_POSE"
    ),  # 操作空间位置控制
)
action_dim = env.action_dim
neutral = np.zeros(action_dim)
action = neutral.copy()
env.reset()

print(env.door_frame_pos, "门框位置坐标")
print(env.door_pos, "门位置坐标")
print(env.door_pos - env.door_frame_pos, "door相对位置坐标1")
print(env.door_latch_pos, "门把手位置坐标")
print(env.door_latch_pos - env.door_pos, "door相对位置坐标2")
print(env._handle_xpos, "门把手site位置坐标")
print(env._handle_xpos - env.door_latch_pos, "door相对位置坐标3")
# #
print(env.cabinet_pos, "抽屉框位置坐标")
print(env.drawer_link_pos, "带把手抽屉位置坐标")
print(env.drawer_link_pos - env.cabinet_pos, "cabinet相对位置坐标")
print(env._slide_handle_xpos, "抽屉把手位置坐标")
print(env._slide_handle_xpos - env.drawer_link_pos, "cabinet相对位置坐标2")
print(env.door_latch_pos - env._slide_handle_xpos, "门把手和抽屉把手相对位置坐标")

print(env.door_joint_pos, "门关节位置坐标")
print(env.cabinet_joint_pos, "抽屉关节的坐标")
env.close()