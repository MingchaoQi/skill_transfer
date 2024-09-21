import numpy as np
import robosuite as suite
from scipy.spatial.transform import Rotation as R
import robosuite_task_zoo
import matplotlib.pyplot as plt
from robosuite.utils.control_utils import *
import robosuite.utils.transform_utils as T
from motion_planning_v2 import Motion_planning

"""
需要完成的工作：
2024.5.3：
1. 使用知识库自动生成起始点与目标点。
2. 根据知识库和物理一致性完成阻抗控制参数的迁移。
3. 使用RL方法根据得到的力传感器数据完成开门过程的优化。
"""


def PD_control(last_pos, cur_pos, goal_pos, goal_vel, kp, kd):  # PD控制生成控制指令
    cur_vel = (cur_pos - last_pos) * 20
    a = kp * (goal_pos - cur_pos) + kd * (goal_vel - cur_vel)
    return a


def vel_control(cur_pos, goal_pos, k):
    a = k * (goal_pos - cur_pos)

    return a


def Force_control(xe, dxe, Fe, Mp, Bp, Cp, freq):
    ddxe = (Fe - Bp * dxe - Cp * xe) / Mp
    dxe_next = dxe + ddxe / freq
    xe_next = xe + dxe_next / freq

    return xe_next


def trace_trajectory_Astar(
    d0, goal_pos, tf, freq, gri_open=True
):  # 达不到柔顺控制的要求（姿态方面达不到，位置方面可以达到）
    plan = Motion_planning(env=env, dx=0.01, dy=0.01, dz=0.01, gripper_open=gri_open)
    Points_recover = plan.path_searching(start=d0, end=goal_pos)
    if Points_recover is not None:
        X, Y, Z = plan.path_smoothing(
            Path_points=Points_recover, t_final=tf, freq=freq
        )  ##轨迹使用二次B样条曲线进行平滑处理
    else:
        print("the path is not found!!")
        return None
    t0 = env.sim.data.time
    action = np.zeros(action_dim)
    goal_force = np.array([0, 0, 0])
    last_pos = np.append(d0, np.zeros(action_dim - 3))
    print('------------------------------------------------------', Points_recover)

    observation = []
    Reward = []
    Force = np.zeros(0)

    cp = np.array([2000, 2000, 2000])
    mp = np.array([1, 1, 1]) * 80
    bp = 2 * 0.707 * np.sqrt(cp * mp)

    for i in range(int(tf * freq * 1.2)):
        if i < len(plan.d):
            ##基于位置的阻抗控制方法
            # # MyEuler1 = R.from_quat(env._eef_xquat).as_euler('zyx')
            # MyEuler1 = T.quat2axisangle(env._eef_xquat)
            current_pos = np.append(env._eef_xpos, np.zeros(action_dim - 3))
            # current_force = env.sim.data.sensordata[0:3]
            # xe = current_pos[0:3] - plan.d[i] + plan.d_dot[i] /freq
            # dxe = (current_pos[0:3] - last_pos[0:3]) * freq - plan.d_dot[i] + plan.d_ddot[i] / freq
            # Fe = current_force - goal_force
            # xd = Force_control(xe, dxe, Fe, mp, bp, cp, freq)
            # x_s = np.append(plan.d[i] + xd, np.zeros(action_dim - 3))
            # 位置控制(d[i]代表插值平滑后的轨迹点)
            x_s = np.append(plan.d[i], np.zeros(action_dim - 3))
            v_s = np.append(plan.d_dot[i], np.zeros(action_dim - 3))
            while (env.sim.data.time - t0) < plan.t[i]:
                MyEuler1 = R.from_quat(env._eef_xquat).as_euler("zyx")
                # MyEuler1 = T.mat2euler(T.quat2mat(env._eef_xquat))
                current_pos = np.append(env._eef_xpos, np.zeros(action_dim - 3))
                kp = np.array([20, 20, 20, 0, 0, 0])
                kp = np.append(kp, np.ones(action_dim - 6))
                # action = vel_control(current_pos, x_s, k=kp)
                kd = 0.7 * np.sqrt(kp)
                action = PD_control(last_pos, current_pos, x_s, v_s, kp, kd)

                obs, reward, done, info = env.step(
                    action
                )  # take action in the environment
                observation.append(obs)
                Reward.append(reward)
                env.render()  # render on display
        else:
            goal_pos = x_s
            MyEuler1 = R.from_quat(env._eef_xquat).as_euler("zyx")
            # MyEuler1 = T.mat2euler(T.quat2mat(env._eef_xquat))
            current_pos = np.append(
                env._eef_xpos, np.array([MyEuler1[2], MyEuler1[1], MyEuler1[0], 0])
            )
            kp = np.array([20, 20, 20, 0, 0, 0])
            kp = np.append(kp, np.ones(action_dim - 6))
            action = vel_control(current_pos, goal_pos, k=kp)
            # action = np.array([0, 0, 0, 0, 0, 0, 0])

            obs, reward, done, info = env.step(action)  # take action in the environment
            observation.append(obs)
            Reward.append(reward)
            env.render()  # render on display
        if i % 2 == 0:  # 隔一步显示一次力信息
            ee_force = env.sim.data.sensordata[0:3]
            # print('ee_force=', ee_force)
            # print("ori_current= ", T.quat2axisangle(env._eef_xquat))
            Force = np.append(Force, ee_force)
        last_pos = current_pos
    print("-----------------end the trajectory--------------------")
    MyEuler = T.quat2axisangle(env._eef_xquat)
    print(
        "ee_pos_x=",
        np.append(env._eef_xpos, np.array([MyEuler[0], MyEuler[1], MyEuler[2]])),
    )
    print("goal_pos=", x_s)

    Force = Force.reshape(-1, 3)

    return observation, Reward, Force


##----------------------------------------------------------------------------------------------------------------------
# create environment instance
robots = "IIWA"
env = robosuite_task_zoo.environments.manipulation.HammerPlaceEnv(
    robots,
    has_renderer=True,
    # gripper_types="RobotiqThreeFingerGripper",
    has_offscreen_renderer=True,
    use_camera_obs=True,
    render_camera="frontview",
    control_freq=20,
    # 操作空间位置控制（选择该控制器action的动作空间可能也发生相应变化）
    controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
)

env.reset()
# env.viewer.set_camera(camera_id=0)
print("-------------------start the tarjectory-------------------------")
print(
    env.cabinet_handle_qpos,
    "############################################################################",
)
# set the action dim
action_dim = env.action_dim  # in robot_env.py
neutral = np.zeros(action_dim)
# print('handle_pos_init=', env._handle_xpos)
# print('slide_handle_pos_init=', env._slide_handle_xpos)

# 第一段轨迹
MyEuler = R.from_quat(env._eef_xquat).as_euler("xyz")
d_0 = env._eef_xpos
print("d0=", d_0)
goal_pos = env._slide_handle_xpos
t_f = 10.0
# 检验t
t1 = env.sim.data.time
obs1, _, force1 = trace_trajectory_Astar(d_0, goal_pos, tf=t_f, freq=20)
print("delta_t=", env.sim.data.time - t1)
print("handle_pos_fin=", env._handle_xpos)
print("slide_handle_pos_fin=", env._slide_handle_xpos)

# 第二段轨迹（开关夹爪）
action = neutral.copy()
action[-1] = 1
for i in range(20):
    obs_2, reward, done, info = env.step(action)
    env.render()  # render on display
print("handle_pos_fin=", env._handle_xpos)
print("slide_handle_pos_fin=", env._slide_handle_xpos)

# 第三段轨迹
MyEuler = R.from_quat(env._eef_xquat).as_euler("zyx")
d_0 = env._eef_xpos
print("d0=", d_0)
goal_pos = env._slide_handle_xpos - np.array([0, 0.2, 0.0])
goal_force = np.array([0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0])
# 检验t
t1 = env.sim.data.time
obs_3, _, force2 = trace_trajectory_Astar(
    d_0, goal_pos, tf=t_f, freq=20, gri_open=False
)
print("delta_t=", env.sim.data.time - t1)
print("handle_pos_fin=", env._handle_xpos)
print("slide_handle_pos_fin=", env._slide_handle_xpos)
print(
    env.cabinet_handle_qpos,
    "############################################################################",
)
env.close()

obs = obs1 + obs_3
pos_x, pos_y, pos_z = [], [], []
for j in range(len(obs)):
    pos_x.append(obs[j]["robot0_eef_pos"][0])
    pos_y.append(obs[j]["robot0_eef_pos"][1])
    pos_z.append(obs[j]["robot0_eef_pos"][2])

ax1 = plt.axes(projection="3d")
ax1.plot3D(pos_x, pos_y, pos_z, "blue")
plt.figure()

force = force1
force = np.append(force, force2)
force = force.reshape(-1, 3)
force_x, force_y, force_z = [], [], []
freq_force = 10
for i in range(len(force)):
    force_x.append(force[i][0])
    force_y.append(force[i][1])
    force_z.append(force[i][2])
plt.subplot(3, 1, 1)
plt.plot(np.arange(len(force_x)) / freq_force, force_x)
plt.ylabel("Fx/N")
plt.subplot(3, 1, 2)
plt.plot(np.arange(len(force_y)) / freq_force, force_y)
plt.ylabel("Fy/N")
plt.subplot(3, 1, 3)
plt.plot(np.arange(len(force_z)) / freq_force, force_z)
plt.ylabel("Fz/N")
plt.xlabel("time/s")
plt.show()

np.savetxt(r"force_x.txt", force_x)
np.savetxt(r"force_y.txt", force_y)
np.savetxt(r"force_z.txt", force_z)
