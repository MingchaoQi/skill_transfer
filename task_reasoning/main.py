import matplotlib.pyplot as plt
import numpy as np
import robosuite as suite
from robosuite.utils.control_utils import *
from scipy.spatial.transform import Rotation as R

import robosuite_task_zoo
import tracjroy as trac

"""
在控制方面，提供的控制其实是速度控制，其能较好地避免震荡，需想办法把速度控制引入轨迹控制中（√）
姿态控制不稳定，症结好像是四元数转换成欧拉角时有异议，控制绕x方向偏转可能会导致y方向欧拉角变化，还有就是欧拉角每次转换结果页不一致.(√)
经过总结，这是因为末端gripper本身就有初始姿态，故绕世界坐标系中的轴进行旋转，会造成转换后的欧拉角与世界坐标系中的旋转角度不一致
"""

def PD_control(last_pos, cur_pos, goal_pos, goal_vel, kp, kd): #PD控制生成控制指令
    cur_vel = (cur_pos - last_pos) * 20
    a = kp * (goal_pos - cur_pos) + kd * (goal_vel - cur_vel)
    return a

def vel_control(cur_pos, goal_pos, k):
    a = k*(goal_pos - cur_pos)

    return a

def Force_control(xe, dxe, Fe, Mp, Bp, Cp, freq):
    ddxe = (Fe - Bp * dxe - Cp * xe) / Mp
    dxe_next = dxe + ddxe / freq
    xe_next = xe + dxe_next / freq

    return xe_next

def trace_trajectory(d0, goal_pos, tf):
    plan = trac.trac_plan(d0, goal_pos, tf, freq=20)
    plan.virtual_trac()  # 路径规划，生成序列点
    last_pos = d0

    observation = []
    Reward = []
    Force = np.zeros(0)

    for i in range(int(tf * 20 * 1.2)):
        MyEuler = R.from_quat(env._eef_xquat).as_euler('xyz')
        current_pos = np.append(env._eef_xpos, np.array([MyEuler[0], MyEuler[1], MyEuler[2], 0]))
        if i < len(plan.d):
            kp = np.append(np.ones(3) * 20, np.ones(3) * 0.4)
            kp = np.append(kp, np.ones(action_dim - 6))
            kd = 0 * np.sqrt(kp)
            action = PD_control(last_pos, current_pos, plan.d[i], plan.d_dot[i], kp, kd)
        else:
            kp = np.append(np.ones(3) * 2, np.ones(3) * 0.4)
            kp = np.append(kp, np.ones(action_dim - 6))
            kd = 2 * np.sqrt(kp)
            goal_vel = np.zeros_like(goal_pos)
            action = PD_control(last_pos, current_pos, goal_pos, goal_vel, kp, kd)
        if i % 2 == 0:
            ee_force = env.sim.data.sensordata[0:3]
            # print('ee_pos_x=', env._eef_xpos)  # in single_arm_env.py
            # print('ee_force=', ee_force)
            # Force.append(ee_force)
            Force = np.append(Force, ee_force)
        last_pos = current_pos
        obs, reward, done, info = env.step(action)  # take action in the environment
        observation.append(obs)
        Reward.append(reward)
        env.render()  # render on display
        if i == len(plan.d):
            print('-----------------end the trajectory--------------------')
            print('ee_pos_x=', np.append(env._eef_xpos, R.from_quat(env._eef_xquat).as_euler('xyz')))

    Force = Force.reshape(-1,3)

    return observation, Reward, Force

def trace_trajectory_soft(d0, goal_pos, tf, freq): #达不到柔顺控制的要求
    plan = trac.trac_plan(d0, goal_pos, tf, freq=freq)
    plan.virtual_trac()  # 路径规划，生成序列点
    t0 = env.sim.data.time
    action = np.zeros(action_dim)
    goal_force = np.array([0, 0, 0, 0, 0, 0, 0])
    last_pos = d0

    observation = []
    Reward = []
    Force = np.zeros(0)

    cp = np.array([5000, 5000, 5000, 1000, 1000, 1000, 1000000])
    mp = np.array([1, 1, 1, 1, 1, 1, 1]) * 80
    bp = 2 * 0.707 * np.sqrt(cp * mp)

    for i in range(int(tf * freq * 1.2)):
        if i < len(plan.d):
            MyEuler1 = R.from_quat(env._eef_xquat).as_euler('zyx')
            current_pos = np.append(env._eef_xpos, np.array([MyEuler1[2], MyEuler1[1], MyEuler1[0], 0]))
            current_force = np.append(env.sim.data.sensordata[0:3], np.array([0, 0, 0, 0]))
            xe = current_pos - plan.d[i] + plan.d_dot[i] /freq
            dxe = (current_pos - last_pos) * freq - plan.d_dot[i] + plan.d_ddot[i] / freq
            Fe = current_force - goal_force
            xd = Force_control(xe, dxe, Fe, mp, bp, cp, freq)
            x_s = plan.d[i] + xd
            while (env.sim.data.time-t0) < plan.t[i]:
                MyEuler1 = R.from_quat(env._eef_xquat).as_euler('zyx')
                # MyEuler1 = T.mat2euler(T.quat2mat(env._eef_xquat))
                current_pos = np.append(env._eef_xpos, np.array([MyEuler1[2], MyEuler1[1], MyEuler1[0], 0]))
                kp = np.array([20, 20, 20, 0, 0, 0])
                kp = np.append(kp, np.ones(action_dim - 6))
                # action = vel_control(current_pos, x_s, k=kp)
                kd = 0.7 * np.sqrt(kp)
                action = PD_control(last_pos, current_pos, x_s, plan.d_dot[i], kp, kd)

                obs, reward, done, info = env.step(action)  # take action in the environment
                observation.append(obs)
                Reward.append(reward)
                env.render()  # render on display
        else:
            goal_pos = x_s
            MyEuler1 = R.from_quat(env._eef_xquat).as_euler('zyx')
            # MyEuler1 = T.mat2euler(T.quat2mat(env._eef_xquat))
            current_pos = np.append(env._eef_xpos, np.array([MyEuler1[2], MyEuler1[1], MyEuler1[0], 0]))
            kp = np.array([100, 100, 100, 0, -10, -10])
            kp = np.append(kp, np.ones(action_dim - 6))
            action = vel_control(current_pos, goal_pos, k=kp)
            # action = np.array([0, 0, 0, 0, 0, 0, 0])

            obs, reward, done, info = env.step(action)  # take action in the environment
            observation.append(obs)
            Reward.append(reward)
            env.render()  # render on display
        if i % 2 == 0:
            ee_force = env.sim.data.sensordata[0:3]
            # print('ee_force=', ee_force)
            Force = np.append(Force, ee_force)
        last_pos = current_pos
    print('-----------------end the trajectory--------------------')
    MyEuler = R.from_quat(env._eef_xquat).as_euler('zyx')
    print('ee_pos_x=', np.append(env._eef_xpos, np.array([MyEuler[2], MyEuler[1], MyEuler[0]])))
    print('true_goal_pos=', x_s)

    Force = Force.reshape(-1,3)

    return observation, Reward, Force

##----------------------------------------------------------------------------------------------------------------------
# create environment instance
robots = "IIWA"
env = robosuite_task_zoo.environments.manipulation.HammerPlaceEnv(
    robots,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    render_camera='frontview',
    control_freq=20,
    controller_configs= suite.load_controller_config(default_controller="OSC_POSE") #操作空间位置控制
)
# reset the environment
env.reset()
# env.viewer.set_camera(camera_id=0)

#set the action dim
action_dim = env.action_dim # in robot_env.py
neutral = np.zeros(action_dim)

# print('handle_pos_init=', env._handle_xpos)
# print('slide_handle_pos_init=', env._slide_handle_xpos)
MyEuler = R.from_quat(env._eef_xquat).as_euler('zyx')
d_0 = np.append(env._eef_xpos, np.array([MyEuler[2], MyEuler[1], MyEuler[0]]))
d_0 = np.append(d_0, np.zeros(1))
print("d0=", d_0)
test_xpos = env._slide_handle_xpos
goal_pos = np.append(env._slide_handle_xpos + np.array([0, 0, 0.2]), np.array([MyEuler[2], MyEuler[1], MyEuler[0], 0]))
t_f = 10.0
##检验t
t1 = env.sim.data.time
_, _, force1 = trace_trajectory_soft(d_0, goal_pos, t_f, freq=20)
print('delta_t=', env.sim.data.time - t1)
print('handle_pos_fin=', env._handle_xpos)
print('slide_handle_pos_fin=', env._slide_handle_xpos)

MyEuler = R.from_quat(env._eef_xquat).as_euler('zyx')
d_0 = np.append(env._eef_xpos, np.array([MyEuler[2], MyEuler[1], MyEuler[0]]))
d_0 = np.append(d_0, np.zeros(1))
print("d0=", d_0)
goal_pos = np.append(env._eef_xpos + np.array([0.0, 0.0, -0.2]), np.array([MyEuler[2], MyEuler[1], MyEuler[0], 0]))
t_f = 10.0
##检验t
t1 = env.sim.data.time
_, _, force2  = trace_trajectory_soft(d_0, goal_pos, t_f, freq=20)
print('delta_t=', env.sim.data.time - t1)
print('handle_pos_fin=', env._handle_xpos)
print('slide_handle_pos_fin=', env._slide_handle_xpos)

action = neutral.copy()
action[-1] = 1
for i in range(20):
    obs, reward, done, info = env.step(action)
    env.render()  # render on display
print('handle_pos_fin=', env._handle_xpos)
print('slide_handle_pos_fin=', env._slide_handle_xpos)

MyEuler = R.from_quat(env._eef_xquat).as_euler('zyx')
d_0 = np.append(env._eef_xpos, np.array([MyEuler[2], MyEuler[1], MyEuler[0]]))
d_0 = np.append(d_0, np.zeros(1))
print("d0=", d_0)
goal_pos = np.append(env._slide_handle_xpos + np.array([0, -0.2, 0.0]), np.array([MyEuler[2], MyEuler[1], MyEuler[0], 0]))
goal_force = np.array([0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0])
##检验t
t1 = env.sim.data.time
_, _, force3 = trace_trajectory_soft(d_0, goal_pos, t_f, freq=20)
print('delta_t=', env.sim.data.time - t1)
print('handle_pos_fin=', env._handle_xpos)
print('slide_handle_pos_fin=', env._slide_handle_xpos)

env.close()

force = force1
force = np.append(force, force2)
force = np.append(force, force3)
force = force.reshape(-1,3)
force_x, force_y, force_z = [], [], []
for i in range(len(force)):
    force_x.append(force[i][0])
    force_y.append(force[i][1])
    force_z.append(force[i][2])
plt.subplot(3, 1, 1)
plt.plot(list(range(len(force_x))), force_x)
plt.subplot(3, 1, 2)
plt.plot(list(range(len(force_y))), force_y)
plt.subplot(3, 1, 3)
plt.plot(list(range(len(force_z))), force_z)
plt.show()


##----------------------------------------------------------------------------------------------------------------------
#第一步：设置世界，采用自带的默认世界MujocoWorldBase
# import numpy as np
# from robosuite.models import MujocoWorldBase
# world = MujocoWorldBase()
#
# #第二步：创建自己的机器人
# from robosuite.models.robots import UR5e
# # mujoco_robot1 = Panda()
# mujoco_robot1 = UR5e()
#
# #我们可以通过创建一个抓手实例并在机器人上调用 add_gripper 方法来为机器人添加一个抓手。
# from robosuite.models.grippers import gripper_factory
# gripper = gripper_factory('PandaGripper')
# mujoco_robot1.add_gripper(gripper)
#
# #要将机器人添加到世界中，我们将机器人放置到所需位置并将其合并到世界中
# mujoco_robot1.set_base_xpos([0.5,0,0.8])#刚好能放置在桌子上，桌子高度为0.8
# world.merge(mujoco_robot1)
# #第三步：创建桌子。我们可以初始化创建桌子和地平面,TableArena代表的是一个拥有桌子的整体环境
# from robosuite.models.arenas import TableArena
#
# mujoco_arena = TableArena()
# mujoco_arena.set_origin([0.8, 0, 0])#这里的set_origin是对所有对象应用恒定偏移。在X轴偏移0.8
# world.merge(mujoco_arena)
#
# #第四步：添加对象。创建一个球并将其添加到世界中。
#
# from robosuite.models.objects import BallObject
# sphere1 = BallObject(
#     name="sphere1",
#     size=[0.04],
#     rgba=[0, 0.5, 0.5, 1]).get_obj()
# sphere1.set('pos', '1.3 0 1.0')
# world.worldbody.append(sphere1)
#
# from robosuite.models.objects import BoxObject
# sphere2 = BoxObject(
#     name = "sphere2",
#     size = [0.07, 0.07, 0.07],
#     rgba=[0, 0.5, 0.5, 1]).get_obj()
# sphere2.set('pos', '1.4 0 1.0')
# world.worldbody.append(sphere2)
#
# from robosuite.models.objects import CapsuleObject
# sphere3 = CapsuleObject(
#     name = "sphere3",
#     size = [0.07,  0.07],
#     rgba=[0.5, 0.5, 0.5, 1]).get_obj()#颜色(三个0.5是黑色，A值代表透明度)
# sphere3.set('pos', '1.0 0 1.0')
# world.worldbody.append(sphere3)
#
# from robosuite.utils.mjcf_utils import new_joint
# from robosuite.utils import OpenCVRenderer
# from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim
#
# #第5步：运行模拟。一旦我们创建了对象，我们可以通过运行mujoco_py获得一个模型
# model = world.get_model(mode="mujoco")
# sim = MjSim(model)
# viewer = OpenCVRenderer(sim)
# render_context = MjRenderContextOffscreen(sim, device_id=-1)
# sim.add_render_context(render_context)
# while True:
#     # Step through sim
#     sim.step()
#     viewer.render()