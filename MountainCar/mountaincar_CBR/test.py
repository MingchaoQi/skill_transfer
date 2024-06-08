import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
# import mujoco
"""
基于CBR+PGPE实现2D小车到3D小车技能迁移
"""

class CBR_TL:
    def __init__(self):
        self.flag = 1
        self.flag_v = 0

    def action_policy(self, obs):###原环境策略，基于规则的PGPE方法训练的结果
        # model = stable_baselines3.DQN.load('mountaincar.pkl')
        # action, _ = model.predict(observation=obs)

        position, velocity = obs
        action = 2
        if velocity >= 0:  # velocity>=0 表示车辆正在向右移动
            if (position > -0.5) and (abs(velocity) < 0.018433):  # 当到达右侧一定位置无法向右时, 则向左
                action = 0
            else:
                action = 2
        elif velocity < 0:  # velocity<0 表示车辆正在向左移动
            if (position < -0.63454) and (abs(velocity) < 0.018433):  # 当到达左侧一定位置无法向左时, 则向右
                action = 2
            else:
                action = 0

        return action


    def case_generation(self, env):
        N = 1
        case_base = np.zeros((1, 3))
        for i in range(N):
            # start_point = -0.4 - (-0.4 + 0.6)/N * i
            start_point = -0.51
            obs = env.reset_manual(start_point)
            # obs = env.reset()
            while True:
                env.render()
                action = CBR_TL.action_policy(self, obs)
                # action = 0
                obs_append = np.append(obs, action)
                case_base = np.append(case_base, obs_append) #扩充一组状态-动作对
                next_obs, reward, done, info = env.step(action)
                obs = next_obs
                if done:
                    break

        ##精简案例库
        case_base = case_base.reshape(-1, 3)
        case_base = case_base[1:]
        delta = np.array([1.8, 0.14]) / 100 #过于相似的case被合并,空间越大最终的case数量越少
        case_base1 = case_base
        i, j = 0, 0
        while i < len(case_base):
            j = i+1
            while j < len(case_base):
                if abs(case_base[i][0] - case_base[j][0]) <= delta[0] and abs(case_base[i][1] - case_base[j][1]) <= delta[1]:
                    if case_base[i][2] == case_base[j][2]:
                        case_base = np.delete(case_base, j, axis=0)
                    else:
                        j += 1
                else:
                     j += 1
            i += 1

        return case_base


    def case_selection(self, case_base, obs_new):
        m = 1
        position_x, velocity_x, position_y, velocity_y = obs_new

        ###构建相似度函数
        Sim_func = np.zeros(case_base.shape[0])
        for i in range(case_base.shape[0]):
            case_p = case_base[i,0]
            case_v = case_base[i,1]
            Sim_func[i] = min(abs(case_p - position_x)+abs(case_v - velocity_x),
                              abs(case_p - position_y)+abs(case_v - velocity_y))

        ###选取最相近的m个case
        case_select = np.zeros((m, case_base.shape[1]))
        for j in range(m):
            case_index = np.argmin(Sim_func)###选取最相似的
            case_select[j] = case_base[case_index]
            case_base = np.delete(case_base, case_index, axis=0)
            Sim_func = np.delete(Sim_func, case_index, axis=0)

        return case_select


    def action_mapping(self, case_select):
        action = np.mean(case_select[:,2])
        map_action = 2
        ###使用自动匹配好的结果，未来可以使用神经网络匹配
        if action == 0:
            map_action = 0 ###映射动作的目的是为最终的动作选择提供参考，由于转移环境的动作维数不对等，往往无法直接选择动作
        elif action == 1:
            map_action = 2
        else:
            map_action = 1

        return map_action


    def action_adaption(self, map_action, obs_new, theta): ###形成最终的动作，需要进一步研究如何把新生成的规则填充进去
        position_x, velocity_x, position_y, velocity_y = obs_new
        adapt_action = map_action
        if position_x >= theta and adapt_action < 2:
            adapt_action = adapt_action + 3

        return adapt_action


    def play_ones(self,env, theta, case_base, index = False): #只有一条规则时的动作策略
        episode_reward = 0  # 记录回合总奖励，初始值为0
        if index:
            obs = env.reset_manual(0.5)
        else:
            obs = env.reset()  # 重置游戏环境，开始新回合
        obs_record = obs

        while True:  # 不断循环，直到回合结束
            env.render()  # 显示图形界面，可以用env.close()关闭
            case_select = self.case_selection(case_base, obs)
            action_map = self.action_mapping(case_select)
            action_adapt = self.action_adaption(action_map, obs, theta)  # 产生新动作
            next_observation, reward, done, info = env.step(action_adapt)  # 执行动作
            episode_reward += reward  # 搜集回合奖励
            if done:  # 判断是否训练智能体
                break
            obs = next_observation

        return episode_reward, obs_record  # 返回回合总奖励

    def play_twice(self, env, theta, beta, case_base): #采用改进后的方法进行训练
        episode_reward = 0  # 记录回合总奖励，初始值为0
        obs = env.reset()  # 重置游戏环境，开始新回合
        obs_record = obs
        action_adapt = 2
        self.flag = 1
        if beta < -0.5:
            self.flag_v = 0
        else:
            self.flag_v = 1

        while True:  # 不断循环，直到回合结束
            env.render()  # 显示图形界面，可以用env.close()关闭
            if self.flag == 1: # 生成的第二条规则
                position_x, velocity_x, position_y, velocity_y = obs
                if beta < -0.5:
                    if abs(position_y) < abs(beta):
                        action_adapt = 3 + self.flag_v #向着目标点beta运动
                    else:
                        action_adapt = 2
                    if abs(position_y) > abs(beta) and abs(velocity_y) < 0.005:
                        self.flag = 0
                else:
                    if abs(position_y) > abs(beta):
                        action_adapt = 3 + self.flag_v #向着目标点beta运动
                    else:
                        action_adapt = 2
                    if abs(position_y) < abs(beta) and abs(velocity_y) < 0.005:
                        self.flag = 0

            if self.flag == 0: #向着目标点运动结束，开始执行原规则
                case_select = self.case_selection(case_base, obs)
                action_map = self.action_mapping(case_select)
                action_adapt = self.action_adaption(action_map, obs, theta)  # 产生新动作
            next_observation, reward, done, info = env.step(action_adapt)  # 执行动作
            episode_reward += reward  # 搜集回合奖励
            if done:  # 判断是否训练智能体
                break
            obs = next_observation

        return episode_reward, obs_record  # 返回回合总奖励


    def policy(self, env, case_base, mu, sigmma, num_range, num_eposide, alpha):
        theta = np.random.normal(mu, sigmma, num_range)  # producing Gauss
        Reward = np.zeros(num_range)
        all_mu = 0
        all_sigmma = 0
        print(Reward)
        print("theta= ", theta)
        for i in range(num_range):
            eposide_reward = 0
            for j in range(num_eposide):
                reward, _ = self.play_ones(env, theta[i], case_base)
                eposide_reward += reward
            eposide_reward /= num_eposide
            Reward[i] = eposide_reward
            print("eposide_reward [%f]= %f" % (theta[i], Reward[i]))
        Reward = Reward - np.mean(Reward)
        if Reward.all() == np.zeros(num_range).all():
            Reward = Reward
        else:
            Reward = Reward / np.std(Reward)
        print("count_reward=", Reward)
        for i in range(len(Reward)):
            all_mu += (theta[i] - mu) / (sigmma * sigmma) * Reward[i]
            all_sigmma += ((theta[i] - mu) * (theta[i] - mu) - (sigmma * sigmma)) / (sigmma * sigmma * sigmma) * Reward[
                i]
        u = mu + (alpha * sigmma * sigmma * 1.25) * all_mu / num_range
        u = np.clip(u, -0.7, 0.7)
        g = sigmma - sigmma/(num_range + 2)
        if Reward.all() == np.zeros(num_range).all():
            u = u
            g = g + 0.01
            if g >= 0.1:
                u = u + 0.01
                g = 0.05

        return u, g


    def policy_twice(self, env, case_base, mu, sigmma, mu_beta, sigmma_beta, num_range, num_eposide, alpha):
        theta = np.random.normal(mu, sigmma, num_range)  # producing Gauss
        beta = np.random.normal(mu_beta, sigmma_beta, num_range)  # producing Gauss
        Reward = np.zeros(num_range)
        all_mu = 0
        all_sigmma = 0
        all_mu1 = 0
        all_sigmma1 = 0
        print(Reward)
        print("theta= ", theta)
        print("beta= ", beta)
        for i in range(num_range):
            eposide_reward = 0
            for j in range(num_eposide):
                reward, _ = self.play_twice(env, theta[i], beta[i], case_base)
                eposide_reward += reward
            eposide_reward /= num_eposide
            Reward[i] = eposide_reward
            print("eposide_reward [%f][%f]= %f" % (theta[i], beta[i], Reward[i]))
        Reward = Reward - np.mean(Reward)
        if Reward.all() == np.zeros(num_range).all():
            Reward = Reward
        else:
            Reward = Reward / np.std(Reward)
        print("count_reward=", Reward)
        for i in range(len(Reward)):
            all_mu += (theta[i] - mu) / (sigmma * sigmma) * Reward[i]
            all_sigmma += ((theta[i] - mu) * (theta[i] - mu) - (sigmma * sigmma)) / (sigmma * sigmma * sigmma) * Reward[
                i]
            all_mu1 += (beta[i] - mu_beta) / (sigmma_beta * sigmma_beta) * Reward[i]
            all_sigmma1 += ((beta[i] - mu_beta) * (beta[i] - mu_beta) - (sigmma_beta * sigmma_beta)) / (
                    sigmma_beta * sigmma_beta * sigmma_beta) * Reward[i]
        u = mu + (alpha * sigmma * sigmma * 1.25) * all_mu / num_range
        u = np.clip(u, -0.7, 0.7)
        g = sigmma - sigmma/(num_range + 2)
        a = mu_beta + (alpha * sigmma_beta * sigmma_beta * 2.5) * all_mu1 / num_range
        a = np.clip(a, -1.2, 0.7)
        b = sigmma_beta - sigmma_beta/(num_range + 2)
        if Reward.all() == np.zeros(num_range).all():
            u = u
            g = g + 0.01
            b = b+0.01
            if g >= 0.1:
                u = u + 0.01
                g = 0.05
                b = 0.05

        return u, g, a, b


    def case_update(self,env, theta, beta, case_base):
        N = 10
        case_base_update = np.zeros((1, 5))
        for i in range(N):
            # obs = env.reset_manual(start_point)
            obs = env.reset()
            while True:
                # env.render()
                action = self.play_twice(env, theta, beta, case_base)
                # action = 0
                obs_append = np.append(obs, action)
                case_base_update = np.append(case_base_update, obs_append) #扩充一组状态-动作对
                next_obs, reward, done, info = env.step(action)
                obs = next_obs
                if done:
                    break

        return case_base_update


if __name__ == '__main__':
    env = gym.make("MountainCar-v0",render_mode="rgb_array")
    # log_dir = "monitor"
    # os.makedirs(log_dir, exist_ok=True)
    # # env = DummyVecEnv([lambda :env]) #使用Q-LEARNING方法训练智能体
    # env = Monitor(env, log_dir)
    # reward = env.get_episode_rewards()
    # print(reward)
    # model =stable_baselines3.DQN(policy='MlpPolicy',
    #                              env=env,
    #                              learning_rate=4e-3,
    #                              batch_size=128,
    #                              buffer_size=50000,
    #                              learning_starts=5000,
    #                              gamma=0.99,
    #                              target_update_interval=600,
    #                              train_freq=16,
    #                              gradient_steps=-1,
    #                              exploration_fraction=0.2,
    #                              exploration_final_eps=0.1,
    #                              policy_kwargs={'net_arch':[256, 256, 128]},
    #                              verbose=1,
    #                              tensorboard_log="LunarLander-v2")
    # model.learn(total_timesteps=1e6)
    # results_plotter.plot_results([log_dir], 1e6, results_plotter.X_EPISODES, "DQN mountaincar3D")
    # model = stable_baselines3.DQN.load('mountaincar3D.pkl')
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
    # print('mean_reward=', mean_reward)
    # print("std_reward=", std_reward)
    # env.close()
    # model.save('mountaincar3D.pkl')


    agent = CBR_TL()
    try:
        case = np.load('casebase_PGPE.npy')
    except:
        case = agent.case_generation(env)
        np.save('casebase_PGPE.npy', arr=case)
    env.close()

    env = gym.make('MountainCar3D-v0',render_mode="rgb_array")

    N1 = 10
    N2 = 20
    theta = np.zeros(N1)
    sigma_theta = np.zeros(N1)
    R = np.zeros(0)

    theta[0] = 0.4
    sigma_theta[0] = 0.1
    num_range = 10
    num_episode = 1
    alpha = 0.95

    for t in range(len(theta)-1):
        r, observation_init = agent.play_ones(env, theta[t], case)
        R = np.append(R, r)
        u,g = agent.policy(env, case, theta[t], sigma_theta[t], num_range, num_episode, alpha)
        theta[t + 1] = u
        sigma_theta[t + 1] = g
        print("mu = ", theta[t + 1])
        print("sigma = ", sigma_theta[t + 1])
        print("reward_toward=", R[t])

    #微调智能体初始位置，收集奖励情况
    R1 = np.zeros(0)
    agent.flag = 1#表示微调的是y方向的初值
    # agent.flag = 0
    state_init = np.zeros((N1, 4))  # 收集初始状态点
    for t in range(len(theta)):
        r, observation_init = agent.play_ones(env, theta[-1], case, index=True)
        R1 = np.append(R1, r)
        print("reward_toward=", R1[t])
        state_init[t] = observation_init
    r_max_index = np.argmax(R1)

    if agent.flag == 0:
        x_target = state_init[r_max_index, 2]
        # y_target = -0.54857755
        x1 = (-0.5 + x_target) / 2
        if x1 < -0.5:
            agent.flag_v = 0 #表示要向负方向运动
        else:
            agent.flag_v = 1 #表示要向正方向运动

        theta_1 = np.zeros(N2)
        sigma_theta_1 = np.zeros(N2)
        theta_2 = np.zeros(N2)
        sigma_theta_2 = np.zeros(N2)
        theta_1[0] = theta[-1] #原参数从优化后的结果开始优化
        sigma_theta_1[0] = 0.01 #探索方差的大小可以再斟酌
        theta_2[0] = x1
        sigma_theta_2[0] = 0.01
    if agent.flag == 1:
        y_target = state_init[r_max_index, 2]
        # y_target = -0.3
        y1 = (-0.5 + y_target) / 2
        if y1 < -0.5:
            agent.flag_v = 0 #表示要向负方向运动
        else:
            agent.flag_v = 1 #表示要向正方向运动

        theta_1 = np.zeros(N2)
        sigma_theta_1 = np.zeros(N2)
        theta_2 = np.zeros(N2)
        sigma_theta_2 = np.zeros(N2)
        theta_1[0] = theta[-1] #原参数从优化后的结果开始优化
        sigma_theta_1[0] = 0.1 #探索方差的大小可以再斟酌
        theta_2[0] = y1
        sigma_theta_2[0] = abs(-0.5 - y_target)/4
    R2 = np.zeros(0)
    num_range = 20

    # theta_1[0] = 0.4
    for t in range(len(theta_1) - 1):
        r, _ = agent.play_twice(env, theta_1[t], theta_2[t], case)
        R2 = np.append(R2, r)
        u, g, a, b = agent.policy_twice(env, case, theta_1[t], sigma_theta_1[t], theta_2[t], sigma_theta_2[t],
                                        num_range, num_episode, alpha)
        theta_1[t + 1] = u
        sigma_theta_1[t + 1] = g
        theta_2[t + 1] = a
        sigma_theta_2[t + 1] = b
        print("mu = ", theta_1[t + 1])
        print("sigma = ", sigma_theta_1[t + 1])
        print("mu_beta = ", theta_2[t + 1])
        print("sigma_beta = ", sigma_theta_2[t + 1])
        print("reward_toward=", R2[t])

