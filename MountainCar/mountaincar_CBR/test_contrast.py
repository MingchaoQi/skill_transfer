import gymnasium as gym
import time
import numpy as np
import math
import matplotlib.pyplot as plt
'''
    基于PGPE强化学习实现3D小车自适应翻越小沟
'''
class BespokeAgent3D:
    def __init__(self, env):
        self.env = env
        self.flag = 0
        self.flag_v = 0

    def decide(self, observation, theta, beta): #动作策略
        action = 2
        if self.flag == 1:  # 生成的第二条规则
            position_x, velocity_x, position_y, velocity_y = observation
            if beta < -0.5:
                if abs(position_y) < abs(beta):
                    action = 3 + self.flag_v  # 向着目标点beta运动
                else:
                    action = 2
                if abs(position_y) > abs(beta) and abs(velocity_y) < 0.005:
                    self.flag = 0
            else:
                if abs(position_y) > abs(beta):
                    action = 3 + self.flag_v  # 向着目标点beta运动
                else:
                    action = 2
                if abs(position_y) < abs(beta) and abs(velocity_y) < 0.005:
                    self.flag = 0

        if self.flag == 0: #执行原策略
            position_x, velocity_x, position_y, velocity_y = observation
            if position_x >= theta:
                if velocity_y >= 0:  # velocity>=0 表示车辆正在向右移动
                    if (position_y > -0.5) and (abs(velocity_y) < 0.018433):  # 当到达右侧一定位置无法向上时, 则向下
                        action = 3
                    else:
                        action = 4
                elif velocity_y < 0:  # velocity<0 表示车辆正在向左移动
                    if (position_y < -0.63454) and (abs(velocity_y) < 0.018433):  # 当到达左侧一定位置无法向下时, 则向上
                        action = 4
                    else:
                        action = 3
            else:
                if velocity_x >= 0:  # velocity>=0 表示车辆正在向右移动
                    if (position_x > -0.5) and (abs(velocity_x) < 0.018433):  # 当到达右侧一定位置无法向右时, 则向左
                        action = 0
                    else:
                        action = 1
                elif velocity_x < 0:  # velocity<0 表示车辆正在向左移动
                    if (position_x < -0.63454) and (abs(velocity_x) < 0.018433):  # 当到达左侧一定位置无法向左时, 则向右
                        action = 1
                    else:
                        action = 0

        return action  # 返回动作

    def learn(self, *args):  # 学习
        pass

    def play_ones(self, theta, beta, render=False, train=False):
        episode_reward = 0  # 记录回合总奖励，初始值为0
        observation = self.env.reset()[0]  # 重置游戏环境，开始新回合
        self.flag = 1
        if beta < -0.5:
            self.flag_v = 0
        else:
            self.flag_v = 1

        while True:  # 不断循环，直到回合结束
            if render:  # 判断是否显示
                self.env.render()  # 显示图形界面，可以用env.close()关闭
            action = self.decide(observation,theta, beta)
            next_observation, reward, done, _, _ = self.env.step(action)  # 执行动作
            episode_reward += reward  # 搜集回合奖励
            if done:# 判断是否训练智能体
                break
            observation = next_observation

        return episode_reward  # 返回回合总奖励

    def policy(self, mu, sigmma, mu_beta, sigmma_beta, num_range, num_eposide, alpha):
        theta = []
        theta = np.random.normal(mu, sigmma, num_range)  # producing Gauss
        beta = []
        beta = np.random.normal(mu_beta, sigmma_beta, num_range)  # producing Gauss
        Reward = np.zeros(num_range, np.float64)
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
                eposide_reward += self.play_ones(theta[i], beta[i], render=True)
            eposide_reward /= j + 1
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
        u = np.clip(u, -0.5, 0.6)
        g = sigmma - sigmma/65
        a = mu_beta + (alpha * sigmma_beta * sigmma_beta * 1.25) * all_mu1 / num_range
        a = np.clip(a, -1.40, 0.0)
        b = sigmma_beta - sigmma_beta/65
        if Reward.all() == np.zeros(num_range).all():
            u = u
            g = g + 0.0001
            b = b+0.001
            if g >= 0.001:
                u = u + 0.001
                g = 0.0001
                b = 0.001

        return u, g, a, b

if __name__ == '__main__':
    # print("mu_beta[0] = -0.7,free")
    env =gym.make('MountainCar3D-v0',render_mode="rgb_array")
    # env.seed(0)  # 设置随机数种子，只是为了让结果可以精确复现，一般情况下可以删除
    agent = BespokeAgent3D(env)

    alpha = 0.95 #studying rate
    N = 20 # parameters number
    M = 1 # eposides number
    mu = list(range(61))
    sigmma = list(range(61))
    mu_beta = list(range(61))
    sigmma_beta = list(range(61))
    R = list()

    # mu[0] = 0.0633
    mu[0] = -0.1
    sigmma[0] = 0.1
    mu_beta[0] = -0.35
    sigmma_beta[0] = 0.05

    for t in range(60): #range number
        u, g, a, b = agent.policy(mu[t], sigmma[t], mu_beta[t], sigmma_beta[t], N, M, alpha)
        mu[t+1] = u
        sigmma[t+1] = g
        mu_beta[t+1] = a
        sigmma_beta[t+1] = b
        print("mu = ", mu[t+1])
        print("sigmma = ", sigmma[t+1])
        print("mu_beta = ", mu_beta[t + 1])
        print("sigmma_beta = ", sigmma_beta[t + 1])
        print("delte_mu=", mu[t + 1] - mu[t])
        print("delte_beta=", mu_beta[t + 1] - mu_beta[t])
        r = agent.play_ones(mu[t], mu_beta[t])
        R.append(r)
        print("reward_toword=",R[t])
        if sigmma[t+1] <0.00005:
            break
    x = list(range(len(R)))
    plt.subplot(3,1,1)
    plt.plot(x,R)
    plt.subplot(3,1,2)
    x1 = list(range(len(mu)))
    plt.plot(x1,mu)
    plt.subplot(3, 1, 3)
    plt.plot(x1, mu_beta)
    plt.show()
    time.sleep(5)  # 停顿5s
    env.close()  # 关闭图形化界面
