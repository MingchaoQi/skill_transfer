import gym
import time
import numpy as np
import math
import matplotlib.pyplot as plt
'''
    基于PGPE强化学习实现2D小车自适应翻越小沟
'''


class BespokeAgent:

    def __init__(self, env):
        self.env = env
        self.flag1 = False
        self.flag2 = False

    def decide(self, observation, theta, beta):
        position, velocity = observation

        # if (position > -0.6) and (abs(velocity) < 0.01):
        #     action = 0
        # elif (position > -0.6) and (abs(velocity) > 0.01):
        #     action = 2
        # elif (position < -0.7) and (abs(velocity) < 0.01):
        #     action = 2
        # elif (position < -0.7) and (abs(velocity) > 0.01):
        #     action = 0
        # else:
        #     action = 1

        action = 2
        if velocity >= 0:  # velocity>=0 表示车辆正在向右移动
            if (position > -0.5) and (abs(velocity)
                                      < theta):  # 当到达右侧一定位置无法向右时, 则向左
                action = 0
            else:
                action = 2
        else:  # velocity<0 表示车辆正在向左移动
            if (position < beta) and (abs(velocity)
                                      < theta):  # 当到达左侧一定位置无法向左时, 则向右
                action = 2
            else:
                action = 0
        return action  # 返回动作

    def learn(self, *args):  # 学习
        pass

    def play_ones(self, theta, beta, render=False, train=False):
        episode_reward = 0  # 记录回合总奖励，初始值为0
        observation = self.env.reset()  # 重置游戏环境，开始新回合
        while True:  # 不断循环，直到回合结束
            # if render:  # 判断是否显示
            #     self.env.render()  # 显示图形界面，可以用env.close()关闭
            action = self.decide(observation, theta, beta)
            next_observation, reward, done, _ = self.env.step(action)  # 执行动作
            episode_reward += reward  # 搜集回合奖励
            if done:  # 判断是否训练智能体
                #policy one
                # if (abs(next_observation[1]) > 0.01 and next_observation[0] >= 0.5):
                #     episode_reward += -20000 * next_observation[1]
                # elif (abs(next_observation[0]) < 0.5):
                #     episode_reward += -20000 * abs(next_observation[1] - 0.5)
                # if (next_observation[0] >=0.5):
                #     episode_reward += math.exp(-65.788*abs(next_observation[1])+4.605)
                # else:
                #     episode_reward = -500
                break
            observation = next_observation
        return episode_reward  # 返回回合总奖励

    def policy(self, mu, sigmma, mu_beta, sigmma_beta, num_range, num_eposide,
               alpha):
        theta = []
        theta = np.random.normal(mu, sigmma, num_range)  # producing Gauss
        beta = []
        beta = np.random.normal(mu_beta, sigmma_beta,
                                num_range)  # producing Gauss
        Reward = np.zeros(num_range, np.float)
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
                eposide_reward += self.play_ones(theta[i],
                                                 beta[i],
                                                 render=True)
            eposide_reward /= j + 1
            Reward[i] = eposide_reward
            print("eposide_reward [%f][%f]= %f" %
                  (theta[i], beta[i], Reward[i]))
        Reward = Reward - np.mean(Reward)
        if Reward.all() == np.zeros(num_range, np.float).all():
            Reward = Reward
        else:
            Reward = Reward / np.std(Reward)
        print("count_reward=", Reward)
        for i in range(len(Reward)):
            all_mu += (theta[i] - mu) / (sigmma * sigmma) * Reward[i]
            all_sigmma += (
                (theta[i] - mu) * (theta[i] - mu) -
                (sigmma * sigmma)) / (sigmma * sigmma * sigmma) * Reward[i]
            all_mu1 += (beta[i] - mu_beta) / (sigmma_beta *
                                              sigmma_beta) * Reward[i]
            all_sigmma1 += (
                (beta[i] - mu_beta) * (beta[i] - mu_beta) -
                (sigmma_beta * sigmma_beta)) / (sigmma_beta * sigmma_beta *
                                                sigmma_beta) * Reward[i]
        u = mu + (alpha * sigmma * sigmma * 1.25) * all_mu / num_range
        u = np.clip(u, -0.07, 0.07)
        g = sigmma - sigmma / 55
        a = mu_beta + (alpha * sigmma_beta * sigmma_beta *
                       1.25) * all_mu1 / num_range
        a = np.clip(a, -1.40, -0.55)
        b = sigmma_beta - sigmma_beta / 55
        if Reward.all() == np.zeros(num_range, np.float).all():
            u = u
            g = g + 0.0001
            if g >= 0.001:
                u = u + 0.001
                g = 0.0003

        return u, g, a, b


if __name__ == '__main__':
    agent = BespokeAgent(gym.make('MountainCar-v0'))

    alpha = 0.95  #studying rate
    N = 20  # parameters number
    M = 1  # eposides number
    mu = list(range(51))
    sigmma = list(range(51))
    mu_beta = list(range(51))
    sigmma_beta = list(range(51))
    R = list()

    mu[0] = 0.01
    sigmma[0] = 0.002
    mu_beta[0] = -0.7
    sigmma_beta[0] = 0.05

    for t in range(50):  #range number
        u, g, a, b = agent.policy(mu[t], sigmma[t], mu_beta[t], sigmma_beta[t],
                                  N, M, alpha)
        mu[t + 1] = u
        sigmma[t + 1] = g
        mu_beta[t + 1] = a
        sigmma_beta[t + 1] = b
        print("mu = ", mu[t + 1])
        print("sigmma = ", sigmma[t + 1])
        print("mu_beta = ", mu_beta[t + 1])
        print("sigmma_beta = ", sigmma_beta[t + 1])
        print("delte_mu=", mu[t + 1] - mu[t])
        print("delte_beta=", mu_beta[t + 1] - mu_beta[t])
        r = agent.play_ones(mu[t], mu_beta[t], render=False)
        R.append(r)
        print("reward_toword=", R[t])
        if sigmma[t + 1] < 0.00005:
            break
    x = list(range(len(R)))
    plt.subplot(3, 1, 1)
    plt.plot(x, R)
    plt.subplot(3, 1, 2)
    x1 = list(range(len(mu)))
    plt.plot(x1, mu)
    plt.subplot(3, 1, 3)
    plt.plot(x1, mu_beta)
    plt.show()
    time.sleep(5)  # 停顿5s
    agent.env.close()  # 关闭图形化界面
