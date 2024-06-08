import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import adam_v2
import matplotlib.pyplot as plt
import os
import math
import h5py
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
EPISODES = 10000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)#经验回放最多存储的训练数据对数
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
#建立神经网络
    def _build_model(self):
        model = Sequential()#神经网络是序列式的
        model.add(Dense(8, input_dim=self.state_size, activation='relu'))#
        # model.add(Dense(48, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse',
                      optimizer=adam_v2.Adam(learning_rate=self.learning_rate))#损失函数采用均方差表示
        model.summary()#显示神经网络层数、参数个数等情况
        return model
#存储新的经验回放数据对
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
#w贪婪算法选取动作值
    def DQN_act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(0,self.action_size,1)#随机选取动作
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def rule_act(self, observation, theta, beta):
        position, velocity = observation[0,]
        if velocity >= 0:  # velocity>=0 表示车辆正在向右移动
            if (position > -0.5) and (abs(velocity) < abs(theta)):  # 当到达右侧一定位置无法向右时, 则向左
                action = 0
            else:
                action = 2
        elif velocity < 0:  # velocity<0 表示车辆正在向左移动
            if (position < beta) and (abs(velocity) < abs(theta)):  # 当到达左侧一定位置无法向左时, 则向右
                action = 2
            else:
                action = 0
        return action  # 返回动作
#更新价值动作函数逼近网络的参数值（神经网络的权值）
    def replay(self, batch_size):
        minibatchs = random.sample(self.memory, batch_size)#junyun caiyang
        for minbatch in minibatchs:
            state, action, reward, next_state, done = minbatch
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.target_model.predict(state)

            # 根据公式更新Q表
            # Q(s,a) = R(s,a)+λmax{Q(s`,a`)}
            # target_f 当前状态的Q值 target下一状态的Q值
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
#更新TD目标网络的参数值
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def choose(self, some_list,  probabilities):
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(some_list, probabilities):
            cumulative_probability += item_probability
            if x < cumulative_probability: break
        return item

if __name__ == "__main__":
    print("更改后的DQN方法，两个神经网络（(经典DQN+PPR方法解决mountaincar问题）")
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load('carpole-dqn.h5')
    # f = h5py.File("carpole-dqn.h5","w")
    done = False
    batch_size = 32  # 采样用于更新神经网络的经验回放数据对
    theta = 0.03
    # beta = -0.7
    number = 200
    R = [0] * EPISODES
    S = list(range(EPISODES))
    DQN_reward = list()
    DQN_time = list()
    rule_reward = list()
    rule_time = list()
    x = list()
    m = list()
    i = 0
    j = 0
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        time = 0
        # c = 0
        if e< number:
            c = agent.choose([0, 1], [0.9/number * e, 1 - 0.9/number * e])
        elif e >= number and e < 2*number:
            c = agent.choose([0, 1], [0.9, 0.1])
        else:
            c = agent.choose([0,1], [1.0, 0.0])
        for time in range(500):
            # env.render()
            # action = agent.DQN_act(state)
            if c == 0:
                action = agent.DQN_act(state)
            else:
                action = agent.rule_act(state, theta, random.uniform(-0.65, -0.6))
            next_state, reward, done, _ = env.step(action)
            reward = reward
            # reward = reward if not done else -20

            if done:  # 判断是否训练智能体
                if (next_state[0] >= 0.5):
                    # reward = 10 # policy 2
                    reward = 10 * math.exp(-0.0023 * time + 2.3049)  # policy 2(1)
                    # reward += 10*math.exp(-0.0023*time+2.3049)# policy 3
                # elif (abs(next_state[0]) < 0.5):
                #     reward = -20000 * abs(next_state[1] - 0.5)
            R[e] += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                if c == 0:
                    print("episode: {}/{}, score: {}, e: {:.4}, DQN"
                          .format(e, EPISODES, time, agent.epsilon))
                    print("eposide_reward = ", R[e])
                    DQN_reward.append(R[e])
                    DQN_time.append(time)
                    x.append(i)
                    i = i+1
                    # plt.subplot(2, 1, 1)
                    plt.plot(x, DQN_reward)
                    plt.xlabel("X/eposide")
                    plt.ylabel("reward")
                else:
                    print("episode: {}/{}, score: {}, e: {:.4}, rule"
                          .format(e, EPISODES, time, agent.epsilon))
                    print("eposide_reward = ", R[e])
                    rule_reward.append(R[e])
                    rule_time.append(time)
                    m.append(j)
                    j = j+1
                    # plt.subplot(2, 1, 2)
                    plt.plot(m, rule_reward)
                    plt.xlabel("X/eposide")
                    plt.ylabel("time")
                break
        plt.show()
        plt.pause(0.1)
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            agent.target_train()
            if e % 10 == 0:
                agent.save("carpole-dqn-1.h5")
        # S[e] = time
