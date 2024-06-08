#使用openai最新的gymnasium库
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

MAX_NUM_EPISODES = 5000
STEPS_PER_EPISODE = 200 # This is specific to MountainCar. May change with env
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
alpha = 0.05 # 学习率
gamma = 0.98 # 折扣率
num_digitized = 50 # 离散化连续状态空间的分桶数量

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

class Q_Learning(object):
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins = num_digitized # 离散化连续状态空间的分桶数量
        # self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        # Create a multi-dimensional array (aka. Table) to represent the Q-values
        # Q_table的形式是根据离散化后状态空间的形式来决定的
        # self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1, self.obs_bins + 1, self.obs_bins + 1,
        #                    self.action_shape)) # (51 × 51 × 3)
        self.Q = np.random.uniform(low=0, high=1, size=(num_digitized ** self.num_states, self.num_actions))
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor
        self.epsilon = 1.0
    
    # def discretize(self, obs):
    #    return tuple(((obs - self.obs_low) / self.bin_width).astype(int))
    
    def discretize(self, observation):
        discretize = []
        for o_i, below_i, above_i in zip(observation, self.obs_low, self.obs_high):
            discretize.append(np.digitize(o_i, bins=bins(below_i, above_i, self.obs_bins)))

        return sum([x * (self.obs_bins ** i) for i, x in enumerate(discretize)])

   
    def get_action(self, obs):
        discretized_obs = self.discretize(obs)# obs是状态空间的观测，返回的是一个numpy中的二维数组
        # (贪婪策略选择)
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:
           return np.argmax(self.Q[discretized_obs][:])
        else:  # 选择一个随机动作
           return np.random.choice([a for a in range(self.num_actions)])
        
    def learn(self, obs, action, reward, next_obs):
        discretized_obs = self.discretize(obs)
        discretized_next_obs = self.discretize(next_obs)
        td_target = reward + self.gamma * np.max(self.Q[discretized_next_obs])
        td_error = td_target - self.Q[discretized_obs][action]
        self.Q[discretized_obs][action] += self.alpha * td_error
        

        
def train(agent, env):
    return_list = [] #记录每一个episode的return
    best_reward = -float('inf')
    episode_num = 0#控制plt从第几个episode开始画
    for episode in range(MAX_NUM_EPISODES):
        terminated = False
        obs = env.reset()[0] #gym版本更新后，返回的不是一个state了，而是一个tuple
        total_reward = 0.0
        while not terminated:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)#返回值数量改变
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
        episode_num += 1
        if episode_num >=100:
            return_list.append(total_reward)
        
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode,
                                     total_reward, best_reward, agent.epsilon))
        
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Q-learning on {}'.format('mountain_car_v0'))
    plt.show()
    # Return the trained policy
    return np.argmax(agent.Q, axis=2)

def test(agent, env, policy):
    terminated = False
    obs = env.reset()[0]
    total_reward = 0.0
    while not terminated:
        action = policy[agent.discretize(obs)]
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward

if __name__ == "__main__":
    env =gym.make('MountainCar3D-v0', render_mode = 'rgb_array') #原来的env.render()方法现在已经不可用，注意如果选择render="human"会导致训练速度大幅下降
    agent = Q_Learning(env)
    learned_policy = train(agent, env)
    # Gym Monitor wrapper 的使用，改为gym.wrappers.RecordVideo
    # gym_monitor_path = "./gym_monitor_output"
    # env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)
    for _ in range(1000):
        test(agent, env, learned_policy)
    env.close()
