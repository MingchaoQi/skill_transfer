import numpy as np


def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]


class QLearning:
    def __init__(self, num_states, num_actions, observation_below, observation_above, num_digitized = 6, gamma = 0.99, eta = 0.5):
        self.num_actions = num_actions
        self.num_digitized = num_digitized
        self.observation_below = observation_below
        self.observation_above = observation_above
        self.gamma = gamma
        self.eta = eta
        self.q_table = np.random.uniform(low=0, high=1, size=(num_digitized ** num_states, num_actions))

    def digitize_state(self, observation):
        digitized = []
        for o_i, below_i, above_i in zip(observation, self.observation_below, self.observation_above):
            digitized.append(np.digitize(o_i, bins=bins(below_i, above_i, self.num_digitized)))

        return sum([x * (self.num_digitized ** i) for i, x in enumerate(digitized)])

    def learn(self, observation, action, reward, observation_next):
        state = self.digitize_state(observation)
        state_next = self.digitize_state(observation_next)
        max_Q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action] + self.eta * (reward + self.gamma * max_Q_next - self.q_table[state, action])

    def choose_action(self, observation, episode):
        state = self.digitize_state(observation)

        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(self.num_actions)
        return action
