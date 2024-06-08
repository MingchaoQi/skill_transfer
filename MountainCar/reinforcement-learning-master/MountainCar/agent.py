from brain import QLearning


class Agent:
    def __init__(self, num_state, num_actions, observation_below, observation_above):
        self.brain = QLearning(num_state, num_actions, observation_below, observation_above)

    def learn(self, observation, action, reward, observation_next):
        self.brain.learn(observation, action, reward, observation_next)

    def choose_action(self, observation, step):
        action = self.brain.choose_action(observation, step)
        return action
