import gym
from agent import Agent

NUM_EPISODES = 2000
SUCCEED_EPISODES = 20
MAX_STEPS = 200
SUCCEED_STEPS = 180


class Environment:
    def __init__(self, env):
        self.env = gym.make(env, MAX_STEPS)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        # observation_below = self.env.observation_space.low
        # observation_above = self.env.observation_space.high
        observation_below = [-2.4, -3.0, -0.5, -2.0]
        observation_above = [2.4, 3.0, 0.5, 2.0]
        self.agent = Agent(num_states, num_actions, observation_below, observation_above)

    def train(self):
        complete_episodes = 0
        is_episode_final = False

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()
            for step in range(MAX_STEPS):
                action = self.agent.choose_action(observation, episode)
                observation_next, _, done, _ = self.env.step(action)
                if done:
                    if step < SUCCEED_STEPS:
                        reward = -1
                        complete_episodes = 0
                    else:
                        reward = 1
                        complete_episodes += 1
                else:
                    reward = 0

                self.agent.learn(observation, action, reward, observation_next)
                observation = observation_next
                if done:
                    print('Episode {0}: The game ends after {1} steps'.format(episode, step + 1))
                    break
            if is_episode_final:
                break
            if complete_episodes >= SUCCEED_EPISODES:
                print('Training is over')
                is_episode_final = True

    def test(self):
        observation = self.env.reset()
        for step in range(MAX_STEPS):
            self.env.render()
            action = self.agent.choose_action(observation, 100000)
            observation_next, _, done, _ = self.env.step(action)
            observation = observation_next
            if done:
                print('Test: The game ends after {0} steps'.format(step + 1))
                break





















