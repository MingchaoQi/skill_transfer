import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
import numpy as np
import robosuite_task_zoo
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(
        suite.make(
            "Door",
            robots="IIWA",  
            use_camera_obs=False,  
            has_offscreen_renderer=True,  
            has_renderer=True,  
            reward_shaping=True,  
            control_freq=20,  
        )
    )

    env.reset(seed=0)
    model = SAC.load("sac_door0")

    num_episodes = 200
    max_steps = 1000
    rewards = []  
    steps = []
    for i_ep in range(num_episodes):
        ep_reward = 0
        ep_step = 0
        done = False
        obs, info = env.reset()
        for _ in range(max_steps):
            env.render()
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _,  info = env.step(action)
            ep_reward += reward
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f"回合：{i_ep + 1}，奖励：{ep_reward:.2f}")


    def smooth(data, weight=0.9):
        """
        Used for smooth curves, similar to the smooth curve in Tensorboard
        """
        last = data[0]
        smoothed = []
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point  
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    plt.figure()  
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(rewards, label='Rewards', color="blue")
    plt.legend()
    plt.show()
