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
            robots="IIWA",  # use Sawyer robot
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=True,  # not needed since not using pixel obs
            has_renderer=True,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
        )
    )
    # env_cabinet = robosuite_task_zoo.environments.manipulation.Cabinet(
    #     robots="IIWA",  # use Sawyer robot
    #     use_camera_obs=False,  # do not use pixel observations
    #     has_offscreen_renderer=True,  # not needed since not using pixel obs
    #     has_renderer=True,  # make sure we can render to the screen
    #     reward_shaping=True,  # use dense rewards
    #     control_freq=20,  # control should happen fast enough so that simulation looks smooth
    # )
    # # Notice how the environment is wrapped by the wrapper
    # env = GymWrapper(env_cabinet)

    env.reset(seed=0)
    model = SAC.load("sac_door0")

    num_episodes = 200
    max_steps = 1000
    rewards = []  # 记录所有回合的奖励
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
        用于平滑曲线，类似于Tensorboard中的smooth曲线
        """
        last = data[0]
        smoothed = []
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # plt.title("开抽屉到开门直接策略迁移奖励曲线")
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(rewards, label='Rewards', color="blue")
    # plt.plot(smooth(rewards), label='平滑奖励曲线', color="red")
    # plt.ylim([1000, max(rewards)])
    plt.legend()
    plt.show()
    # plt.savefig(fname='./直接开抽屉奖励测试函数曲线' + '.png', format='png')
        # all_episode_rewards.append(sum(episode_rewards))

    # mean_episode_reward = np.mean(all_episode_rewards)
    # print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
