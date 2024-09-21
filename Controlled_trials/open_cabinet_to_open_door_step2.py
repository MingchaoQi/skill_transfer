import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC

if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(
        suite.make(
            "Door",
            robots="IIWA",  # use Sawyer robot
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=False,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
        )
    )
    # env = env.gym.make(render="human")
    env.reset(seed=0)
    model_reference = SAC.load("sac_cabinet2")
    model = SAC("MlpPolicy",  # MlpPolicy定义策略网络为MLP网络
                env,
                verbose=1,
                tensorboard_log="./tensorboard_logs_policy_transfer2/",
                use_sde=True)
    model.policy.actor = model_reference.policy.actor
    model.learn(total_timesteps=1000000,
                log_interval=1)
    model.save("sac_transfer")
    model.save_replay_buffer("sac_transfer_buffer2")

    # policy = model.policy
    # print(policy.actor)
    # model.load_replay_buffer("sac_cabinet_replay_buffer2")
    # model.set_env(env, force_reset=True)
    # model.learn(total_timesteps=1000000,
    #             log_interval=1)
    # obs, info = env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     env.render
    #     # if terminated or truncated:
    #     #     obs, info = env.reset()