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

    env.reset(seed=0)

    model = SAC("MlpPolicy",  # MlpPolicy定义策略网络为MLP网络
                env,
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                use_sde=True)
    model.learn(total_timesteps=1000000,
                log_interval=4)
    model.save("sac_door")

