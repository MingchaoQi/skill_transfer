import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
import robosuite_task_zoo

if __name__ == "__main__":

    env_cabinet = robosuite_task_zoo.environments.manipulation.Cabinet(
        robots="IIWA",  # use Sawyer robot
        use_camera_obs=False,  # do not use pixel observations
        has_offscreen_renderer=False,  # not needed since not using pixel obs
        has_renderer=False,  # make sure we can render to the screen
        reward_shaping=True,  # use dense rewards
        control_freq=20,  # control should happen fast enough so that simulation looks smooth
    )
    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(env_cabinet)

    env.reset(seed=0)
    model = SAC.load("sac_door")
    # model.load_replay_buffer("sac_cabinet_replay_buffer")
    model.set_env(env, force_reset=True)

    # model = SAC("MlpPolicy",  # MlpPolicy定义策略网络为MLP网络
    #             env,
    #             verbose=1,
    #             tensorboard_log="./tensorboard_logs_policy_transfer/",
    #             use_sde=True)
    model.learn(total_timesteps=1200000,
                log_interval=1)
    model.save("sac_cabinet2")
    model.save_replay_buffer("sac_cabinet_replay_buffer2")
    # # Save the policy independently from the model
    # # Note: if you don't save the complete model with `model.save()`
    # # you cannot continue training afterward
    # policy = model.policy
    # policy.save("sac_policy_pendulum")
    # loaded_model.load_replay_buffer("sac_replay_buffer")


