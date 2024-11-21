import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
import robosuite_task_zoo

if __name__ == "__main__":

    env_cabinet = robosuite_task_zoo.environments.manipulation.Cabinet(
        robots="IIWA",  
        use_camera_obs=False,  
        has_offscreen_renderer=False, 
        has_renderer=False, 
        reward_shaping=True,  
        control_freq=20, 
    )
    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(env_cabinet)

    env.reset(seed=0)
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs_policy_transfer/",
        use_sde=True,
    )

    model.learn(total_timesteps=1200000, log_interval=1)
    model.save("SAC_open_cabinet")
    model.save_replay_buffer("sSAC_open_cabinet_replay_buffer")
