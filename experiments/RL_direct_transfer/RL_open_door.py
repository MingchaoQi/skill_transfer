import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC

if __name__ == "__main__":
    env = GymWrapper(
        suite.make(
            "Door",
            robots="IIWA",  
            use_camera_obs=False,  
            has_offscreen_renderer=False,  
            has_renderer=False,  
            reward_shaping=True,  
            control_freq=20,  
        )
    )

    env.reset(seed=0)
    model = SAC("MlpPolicy", 
                env,
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                use_sde=True)
    model.learn(total_timesteps=1000000,
                log_interval=4)
    model.save("sac_door")

