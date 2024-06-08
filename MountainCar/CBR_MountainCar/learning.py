import gymnasium as gym

# # env = gym.make('MountainCar3D-v0')
env = gym.make('MountainCar-v0')
# print(env.reset())
# print(type(env.reset()[0]))
# print(env.observation_space.low)
# print(env.reset()[0] - (env.observation_space.low))
# print(env.action_space.n)
# print(env.observation_space.shape[0])
# # import numpy as np

# print(np.linspace(-50, 50, 3))
# # print(type(np.linspace(-50, 50, 10)))
# print(np.linspace(-50, 50, 3)[:-1])

# import numpy as np

# # print(np.zeros((1,3)))

# # case_base = np.zeros((1,3))
# # print(case_base)
# # def case_generation(state, action):
# #     global case_base
# #     case_base = np.append(case_base, np.append(state, action))
# #     return case_base

# # case_generation(np.array([1, 1]), np.array([1]))
# # print(case_base)

# # a = np.zeros((2,3))
# # print(a)
# # np.save("case_base.npy", a)
# b = np.load("case_base.npy")
# print(b)

state = env.reset()[0]
