from environment import Environment

if __name__ == '__main__':
    ENV = 'MountainCar3D-v0'
    env = Environment(ENV)
    env.train()
    env.test()