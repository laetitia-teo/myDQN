import gym

env = gym.make('Pong-v0')

T = 300
env.reset()

for _ in range(T):
    env.render()
    action = env.action_space.sample()
    env.step(action)

env.close()
