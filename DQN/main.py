import gym
import keras
import time
import image_preprocessing as ip
import matplotlib.pyplot as plt
import dqn

env = gym.make('Pong-v0')
"""
T = 100
env.reset()

for _ in range(T):
    env.render()
    action = env.action_space.sample()
    observation, reward, _, _ = env.step(action)
    print(reward)
print(observation.shape)
print(ip.to_Ychannel(observation))
env.close()
print(type(ip.to_Ychannel(observation)))
plt.imshow(ip.to_Ychannel(observation), cmap='gray')
plt.show()
print(ip.to_Ychannel(observation).shape)
"""
DQN = dqn.dqn(env)
print(DQN.learn(1))
