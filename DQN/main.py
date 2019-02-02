import gym
import time
import image_preprocessing as ip
import matplotlib.pyplot as plt
import dqn
import os

env = gym.make('Pong-v0')
DQN = dqn.dqn(env)

def plot(d):
    plt.figure()
    plt.plot(d.losses)
    plt.plot(d.rewards)
    plt.show()
#r = DQN.learn(1)
#print(r)
#plt.plot(r)
