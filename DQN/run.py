import gym
from dqn import DQN

env = gym.make('CartPole-v0')
agent = DQN(env, 2000)
