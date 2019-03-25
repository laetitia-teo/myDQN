# Deep Q Network

This directory contains an implementation of the seminal paper [Human-level control through deep reinforcement learning]. The paper builds on approximate Q-learning with deep neural network approximators, with several tricks that allow for efficient training.

### Modules

Tu run, go in the DQN folder and use :

```python
from run import *
agent.train(7000, 100)
```

The agent architecture is defined in dqn.py.
The memory buffer is defined in memory_buffer.py
The module Qmodels defines different neural approximators, that may be used for different environments.

### Results

![CartPole](https://github.com/laetitia-teo/myDQN/images/cartpole2.png)

Average of 5 runs on the gym CartPole-v0 environment over 7000 iterations.
