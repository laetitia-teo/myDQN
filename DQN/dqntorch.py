import numpy as np
import torch
import torch.nn.functional as F
#import matplotlib.pyplot as plt
from image_preprocessing import to_Ychannel
from tqdm import tqdm

class Qnet(torch.nn.Module):
    
    def __init__(self, n_actions):
    
        super(Qnet, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(1, 32, 5)
        self.pool = torch.nn.MaxPool2d(2)
        
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        
        self.fc1 = torch.nn.Linear(64 * 18 * 18, 258)
        self.fc2 = torch.nn.Linear(258, n_actions)
    
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class dqn():
    
    def __init__(self,
                 env,
                 T=1000,
                 discount=0.99,
                 lr=0.00025,
                 d=2,
                 N=100000,
                 C=100):
        # algorithm parameters
        self.env = env
        self.d = d
        self.T = T
        self.discount = discount
        self.lr=lr
        self.N = N
        self.C = C
        
        # env parameters
        self.state_space = [1, 80, 80]
        self.n_actions = self.env.action_space.n
        
        # memories
        self.im_mem = np.array([])
        self.s_mem = np.array([])
        self.r_mem = np.array([])
        self.a_mem = np.array([], dtype='int32')
        
        # torch stuff
        self.Qnet = self.initQnet()
        self.Qnetcopy = self.initQnet()
        self.loss = torch.nn.MSELoss(reduction='sum')
        
        # traces
        self.losses = []
        self.rewards = []
    
    def initQnet(self):
        return Qnet(self.n_actions).double()
    '''
    def initQnet(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Linear(1152, 258),
            torch.nn.ReLU(),
            torch.nn.Linear(258, self.n_actions))
        return model.double()
    '''
    def random_action(self):
        return np.random.choice(self.n_actions)
    
    def to_categorical(self, actions):
        """
        Converts an array of actions as integers as a numpy array of
        categorical actions.
        """
        a = np.zeros([self.n_actions, len(actions)], dtype=int)
        a[actions, np.arange(len(actions))] = 1
        return a
    
    def memory_add_im(self, elmt): 
        """
        Adds elmt in memory corresponding to mode.
        """
        if not self.im_mem.any():
            self.im_mem = np.array([elmt])
        elif len(self.im_mem) < self.N:
            self.im_mem = np.insert(self.im_mem, len(self.im_mem), elmt, axis=0)
        else:
            self.im_mem = np.roll(self.im_mem, -1, axis=0)
            self.im_mem[-1] = elmt
    
    def memory_add_r(self, elmt): 
        """
        Adds elmt in memory corresponding to mode.
        """
        if not len(self.r_mem):
            self.r_mem = np.array([elmt])
        elif len(self.r_mem) < self.N:
            self.r_mem = np.insert(self.r_mem, len(self.r_mem), elmt, axis=0)
        else:
            self.r_mem = np.roll(self.r_mem, -1, axis=0)
            self.r_mem[-1] = elmt
    
    def memory_add_a(self, elmt): 
        """
        Adds elmt in memory corresponding to mode.
        """
        if not len(self.a_mem):
            self.a_mem = np.array([elmt], dtype='int32')
        elif len(self.a_mem) < self.N:
            self.a_mem = np.insert(self.a_mem, len(self.a_mem), elmt, axis=0)
        else:
            self.a_mem = np.roll(self.a_mem, -1, axis=0)
            self.a_mem[-1] = elmt
            
    def sample_from_memory(self, n):
        """Samples n samples from memory, if possible."""
        if len(self.a_mem) == 0:
            return None
        elif len(self.a_mem) <= n+2:
            return np.arange(2, len(self.a_mem))
        else:
            return np.random.choice(np.arange(2, len(self.a_mem)), size=n)
    
    def sample_action(self, state, eps=0.1):
        """
        Epsilon-greedy sampling of the action through the evaluation by the
        neural net.
        """
        p = np.random.random()
        if p < eps:
            return np.random.choice(self.n_actions)
        else:
            state = torch.from_numpy(state)
            action = torch.max(self.Qnet(state), -1)[1] #TODO checker l'output de ca
            return action.item()
    
    def random_episodes(self, nep=6, render=False):
        """
        Fills memory with nep episodes of random actions.
        """
        for ep in range(nep):
            done = False
            self.env.reset()
            i = 0
            reward = 0.0
            while not done:
                if render:
                    self.env.render()
                action = self.random_action()
                image, r, done, _ = self.env.step(action)
                if i % self.d == 0:
                    reward += r
                    self.memory_add_im(to_Ychannel(image, data_format='channels_first'))
                    self.memory_add_r(reward)
                    self.memory_add_a(action)
                elif i % self.d == 1:
                    reward = r
                else:
                    reward += r
                i += 1
    
    def episode(self, render=False, batch_size=32):
        """
        Performs an episode on the environment.
        """
        R = 0.0
        self.env.reset()
        reward = 0.0
        done = False
        for it in tqdm(range(self.T)):
            if done:
                break
            if render:
                self.env.render()
            if it % self.d == 0:
                state = np.array([self.im_mem[-1] - self.im_mem[-2]])
                action = self.sample_action(state)
            image, next_reward, done, _ = self.env.step(action)
            R += next_reward
            if it % self.d == 0:
                reward += next_reward
            if it % self.C == 0:
                self.Qnetcopy.load_state_dict(self.Qnet.state_dict())
            if it % self.d == 0:
                #print("reward : %s" % reward)
                self.memory_add_a(action)
                self.memory_add_r(reward)
                self.memory_add_im(to_Ychannel(image, data_format='channels_first'))
                
                # choose a batch
                ids = np.random.choice(np.arange(2, len(self.a_mem)),
                                             size=batch_size)
                
                states = torch.from_numpy(self.im_mem[ids-1] - self.im_mem[ids-2])
                n_states = torch.from_numpy(self.im_mem[ids] - self.im_mem[ids-1])
                actions = self.a_mem[ids]
                rewards = torch.from_numpy(self.r_mem[ids])
                
                maxq = self.discount * torch.max(self.Qnet(n_states), dim=-1)[0]
                q = self.Qnet(states)[np.arange(len(ids)), actions]
                
                loss = self.loss(q, rewards + maxq)
                #print(it, loss.item())
                self.losses.append(loss.item())
                
                self.Qnet.zero_grad()
                loss.backward()
                
                with torch.no_grad():
                    for param in self.Qnet.parameters():
                        param -= self.lr * param.grad.data.clamp_(-1, 1)
        return R
        
    def learn(self, I):
        self.random_episodes(render=True)
        Rs = []
        for i in tqdm(range(I)):
            Rs.append(self.episode())
        return Rs
        
        
                











































        
