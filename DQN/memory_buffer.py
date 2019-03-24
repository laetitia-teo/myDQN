import numpy as np

class Buffer():
    
    def __init__(self, maxsize):
        """
        Class for the memory buffer object of the DQN.
        """
        self.N = maxsize
        self._content = {}
    
    def __len__(self):
        if not self._content:
            return 0
        return len(self._content['s'])
    
    def full(self):
        return len(self) >= self.N
    
    def store(self, experience):
        """
        Stores an experience in the replay buffer. Performs a check to see if
        the buffer is full before storing experience.
        """
        if not self._content:
            self.first_add(experience)
        if self.full():
            self._remove_add(experience)
        else:
            self._add(experience)
    
    def first_add(self, experience):
        """
        Adds the first piece of experience.
        """
        s, a, r, n_s = experience
        # state
        self._content['s'] = np.array([s])
        # state
        self._content['a'] = np.array([a])
        # reward
        self._content['r'] = np.array([r])
        # next state
        self._content['n_s'] = np.array([n_s])
    
    def _add(self, experience):
        """
        Simply adds a new piece of experience to the replay buffer.
        """
        s, a, r, n_s = experience
        # state
        self._content['s'] = np.concatenate((self._content['s'], [s]), axis=0)
        # state
        self._content['a'] = np.concatenate((self._content['a'], [a]), axis=0)
        # reward
        self._content['r'] = np.concatenate((self._content['r'], [r]), axis=0)
        # next state
        self._content['n_s'] = np.concatenate((self._content['n_s'], [n_s]), axis=0)
        
    
    def _remove_add(self, experience):
        """
        Shifts all buffer memory to the left, forgetting the first one, and 
        adds the new experience at the rightmost end.
        """
        s, a, r, n_s = experience
        # state
        self._content['s'][:-1] = self._content['s'][1:]
        self._content['s'][-1] = np.array([s])
        # action
        self._content['a'][:-1] = self._content['a'][1:]
        self._content['a'][-1] = np.array([a])
        # reward
        self._content['r'][:-1] = self._content['r'][1:]
        self._content['r'][-1] = np.array([r])
        # next state
        self._content['n_s'][:-1] = self._content['n_s'][1:]
        self._content['n_s'][-1] = np.array([n_s])
    
    def sample(self, batch_size):
        """
        Samples a minibatch of size batch_size from the replay buffer.
        """
        indices = np.random.randint(len(self), size=batch_size)
        states = self._content['s'][indices]
        actions = self._content['a'][indices]
        rewards = self._content['r'][indices]
        n_states = self._content['n_s'][indices]
        return (states, actions, rewards, n_states)






























