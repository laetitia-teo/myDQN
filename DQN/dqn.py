#import torch
import numpy as np
import tensorflow as tf
from image_preprocessing import to_Ychannel
import matplotlib.pyplot as plt
import os

class dqn():
    
    INIT = 0
    LOAD = 1

    def __init__(self, env, T=1000, discount=0.99, learning_rate=0.01, d=2, N=100000):
        self.env = env
        self.d = d
        # Neural net that takes a state as input and outputs a vector of probabilies
        # on action space
        self.state_shape = [80, 80, 1] # channel is the number of images
        self.n_actions = self.env.action_space.n
        self.Qnetwork = self.initialize_Qnetwork()
        #self.Qnetwork_prev = self.Qnetwork
        self.memory = self.initialize_memory()
        self.discount = discount
        self.learning_rate = learning_rate
        self.N = N
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.T = T
    
    def initialize_memory(self):
        return []
    
    def memory_add(self, elmt):
        if len(self.memory) >= self.N:
            self.memory.pop(0)
        self.memory.append(elmt)
    
    def initialize_Qnetwork(self):
        """Model function for Q-network"""
        # None or number of frames presented to the network ?
        # Consider feeding difference frames
        self.images = tf.placeholder("float", shape=[None]+self.state_shape)
        num_filters = [32, 64, 64]
        kernel = [(5, 5), (4, 4), (3, 3)]
        
        # Max pooling the input ?        
        
        # First layer 
        conv1 = tf.layers.conv2d(
            inputs=self.images,
            filters=num_filters[0],
            kernel_size=kernel[0],
            padding="same",
            data_format="channels_last",
            activation=tf.nn.relu,
            name="conv1")
        layer1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2)
        
        # Second layer
        conv2 = tf.layers.conv2d(
            inputs=layer1,
            filters=num_filters[1],
            kernel_size=kernel[1],
            padding="same",
            data_format="channels_last",
            activation=tf.nn.relu,
            name="conv2")
        layer2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=2)
        
        # Third layer
        conv3 = tf.layers.conv2d(
            inputs=layer2,
            filters=num_filters[2],
            kernel_size=kernel[2],
            padding="same",
            data_format="channels_last",
            activation=tf.nn.relu,
            name="conv3")
        layer3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2, 2), strides=2)
        # Rest of the layers
        
        flat = tf.reshape(layer3, shape=[-1, 100*64])
        dense = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.4)
        
        return tf.layers.dense(inputs=dropout, units=self.n_actions)
        
    def loss(self, state, action, reward, next_state):
        q_value = self.Qnetwork[action]
        q_value_prev = tf.max(self.Qnetwork)
        return (reward + self.discount*q_value_prev - q_value)**2
    
    def sample_action(self, sess, state, eps=0.01):
        """
        Epsilon-greedy sampling of the action through the evaluation by the
        neural net.
        """
        p = np.random.random()
        if p < eps:
            return np.random.choice(self.n_actions)
        else :
            inpt = np.array([state[0] - state[1]]) # TODO : generalize this !
            print(inpt.shape)
            action = sess.run(tf.argmax(self.Qnetwork, axis=-1), feed_dict={self.images:inpt})
            return action.item()
    
    def random_action(self):
        return np.random.choice(self.n_actions)
    
    def dq_update(self, state, action, reward, next_state):
        return NotImplemented 
    
    def _episode(self, sess, render=False, s=5):
        """
        Performs an episode on the environment.
        
        Must be called from within the learning loop !
        (Because the Tensorflow session is defined in it).
        
        Args :
        sess - The Tensorflow session in which the network computations are done
        render (bool) - Whether or not to render the episode
        s (Integer) - Number of selected samples in the minibatch sampled from memory
        
        Returns:
        Total reward collected during the episode.
        """
        done = False
        state = np.zeros(self.d)
        state = [to_Ychannel(self.env.reset())]
        # 2 iterations to fill the state space
        for i in range(self.d-1):
            action = self.random_action()
            state.append(to_Ychannel(self.env.step(action)[0]))
        state = np.array(state)
        print(state.shape)
        R = 0
        it = 0
        while not done and it < self.T:
            print("iteration : %s" % it)
            if it % self.d == 0:
                action = self.sample_action(sess, state)
            print(action)
            next_image, reward, done, _ = self.env.step(action)
            if it == self.T - 1 or it == self.T - 2: # TODO : generalize this !
                done = True
            next_state = np.append(state[1:], to_Ychannel(next_image))
            state = next_state
            R += reward
            if it % self.d == 0:
                self.memory_add(np.array([state, action, reward, done, next_state]))
                # select a minibatch of samples in memory and perform a step
                # of gradient descent on it
                minibatch = np.random.choice(self.memory, size=s) # TODO : FIX THIS (empty mem)
                loss = np.zeros(s)
                for i, [s, a, r, done, n_s] in enumerate(minibatch):
                    if done: # this is probably not correct
                        loss[i] = (self.Qnetwork[a] - r)**2
                    else:
                        rq = r + self.discount*tf.max(self.Qnetwork)
                        loss[i] = (self.Qnetwork[a] - rq)**2
                opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss)
                # check if this is correct and gives the states !
                inputs = minibatch[:, 0][0] - minibatch[:, 0][1] # TODO : generalize this !
                _, tmploss = sess.run([opt, loss], feed_dict={self.images:inputs})
            it += 1
        return R
        
    def learn(self, I):
        """
        DQN learning algorithm with experience replay.
        
        Args : 
        I (Integer) - the number of episodes to consider in the learning process.
        """
        Rs = []
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for i in range(I):
                Rs.append(self._episode(sess))
        return Rs
                
    def im_plot(self, images):
        plt.figure()
        for im in images:
            plt.subplot()
            plt.imshow(im, cmap='gray')
        plt.show()














































        
