#import torch
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Dense, Flatten
from image_preprocessing import to_Ychannel
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class dqn():
    
    def __init__(self,
                 env,
                 T=500,
                 discount=0.99,
                 learning_rate=0.0005,
                 d=2,
                 N=6000,
                 C=100):
        # learning algorithm parameters :
        self.env = env
        self.d = d
        self.N = N
        self.T = T
        self.discount = discount
        self.learning_rate = learning_rate
        self.C = C
        
        # env parameters
        self.state_shape = [80, 80, 1]
        self.n_actions = self.env.action_space.n
        
        # memories :
        self.mem = np.array([], dtype='int32')
        self.s_mem = np.array([])
        self.im_mem = np.array([])
        self.ns_mem = np.array([])
        self.a_mem = np.array([], dtype='int32')
        self.r_mem = np.array([])
        self.d_mem = np.array([])
        
        # tensorflow stuff:
        self.sess = tf.Session()
        K.set_session(self.sess)
        self.min = tf.constant(-1., name="min")
        self.max = tf.constant(1., name="max")
        self.Qnet = self.init_Qnet()
        self.loss = self.initialize_loss()
        self.optimizer = self.initialize_optimizer()
        self.copyQnet = keras.models.clone_model(self.Qnet, input_tensors=self.images)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.copyQnet.set_weights(self.Qnet.get_weights())
        
        # additionnal variables :
        #self.dir = os.path.dirname(os.path.realpath(__file__))
        #self.saver = tf.train.Saver()
        self.losses = []
        self.rewards = []
    
    def init_Qnet(self):
        self.images = tf.placeholder("float", shape=[None]+self.state_shape)
        model = Sequential()
        model.add(InputLayer(input_tensor=self.images,
                             input_shape=[None]+self.state_shape))
        
        model.add(Conv2D(32, (5, 5), padding="same", activation="relu"))
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.n_actions))
        
        return model
    
    def initialize_Qnet(self):
        """ConvNet for estimating the state-action value"""
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
    
    def initialize_loss(self):
        # inputs :
        self.action_tensor = tf.placeholder("float", shape=[self.n_actions, None])
        self.reward_tensor = tf.placeholder("float", shape=[None])
        self.maxq = tf.placeholder("float", shape=[None])
        # operations :
        rq = self.reward_tensor + self.maxq
        # the following should be diagonal :
        qval = tf.diag_part(tf.matmul(self.Qnet.output, self.action_tensor)) 
        return tf.reduce_mean(tf.clip_by_value(qval - rq,  self.min, self.max)**2)
    
    def initialize_optimizer(self):
        adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        return adam.minimize(tf.reduce_mean(self.loss))
        
    def sample_action(self, state, eps=0.1):
        """
        Epsilon-greedy sampling of the action through the evaluation by the
        neural net.
        """
        p = np.random.random()
        if p < eps:
            return np.random.choice(self.n_actions)
        else :
            action = self.sess.run(tf.argmax(self.Qnet.output, axis=-1), 
                                             feed_dict={self.images:state})
            return action.item()
    
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
    
    def memory_add(self, elmt): 
        """
        Adds elmt in memory corresponding to mode.
        """
        if not len(self.mem):
            self.mem = np.array([elmt], dtype='int32')
        elif len(self.mem) < self.N-1:
            self.mem = np.insert(self.mem, len(self.mem), elmt, axis=0)
        else:
            self.mem = np.roll(self.mem, -1, axis=0)
            self.mem[-1] = elmt
    
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
    
    def memory_add_s(self, elmt): 
        """
        Adds elmt in memory corresponding to mode.
        """
        if not self.s_mem.any():
            self.s_mem = np.array([elmt])
        elif len(self.s_mem) < self.N:
            self.s_mem = np.insert(self.s_mem, len(self.s_mem), elmt, axis=0)
        else:
            self.s_mem = np.roll(self.s_mem, -1, axis=0)
            self.s_mem[-1] = elmt
    
    def memory_add_ns(self, elmt): 
        """
        Adds elmt in memory corresponding to mode.
        """
        if not self.ns_mem.any():
            self.ns_mem = np.array([elmt])
        elif len(self.ns_mem) < self.N:
            self.ns_mem = np.insert(self.ns_mem, len(self.ns_mem), elmt, axis=0)
        else:
            self.ns_mem = np.roll(self.ns_mem, -1, axis=0)
            self.ns_mem[-1] = elmt
    
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
    
    def memory_add_d(self, elmt): 
        """
        Adds elmt in memory corresponding to mode.
        """
        if not len(self.d_mem):
            self.d_mem = np.array([elmt])
        elif len(self.d_mem) < self.N:
            self.d_mem = np.insert(self.d_mem, len(self.d_mem), elmt, axis=0)
        else:
            self.d_mem = np.roll(self.d_mem, -1, axis=0)
            self.d_mem[-1] = elmt
    
    def sample_from_memory(self, n):
        """Samples n samples from memory, if possible."""
        if len(self.a_mem) == 0:
            return None
        elif len(self.a_mem) <= n+2:
            return np.arange(2, len(self.a_mem))
        else:
            return np.random.choice(np.arange(2, len(self.a_mem)), size=n)
    
    def dq_update(self, state, action, reward, next_state):
        return NotImplemented 
    
    def _episode(self, render=False, batch_size=32, random_start=6000):
        """
        Performs an episode on the environment.
        
        Args :
        sess - The Tensorflow session in which the network computations are done;
        render (bool) - Whether or not to render the episode;
        batch_size (Integer) - Number of selected samples in the minibatch sampled 
        from memory.
        
        Returns:
        Total reward collected during the episode.
        """
        #done = False
        reward = 0.0
        R = 0
        self.env.reset()
        done = False
        # a number of random plays to fill the memory
        i = 0
        if len(self.a_mem) == 0:
            while not done:
                self.env.render()
                action = self.random_action()
                image, r, done, _ = self.env.step(action)
                print(r)
                if i % self.d == 0:
                    reward += r
                    self.memory_add_im(to_Ychannel(image))
                    self.memory_add_r(reward)
                    self.memory_add_a(action)
                else:
                    reward = r
                i += 1
            self.env.reset()
        #it = 0
        for it in tqdm(range(self.T)):
            #print("iteration : %s" % it)
            if it % self.d == 0:
                state = np.array([self.im_mem[-1] - self.im_mem[-2]])
                action = self.sample_action(state)
            image, next_reward, done, _ = self.env.step(action)
            #print("next_reward : %s" % next_reward)
            if it % self.d == 0:
                reward += next_reward
            else:
                reward = next_reward
            if it % self.C == self.C-1:
                self.copyQnet.set_weights(self.Qnet.get_weights())
            if it % self.d == 0:
                #print("reward : %s" % reward)
                self.memory_add_a(action)
                self.memory_add_r(reward)
                self.memory_add_im(to_Ychannel(image))
                
                # select a minibatch of samples in memory and perform a step
                # of gradient descent on it
                batch_ids = np.random.choice(np.arange(2, len(self.a_mem)),
                                             size=batch_size)
                #print("batch_ids : %s" % batch_ids)
                #print("r_mem : %s" % self.r_mem)
                if batch_ids is not None: # delete this ?
                    #print("smem : %r" % (self.s_mem.shape))
                    s_batch = self.im_mem[batch_ids-1] - self.im_mem[batch_ids-2]
                    ns_batch = self.im_mem[batch_ids] - self.im_mem[batch_ids-1]
                    a_batch = self.a_mem[batch_ids]
                    r_batch = self.r_mem[batch_ids]
                    actions = self.to_categorical(a_batch)
                    
                    # check if this is correct and gives the states !
                    #self.im_plot([s_batch[:, 0, :, :, :], s_batch[: 1, :, :, :]])  
                    #print(s_batch[:, 1, :, :, :])
                    maxq = self.discount*self.sess.run(
                        tf.reduce_max(self.copyQnet.output, axis=-1),
                        feed_dict={self.images:ns_batch})
                    _, tmploss = self.sess.run([self.optimizer, self.loss],
                                          feed_dict={self.images:s_batch,
                                                     self.action_tensor:actions,
                                                     self.reward_tensor:r_batch,
                                                     self.maxq:maxq})
                    self.losses.append(tmploss)
                    self.rewards.append(reward)
                R += reward
            it += 1
        return R
        
    def learn(self, I):
        """
        DQN learning algorithm with experience replay.
        
        Args : 
        I (Integer) - the number of episodes to consider in the learning process.
        """
        Rs = []
        for i in range(I):
            Rs.append(self._episode())
        return Rs
                
    def im_plot(self, images):
        plt.figure()
        for im in images:
            if im.shape[-1] == 1:
                im = np.squeeze(im, axis=-1)
            plt.imshow(im, cmap='gray')
            plt.show()














































        
