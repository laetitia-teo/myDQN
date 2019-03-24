import gym
import numpy as np
import os
import tensorflow as tf
from image_preprocessing import preproc
from Qmodels import one_hidden_mlp
from memory_buffer import Buffer
from copy import deepcopy

class DQN():
    """
    Class for the Deep Q Network.
    """
    def __init__(self, env, N, discount=0.995):
        
        # environment 
        self.env = env
        self.a_space = self.env.action_space
        self.n_a = self.a_space.n
        self.s_space = self.env.observation_space
        self.s_shape = self.s_space.shape
        self.discount = discount
        
        # memory buffer
        self.N = N
        self.memory = Buffer(maxsize=self.N)
        
        self.s = tf.placeholder(tf.float32, shape=((None,) + self.s_shape))
        #self.r = tf.placeholder(tf.float32, shape=(None,))
        self.n_s = tf.placeholder(tf.float32, shape=((None,) + self.s_shape))
        self.aslice = tf.placeholder(tf.int32, shape=((None, self.n_a)))
        
        with tf.variable_scope("Q"):
            self.Q = self.Qmodel(self.s)
        with tf.variable_scope("Qclone"):
            self.Qclone = self.Qmodel(self.n_s)
            
        # clone op
        self.clone_ops = []
        var_list = [var for var in tf.get_collection(\
                        tf.GraphKeys.GLOBAL_VARIABLES, scope='Q/')]
        clone_var_list = [var for var in tf.get_collection(\
                              tf.GraphKeys.GLOBAL_VARIABLES,
                              scope='Qclone/')]
        
        # define the cloning operations
        for i, clone_var in enumerate(clone_var_list):
            self.clone_ops.append(clone_var.assign(var_list[i]))
        
        # greedy action
        self.a = tf.argmax(self.Q, 1)
        
        self.frozenvalue = tf.reduce_max(self.Qclone, 1)
        self.objective = tf.placeholder(tf.float32, shape=(None,))
        self.Q_a = tf.gather_nd(self.Q, self.aslice)
        
        self.loss = tf.reduce_sum(\
            tf.squared_difference(self.Q_a, self.objective), 0)
        self.optim = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        
        # saver stuff
        self.save_path = os.getcwd() + "/saves/model"
    
    def play_one(self):
        """
        Plays one episode, with rendering.
        For demontration purposes.
        """
        done = False
        saver = tf.train.Saver()
        s = self.env.reset()
        with tf.Session() as sess:
            try:
                saver.restore(sess, self.save_path)
            except:
                sess.run(tf.global_variables_initializer())
            while not done:
                self.env.render()
                a = sess.run(self.a, feed_dict={self.s: np.array([s])})[0]
                n_s, r, done, _ = self.env.step(a)
                s = n_s
                print(r)
        self.env.close()
    
    def evaluate(self, sess):
        """
        Plays one episode, without rendering.
        For evaluation purposes.
        """
        done = False
        env = deepcopy(self.env)
        s = env.reset()
        R = 0.
        while not done:
            a = sess.run(self.a, feed_dict={self.s: np.array([s])})[0]
            n_s, r, done, _ = env.step(a)
            s = n_s
            R += r
        env.close()
        return R
    
    def Qmodel(self, inpt):
        """
        Model for the Q function.
        """
        return one_hidden_mlp(inpt, 20, self.n_a)
    
    def fill_memory(self, n):
        """
        Plays a certain number of random moves to fill the replay memory with
        n samples of experience.
        """
        i = 0
        s = self.env.reset()
        done = False
        for i in range(n):
            a = self.a_space.sample()
            n_s, r, done, _ = self.env.step(a)
            self.memory.store((s, a, r, n_s))
            s = n_s
            if done:
                self.env.reset()
                done = False
    
    def train(self, n, C, eps=0.1, batch_size=64, record_performance=True):
        """
        Training algorithm for DQN.
        """
        saver = tf.train.Saver()
        i = 0
        self.fill_memory(2000)
        s = self.env.reset()
        done = False
        perf = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            while i < n:
                #print('iteration : %s' % i)
                # play :
                # choose action
                p = np.random.random()
                if p > eps:
                    a = sess.run(self.a, feed_dict={self.s: np.array([s])})[0]
                    #print(a.shape)
                else:
                    a = self.a_space.sample()
                # perform a step
                n_s, r, done, _ = self.env.step(a)
                r = 0.
                if done:
                    r = -1
                # store in memory
                self.memory.store((s, a, r, n_s))
                s = n_s
                if done:
                    s = self.env.reset()
                # replay :
                # sample minibatch
                sbatch, abatch, rbatch, n_sbatch = self.memory.sample(batch_size)
                # gradient update
                value = sess.run(self.frozenvalue, 
                                 feed_dict={self.n_s: n_sbatch})
                objective = rbatch + self.discount * value
                # TODO check this
                indices = np.array([np.arange(batch_size), abatch]).T
                _, loss = sess.run((self.optim, self.loss),
                                   feed_dict={self.s: sbatch,
                                              self.objective: objective,
                                              self.aslice: indices})
                print(loss)
                # every C steps copy Q
                if i % C == 0:
                    sess.run(self.clone_ops)
                    # save model
                    #save_path = saver.save(sess, self.save_path)
                i += 1
                if record_performance:
                    perf.append(self.evaluate(sess))
            save_path = saver.save(sess, self.save_path)
            print('Model saved in path : %s' % save_path)
        self.env.close()
        if record_performance:
            return perf
            
    
    
    
           






























