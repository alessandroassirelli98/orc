import numpy as np
import gym
from gym import wrappers
import tensorflow as tf
from tensorflow import keras
from time import sleep
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_squared_error
from keras import layers
from matplotlib import pyplot as plt


env = gym.make('Pendulum')
nx = env.observation_space.shape
nu = env.action_space.shape

class DQNAgent:
    def __init__(self):

        self.QVALUE_LEARNING_RATE = 1e-3
        self.DISCOUNT = 0.99
        self.MAX_MEMORY_BUFF = 50_000

        self.lr = 0.001
        self.exploration_prob = 1

        self.memory_buff = []

        inputs = layers.Input(shape=(nx+nu,1))
        state_out1 = layers.Dense(16, activation="relu")(inputs) 
        state_out2 = layers.Dense(32, activation="relu")(state_out1) 
        state_out3 = layers.Dense(64, activation="relu")(state_out2) 
        state_out4 = layers.Dense(64, activation="relu")(state_out3)
        outputs = layers.Dense(1)(state_out4) 

        self.Q = tf.keras.Model(inputs, outputs)
        self.Q.compile(loss="mse",optimizer = Adam(lr=self.lr))

        self.Q_target = self.Q

    def compute_action(self, state):
        if np.random.uniform() < self.exploration_prob :
                # take random action
                u = np.random.randint(0, env.nu)
        else:
            u = np.argmax(self.Q.predict(state))

    def train(self, xu_batch, reward_batch, xu_next_batch, done_batch):
        X = []
        Y = []
        for current_state, reward, next_state, done in xu_batch, reward_batch, xu_next_batch, done_batch:
            if not done:
                max_future_Q = np.max(self.Q_target.predict(next_state))
                new_Q = reward + self.DISCOUNT * max_future_Q
            else:
                new_Q = reward

            X.append(current_state)
            Y.append(new_Q)

        self.Q.fit(X, Y)

            