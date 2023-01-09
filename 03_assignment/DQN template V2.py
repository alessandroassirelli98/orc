import os
import tensorflow as tf
from tensorflow.keras import layers
from keras.callbacks import TensorBoard
import numpy as np
import random
from numpy.random import randint, uniform
import gym
import time 
from dpendulum import DPendulum
from ddoublependulum import DDoublePendulum

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
 
def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()


#...

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.writer.flush()

QVALUE_LEARNING_RATE = 1e-3
DISCOUNT = 0.99
BATCH_SIZE = 32
MEMORY_BUFFER_LENGTH = 10_000
MIN_BUFFER_TO_TRAIN = 1000
EXPLORATION_PROBABILITY_DECAY = 0.0001
MIN_EXPLORATION_PROBABILITY = 0.1
TARGET_NETWORK_UPDATE_FREQUENCY = 10

class DQNAgent():
    
    def __init__(self, load_file = None):
        # Main model
        self.model = self.get_critic()

        # Target network
        self.target_model = self.get_critic()
        if load_file:
            self.model.load_weights(load_file)
        self.target_model.set_weights(self.model.get_weights())

        self.critic_optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)
        self.loss_fcn = tf.keras.losses.MeanSquaredError()
        self.loss_value = 0

        # An array with last n steps for training
        self.replay_memory = []

        self.exploration_probability = 1

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    
    def get_critic(self):
        ''' Create the neural network to represent the Q function '''
        inputs = layers.Input(shape=(2,))
        # state_out1 = layers.Dense(16, activation="relu")(inputs) 
        # state_out2 = layers.Dense(32, activation="relu")(state_out1) 
        state_out3 = layers.Dense(64, activation="relu")(inputs) 
        state_out4 = layers.Dense(64, activation="relu")(state_out3)
        outputs = layers.Dense(ndu)(state_out4) 

        model = tf.keras.Model(inputs, outputs)
        # model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        model.summary()
        return model

    
    def store_episode(self, current_state, action, cost, next_state, done):
        #We use a dictionnary to store them
        self.replay_memory.append({
            "current_state":current_state,
            "action":action,
            "cost":cost,
            "next_state":next_state,
            "done" :done
        })
        # If the size of memory buffer exceeds its maximum, we remove the oldest experience
        if len(self.replay_memory) > MEMORY_BUFFER_LENGTH:
            self.replay_memory.pop(0)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model(np.array(state).reshape(1,-1))[0]

    def update_probability(self, step):
        if self.exploration_probability > MIN_EXPLORATION_PROBABILITY:
            self.exploration_probability = np.exp(-EXPLORATION_PROBABILITY_DECAY * step)
    
    def compute_action(self):
        if np.random.uniform() < self.exploration_probability:
            # take random action
            iu = np.array([np.random.randint(0, ndu)])
        else:
            iu = np.array([np.argmin(agent.get_qs(current_state))])
        
        return iu

    def train(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_BUFFER_TO_TRAIN:
            return

        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition['current_state'] for transition in minibatch])
        current_qs_list = tf2np(self.model(current_states))

        next_states = np.array([transition['next_state'] for transition in minibatch])
        future_qs_list = tf2np(self.target_model(next_states))

        X = []
        Y = []
        
        for index, transition in enumerate(minibatch):
            if transition["done"]:
                y = transition["cost"]
            else:
                y = transition["cost"] + DISCOUNT * np.min(future_qs_list[index])
            
            current_q = current_qs_list[index]
            current_q[transition["action"]] = y

            X.append(transition["current_state"])
            Y.append(current_q)

        X = np.array(X)
        Y = np.array(Y)
        self.update(X, Y)

        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > TARGET_NETWORK_UPDATE_FREQUENCY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


    
    def update(self, x, y):
        ''' Update the weights of the Q network using the specified batch of data '''
        # all inputs are tf tensors
        with tf.GradientTape() as tape:         
            # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
            # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. 
            # Tensors can be manually watched by invoking the watch method on this context manager.
            logits = self.model(x, training=True)
            self.loss_value = self.loss_fcn(y, logits)
        # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
        Q_grad = tape.gradient(self.loss_value, self.model.trainable_variables)          
        # Update the critic backpropagating the gradients
        self.critic_optimizer.apply_gradients(zip(Q_grad, self.model.trainable_variables))


def xy_to_t(state):
    return np.array([np.arctan2(state[1], state[0]), state[2]])

def t_to_xy(state):
    return np.array([np.cos(state[0]), np.sin(state[0]), state[1]])

uMax = 3
ndu = 3
env = DPendulum(ndu=ndu, uMax=uMax, dt=0.05)
nx = env.nx
nu = env.nu

SHOW_PREVIEW = False
AGGREGATE_STATS_EVERY = 5
MAX_NUMBER_OF_EPISODES = 1000
STEP_BEFORE_TRAIN = 4

ep_costs = []
ep_accuracy = []

average_accuracy_history = [0]
average_cost_history = [1e6]

MODEL_NAME = "costscaled_Pendulum_d64_d64_ndu" + str(ndu) + "uMax" + str(uMax) + "_"

agent = DQNAgent()
step = 1
for episode in range (1, MAX_NUMBER_OF_EPISODES):
    agent.tensorboard.step = episode
    print("Episode ", episode)

    episode_cost = 0
    current_state = env.reset()
    done = False
    jj = 0
    while not done:
        iu = agent.compute_action()
        next_state, cost, done = env.step(iu)

        episode_cost += cost

        agent.store_episode(current_state, iu, cost, next_state, done)
        agent.update_probability(step)
        if not jj % STEP_BEFORE_TRAIN or done:
            agent.train(done)

        if done: ep_accuracy.append((1-cost)*100)
        
        current_state = next_state
        step += 1
        jj += 1
    

    # Append episode cost to a list and log stats (every given number of episodes)
    ep_costs.append(episode_cost)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_cost = (sum(ep_costs[-AGGREGATE_STATS_EVERY:])/len(ep_costs[-AGGREGATE_STATS_EVERY:]))
        average_accuracy = (sum(ep_accuracy[-AGGREGATE_STATS_EVERY:])/len(ep_accuracy[-AGGREGATE_STATS_EVERY:]))
        min_cost = min(ep_costs[-AGGREGATE_STATS_EVERY:])
        max_cost = max(ep_costs[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(cost_avg=average_cost, 
                                        accuracy_avg=average_accuracy,
                                        cost_min=min_cost, 
                                        cost_max=max_cost, 
                                        epsilon=agent.exploration_probability, 
                                        loss=agent.loss_value)

        if average_accuracy > max(average_accuracy_history):
            agent.model.save_weights(MODEL_NAME + "pendulum_best_accuracy.h5")
        if average_cost > min(average_cost_history):
            agent.model.save_weights(MODEL_NAME + "pendulum_best_cost.h5")

        average_accuracy_history.append(average_accuracy)
        average_cost_history.append(average_cost)

# agent.model.load_weights("Pendulum_d64_d64_ndu3uMax3_pendulum_best.h5")

current_state = env.reset().reshape(1,-1)
done = False
while not done:
    action = np.array([np.argmin(agent.model(current_state))])
    next_state, r, done = env.step(action)
    current_state = next_state.reshape(1,-1)
    env.render()
    if done: break