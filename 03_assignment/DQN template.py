import tensorflow as tf
from tensorflow.keras import layers
from keras.callbacks import TensorBoard
import numpy as np
import random
from numpy.random import randint, uniform
import gym
import time 

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
    import os
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

class DQNAgent():
    def __init__(self):
        # Main model
        self.model = self.get_critic(nx, nu)

        # Target network
        self.target_model = self.get_critic(nx, nu)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = []

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
    
    def get_critic(self, nx, nu):
        ''' Create the neural network to represent the Q function '''
        inputs = layers.Input(shape=(2,))
        state_out1 = layers.Dense(16, activation="relu")(inputs) 
        state_out2 = layers.Dense(32, activation="relu")(state_out1) 
        state_out3 = layers.Dense(64, activation="relu")(state_out2) 
        state_out4 = layers.Dense(64, activation="relu")(state_out3)
        outputs = layers.Dense(2)(state_out4) 

        model = tf.keras.Model(inputs, outputs)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        model.summary()
        return model

    def store_episode(self, current_state, action, reward, next_state, done):
        #We use a dictionnary to store them
        self.replay_memory.append({
            "current_state":current_state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
            "done" :done
        })
        # If the size of memory buffer exceeds its maximum, we remove the oldest experience
        if len(self.replay_memory) > MEMORY_BUFFER_LENGTH:
            self.replay_memory.pop(0)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(1,-1))[0]

    def train(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_BUFFER_TO_TRAIN:
            return

        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition['current_state'] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        next_states = np.array([transition['next_state'] for transition in minibatch])
        future_qs_list = self.target_model.predict(next_states)

        X = []
        Y = []
        for index, transition in enumerate(minibatch):
            if transition["done"]:
                y = transition["reward"]
            else:
                y = transition["reward"] + DISCOUNT * np.max(future_qs_list[index])
            
            current_q = current_qs_list[index]
            current_q[transition["action"]] = y

            X.append(transition["current_state"])
            Y.append(current_q)

        X = np.array(X)
        Y = np.array(Y)
        self.model.fit(X, Y, batch_size=BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > TARGET_NETWORK_UPDATE_FREQUENCY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

def xy_to_t(state):
    return np.array([np.arctan2(state[1], state[0]), state[2]])

def t_to_xy(state):
    return np.array([np.cos(state[0]), np.sin(state[0]), state[1]])

    # def update(xu_batch, reward_batch, action_batch, xu_next_batch):
    #     ''' Update the weights of the Q network using the specified batch of data '''
    #     # all inputs are tf tensors
    #     with tf.GradientTape() as tape:         
    #         # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
    #         # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. 
    #         # Tensors can be manually watched by invoking the watch method on this context manager.
    #         target_values = Q_target(xu_next_batch, training=True)
    #         # Compute 1-step targets for the critic loss
    #         y = reward_batch + DISCOUNT* np.max(target_values)                        
    #         # Compute batch of Values associated to the sampled batch of states
    #         Q_value = Q(xu_batch, training=True)[0][action_batch]                         
    #         # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
    #         Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))  
    #     # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
    #     Q_grad = tape.gradient(Q_loss, Q.trainable_variables)          
    #     # Update the critic backpropagating the gradients
    #     critic_optimizer.apply_gradients(zip(Q_grad, Q.trainable_variables))    


env = gym.make('Pendulum')
nx = 2
nu = 1
Q_episodes_history = []
Q_max_history = []

MODEL_NAME = "DQN0"
SHOW_PREVIEW = False
AGGREGATE_STATS_EVERY = 5

QVALUE_LEARNING_RATE = 1e-3
DISCOUNT = 0.99
BATCH_SIZE = 32
MAX_NUMBER_OF_EPISODES = 1000
MAX_EPISODE_LENGTH = 201
MEMORY_BUFFER_LENGTH = 50_000
# STEP_BEFORE_TRAIN = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 5
MIN_BUFFER_TO_TRAIN = 500
EXPLORATION_PROBABILITY_DECAY = 0.001
MIN_EXPLORATION_PROBABILITY = 0.1

exploration_prob = 1
ep_rewards = []

control_map = [-1, 1]

agent = DQNAgent()

for episode in range (MAX_NUMBER_OF_EPISODES):
    agent.tensorboard.step = episode

    episode_reward = 0
    current_state = xy_to_t(env.reset())
    done = False
    for iter in range(MAX_EPISODE_LENGTH):
        print("Episode ", episode, "step: ", iter)
        if np.random.uniform() < exploration_prob :
            # take random action
            u = np.array([np.random.choice(control_map)])
        else:
            u = np.array([control_map[np.argmax(agent.get_qs(current_state))]])

        next_state, reward, done, _ = env.step(u)
        episode_reward += reward
        next_state = xy_to_t(next_state)

        agent.store_episode(current_state, u, reward, next_state, done)
        agent.train(done)

        if done:
            continue

        current_state = next_state
        step += 1

        # Decay epsilon
        if exploration_prob > MIN_EXPLORATION_PROBABILITY:
            exploration_prob = np.exp(-EXPLORATION_PROBABILITY_DECAY * step)
            exploration_prob = max(exploration_prob, MIN_EXPLORATION_PROBABILITY)

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=exploration_prob)

        # Save model, but only when min reward is greater or equal a set value
        # if min_reward >= MIN_REWARD:
        #     agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')





current_state=xy_to_t(env.reset())
for step in range(MAX_EPISODE_LENGTH):
    action = np.array([control_map[np.argmax(agent.get_qs(current_state))]])
    next_state, r, done, _ = env.step(action)
    current_state = xy_to_t(next_state)
    env.render()
    # time.sleep(0.01)











# --------------------------------------------------------------- #
# Create critic and target NNs
# Q.summary()

# Set initial weights of targets equal to those of the critic
# Q_target.set_weights(Q.get_weights())

# Set optimizer specifying the learning rates
# critic_optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)

# step = 0
# for episode in range (MAX_NUMBER_OF_EPISODES):
#     print("------ Episode ", episode, "over ", MAX_NUMBER_OF_EPISODES, " ------")
#     current_state = xy_to_t(env.reset())
#     for k in range(MAX_EPISODE_LENGTH):
#         print("Episode ", episode, "step: ", k)
#         if np.random.uniform() < exploration_prob :
#                 # take random action
#                 u = np.array([np.random.choice(control_map)])
#         else:
#             u = np.array([control_map[np.argmax(Q.predict(current_state.reshape(1,-1)))]])
        
#         next_state, reward, done, _ = env.step(u)
#         next_state = xy_to_t(next_state)
#         store_episode(memory_buffer, current_state, u, reward, next_state, done)
#         current_state = next_state
#         step += 1

#         if (step > MIN_BUFFER_TO_TRAIN):

#             np.random.shuffle(memory_buffer)
#             batch_sample = memory_buffer[: BATCH_SIZE]
#             xu_batch = [x["current_state"] for x in batch_sample]
#             reward_batch = [x["reward"] for x in batch_sample]
#             xu_next_batch = [x["next_state"] for x in batch_sample]
#             action_batch = [x["action"] for x in batch_sample]

#             if (step % TARGET_NETWORK_UPDATE_FREQUENCY == 0):
#                 Q_target.set_weights(Q.get_weights())

#             for xu, reward, action, xu_next in zip(xu_batch, reward_batch, action_batch, xu_next_batch):
#                 update(xu, reward, action, xu_next)

#         exploration_prob = np.exp(-EXPLORATION_PROBABILITY_DECAY * step)
#         exploration_prob = max(exploration_prob, MIN_EXPLORATION_PROBABILITY)
#         Q_max_history.append(np.max(Q.predict(current_state.reshape(1,-1))))
#     Q_episodes_history.append(Q_max_history)


# w = Q.get_weights()
# for i in range(len(w)):
#     print("Shape Q weights layer", i, w[i].shape)
    
# for i in range(len(w)):
#     print("Norm Q weights layer", i, np.linalg.norm(w[i]))
    
# print("\nDouble the weights")
# for i in range(len(w)):
#     w[i] *= 2
# Q.set_weights(w)

# w = Q.get_weights()
# for i in range(len(w)):
#     print("Norm Q weights layer", i, np.linalg.norm(w[i]))

# print("\nSave NN weights to file (in HDF5)")
# Q.save_weights("namefile.h5")

# print("Load NN weights from file\n")
# Q_target.load_weights("namefile.h5")

# w = Q_target.get_weights()
# for i in range(len(w)):
#     print("Norm Q weights layer", i, np.linalg.norm(w[i]))


# current_state=xy_to_t(env.reset()).reshape(1,-1)
# for step in range(MAX_EPISODE_LENGTH):
#     action = np.array([control_map[np.argmax(Q.predict(current_state))]])
#     next_state, r, done, _ = env.step(action)
#     current_state = xy_to_t(next_state).reshape(1,-1)
#     env.render()
#     sleep(0.01)