import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import randint, uniform
import gym
from time import sleep

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
 
def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()


def get_critic(nx, nu):
    ''' Create the neural network to represent the Q function '''
    inputs = layers.Input(shape=(2))
    state_out1 = layers.Dense(16, activation="relu")(inputs) 
    state_out2 = layers.Dense(32, activation="relu")(state_out1) 
    state_out3 = layers.Dense(64, activation="relu")(state_out2) 
    state_out4 = layers.Dense(64, activation="relu")(state_out3)
    outputs = layers.Dense(2)(state_out4) 

    model = tf.keras.Model(inputs, outputs)

    return model

def update(xu_batch, reward_batch, action_batch, xu_next_batch):
    ''' Update the weights of the Q network using the specified batch of data '''
    # all inputs are tf tensors
    with tf.GradientTape() as tape:         
        # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
        # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. 
        # Tensors can be manually watched by invoking the watch method on this context manager.
        target_values = Q_target(xu_next_batch, training=True)
        # Compute 1-step targets for the critic loss
        y = reward_batch + DISCOUNT* np.max(target_values)                        
        # Compute batch of Values associated to the sampled batch of states
        Q_value = Q(xu_batch, training=True)[0][action_batch]                         
        # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
        Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))  
    # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
    Q_grad = tape.gradient(Q_loss, Q.trainable_variables)          
    # Update the critic backpropagating the gradients
    critic_optimizer.apply_gradients(zip(Q_grad, Q.trainable_variables))    


env = gym.make('Pendulum')
nx = 2
nu = 1
Q_episodes_history = []
Q_max_history = []
QVALUE_LEARNING_RATE = 1e-3
DISCOUNT = 0.99
BATCH_SIZE = 32
MAX_NUMBER_OF_EPISODES = 100
MAX_EPISODE_LENGTH = 80
MEMORY_BUFFER_LENGTH = 50_000
# STEP_BEFORE_TRAIN = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 10
MIN_BUFFER_TO_TRAIN = 100
EXPLORATION_PROBABILITY_DECAY = 0.001
MIN_EXPLORATION_PROBABILITY = 0.1

exploration_prob = 1
memory_buffer = []

control_map = [-1, 1]

def store_episode(mem, current_state, action, reward, next_state, done):
        #We use a dictionnary to store them
        mem.append({
            "current_state":current_state.reshape(1,-1),
            "action":action,
            "reward":reward,
            "next_state":next_state.reshape(1,-1),
            "done" :done
        })
        # If the size of memory buffer exceeds its maximum, we remove the oldest experience
        if len(mem) > MEMORY_BUFFER_LENGTH:
            mem.pop(0)

def xy_to_t(state):
    return np.array([np.arctan2(state[1], state[0]), state[2]])

def t_to_xy(state):
    return np.array([np.cos(state[0]), np.sin(state[0]), state[1]])

# Create critic and target NNs
Q = get_critic(nx, nu)
Q_target = get_critic(nx, nu)

Q.summary()

# Set initial weights of targets equal to those of the critic
Q_target.set_weights(Q.get_weights())

# Set optimizer specifying the learning rates
critic_optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)

step = 0
for episode in range (MAX_NUMBER_OF_EPISODES):
    print("------ Episode ", episode, "over ", MAX_NUMBER_OF_EPISODES, " ------")
    current_state = xy_to_t(env.reset())
    for k in range(MAX_EPISODE_LENGTH):
        print("Episode ", episode, "step: ", k)
        if np.random.uniform() < exploration_prob :
                # take random action
                u = np.array([np.random.choice(control_map)])
        else:
            u = np.array([control_map[np.argmax(Q.predict(current_state.reshape(1,-1)))]])
        
        next_state, reward, done, _ = env.step(u)
        next_state = xy_to_t(next_state)
        store_episode(memory_buffer, current_state, u, reward, next_state, done)
        current_state = next_state
        step += 1

        if (step > MIN_BUFFER_TO_TRAIN):

            np.random.shuffle(memory_buffer)
            batch_sample = memory_buffer[: BATCH_SIZE]
            xu_batch = [x["current_state"] for x in batch_sample]
            reward_batch = [x["reward"] for x in batch_sample]
            xu_next_batch = [x["next_state"] for x in batch_sample]
            action_batch = [x["action"] for x in batch_sample]

            if (step % TARGET_NETWORK_UPDATE_FREQUENCY == 0):
                Q_target.set_weights(Q.get_weights())

            for xu, reward, action, xu_next in zip(xu_batch, reward_batch, action_batch, xu_next_batch):
                update(xu, reward, action, xu_next)

        exploration_prob = np.exp(-EXPLORATION_PROBABILITY_DECAY * step)
        exploration_prob = max(exploration_prob, MIN_EXPLORATION_PROBABILITY)
        Q_max_history.append(np.max(Q.predict(current_state.reshape(1,-1))))
    Q_episodes_history.append(Q_max_history)


w = Q.get_weights()
for i in range(len(w)):
    print("Shape Q weights layer", i, w[i].shape)
    
for i in range(len(w)):
    print("Norm Q weights layer", i, np.linalg.norm(w[i]))
    
print("\nDouble the weights")
for i in range(len(w)):
    w[i] *= 2
Q.set_weights(w)

w = Q.get_weights()
for i in range(len(w)):
    print("Norm Q weights layer", i, np.linalg.norm(w[i]))

print("\nSave NN weights to file (in HDF5)")
Q.save_weights("namefile.h5")

print("Load NN weights from file\n")
Q_target.load_weights("namefile.h5")

w = Q_target.get_weights()
for i in range(len(w)):
    print("Norm Q weights layer", i, np.linalg.norm(w[i]))


current_state=xy_to_t(env.reset()).reshape(1,-1)
for step in range(MAX_EPISODE_LENGTH):
    action = np.array([control_map[np.argmax(Q.predict(current_state))]])
    next_state, r, done, _ = env.step(action)
    current_state = xy_to_t(next_state).reshape(1,-1)
    env.render()
    sleep(0.01)