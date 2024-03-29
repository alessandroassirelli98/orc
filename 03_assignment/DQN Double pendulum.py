import os
import time
import tensorflow as tf
from tensorflow.keras import layers
from keras.callbacks import TensorBoard
from ModifiedTensorboard import ModifiedTensorBoard
import numpy as np
import random
from ddoublependulum import DDoublePendulum
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('seaborn')

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
 
def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()

QVALUE_LEARNING_RATE = 1e-3
DISCOUNT = 0.99
BATCH_SIZE = 128
MEMORY_BUFFER_LENGTH = 100_000
MIN_BUFFER_TO_TRAIN = 10000
EXPLORATION_PROBABILITY_DECAY = 0.00002
MIN_EXPLORATION_PROBABILITY = 0
TARGET_NETWORK_UPDATE_FREQUENCY = 10

class DQNAgent():
    
    def __init__(self):
        # Main model
        self.model = self.get_critic()

        # Target network
        self.target_model = self.get_critic()
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
        inputs = layers.Input(shape=(nx,))

        state_out1 = layers.Dense(64, activation="relu")(inputs) 
        state_out2 = layers.Dense(128, activation="relu")(state_out1)
        state_out3 = layers.Dense(64, activation="relu")(state_out2) 
        
        outputs = layers.Dense(ndu)(state_out3) 

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

uMax = 2
ndu = 201
dt = 0.02
env = DDoublePendulum(ndu=ndu, uMax=uMax, vMax1=10, vMax2 = 25, dt=dt)
nx = env.nx
nu = env.nu

SHOW_PREVIEW = False
AGGREGATE_STATS_EVERY = 10
MAX_NUMBER_OF_EPISODES = 2000
STEP_BEFORE_TRAIN = 4


MODEL_NAME = "DQNDoublePendulum_{}_ndu_{}_umax_{}_lr_{}_epsdec_{}".\
            format("64_128_64",ndu, uMax, QVALUE_LEARNING_RATE, EXPLORATION_PROBABILITY_DECAY)

ep_costs = []
avg_cost = []

agent = DQNAgent()
step = 1
start_time = time.time()
for episode in range (1, MAX_NUMBER_OF_EPISODES):
    agent.tensorboard.step = episode
    print("Episode ", episode )

    episode_cost = 0
    current_state = env.reset(random=False)
    done = False
    jj = 0
    while not done:
        iu = agent.compute_action()
        next_state, cost, done = env.step(iu)

        episode_cost += DISCOUNT**jj * cost

        agent.store_episode(current_state, iu, cost, next_state, done)
        agent.update_probability(step)

        if not jj % STEP_BEFORE_TRAIN or done:
            agent.train(done)
        
        current_state = next_state
        step += 1
        jj += 1

    # Append episode cost to a list and log stats (every given number of episodes)
    if len(ep_costs) == 0 or episode_cost < min(ep_costs):
        agent.model.save_weights(MODEL_NAME + "best_min.h5")
    ep_costs.append(episode_cost)

    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_cost = (sum(ep_costs[-AGGREGATE_STATS_EVERY:])/len(ep_costs[-AGGREGATE_STATS_EVERY:]))
        min_cost = min(ep_costs[-AGGREGATE_STATS_EVERY:])
        max_cost = max(ep_costs[-AGGREGATE_STATS_EVERY:])
        var_cost = np.var(ep_costs[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(cost_avg=average_cost,
                                        cost_min=min_cost, 
                                        cost_max=max_cost, 
                                        var_cost=var_cost,
                                        epsilon=agent.exploration_probability, 
                                        loss=agent.loss_value)
        if len(avg_cost) == 0 or average_cost < min(avg_cost): # This makes sense if the state is initialized randomly
            agent.model.save_weights(MODEL_NAME + "best_avg.h5")
        avg_cost.append(average_cost)

total_time = time.time()-start_time
print("Total training time: ", total_time)


a = []
s = []
current_state = env.reset().reshape(1,-1)
done = False
while not done:
    s.append(current_state[0])
    action = np.array([np.argmin(agent.model(current_state))])
    a.append(action)
    next_state, cost, done = env.step(action)
    current_state = next_state.reshape(1,-1)
    env.render()
    if done: break

time = np.arange(0, 200,1) * dt
s = np.array(s)

plt.figure()
plt.plot(time, s[:,0])
plt.plot(time, s[:,1])
plt.title("Position", fontsize=18)
plt.legend(["theta_1", "theta_2"], fontsize=18)
plt.xlabel("time [s]", fontsize=18)
plt.ylabel("angle [rad]", fontsize=18)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.savefig("Double positions.png")
plt.draw()

plt.figure()
plt.plot(time, s[:,2])
plt.plot(time, s[:,3])
plt.title("Velocities", fontsize=18)
plt.legend(["d_theta_1", "d_theta_2"], fontsize=18)
plt.xlabel("time [s]", fontsize=18)
plt.ylabel("angular velocity [rad/s]", fontsize=18)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.savefig("Double velocities.png")
plt.draw()


plt.figure()
plt.plot(env.control_map[a])
plt.title("Torque", fontsize=18)
plt.legend(["torque"], fontsize=18)
plt.xlabel("time [s]", fontsize=18)
plt.ylabel("Torque [Nm]", fontsize=18)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.savefig("Double torques.png")
plt.show()

