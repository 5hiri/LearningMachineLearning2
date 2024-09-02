import numpy as np
import gym
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random

# Create the environment
env = gym.make("CartPole-v1")

# Define Q-learning parameters
ALPHA = 0.001  # Learning rate
GAMMA = 0.99  # Discount factor
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.9995  # Decay rate for epsilon
MIN_EPSILON = 0.01  # Minimum value for epsilon
EPISODES = 1500  # Number of training episodes
BATCH_SIZE = 32  # Size of batches for training

# Create a replay memory
memory = deque(maxlen=2000)

# Build the Q-network (function approximator)
def build_model(state_size, action_size):
    model = Sequential([
        Dense(64, input_shape=(state_size,), activation='relu'),
        Dense(64, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=ALPHA))
    return model

# Epsilon-greedy action selection
def epsilon_greedy_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])

# Store experience in replay memory
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# Train the model with a batch from memory
def replay(batch_size):
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + GAMMA * np.amax(model.predict(next_state, verbose=0)[0])
        target_f = model.predict(state, verbose=0)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)

# Initialize the model
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = build_model(state_size, action_size)

model.load_weights('./rl/cartpole/dqn_model.h5')

highest_reward = 0

# Training loop
for episode in range(EPISODES):
    state = np.reshape(env.reset(), [1, state_size])
    done = False
    episode_reward = 0

    while not done:
        action = epsilon_greedy_action(state, EPSILON)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        remember(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

    if len(memory) > BATCH_SIZE:
        replay(BATCH_SIZE)

    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY

    # if episode % 50 == 0:
    print(f'Episode: {episode}, Reward: {episode_reward}, Epsilon: {EPSILON:.2f}')
    if episode_reward > highest_reward:
        highest_reward = episode_reward
        model.save_weights(f'./rl/cartpole/dqn_model_reward.h5')

model.save_weights('./rl/cartpole/dqn_model.h5')

# Evaluation loop
def evaluate_model(num_episodes=100):
    for _ in range(num_episodes):
        state = np.reshape(env.reset(), [1, state_size])
        done = False
        total_reward = 0
        while not done:
            action = np.argmax(model.predict(state, verbose=0)[0])
            next_state, reward, done, _ = env.step(action)
            state = np.reshape(next_state, [1, state_size])
            total_reward += reward
            env.render()
        print(f'Evaluation episode reward: {total_reward}')
    env.close()

# Uncomment to run evaluation
evaluate = True
while evaluate == True:
    go = input('Enter Y to continue to model evaluation: ')

    if go.lower() != 'y':
        evaluate = False
        break
    
    evaluate_model()
