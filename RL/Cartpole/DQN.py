import numpy as np
import gym, math, time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

#Create the environment
env = gym.make("CartPole-v1")

#Reset the env to its initial state
state, _ = env.reset()

#define q learning parameters
alpha = 0.1 # Learning rate
gamma = 0.99 # Discount factor
epsilon = 1.0 # Exploration rate
epsilon_decay = 0.995 # decay rate for epsilon
min_epsilon = 0.01 # minimum value for epsilon
episodes = 1000 # number of training episodes
total = 0
total_reward = 0
prior_reward = 0
highest_reward = 0

# Build the Q-network (function approximator)
model = Sequential()
model.add(Dense(32, input_shape=(env.observation_space.shape[0],), activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
#Compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Use the model to predict Q-values and update
for episode in range(episodes):
    t0 = time.time()
    state = np.reshape(env.reset()[0], [1, env.observation_space.shape[0]])
    done = False
    episode_reward = 0

    if episode %50 == 0:
        print(f'Episode: {episode}')

    while not done:
        q_values = model.predict(state, verbose=0)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:  
            action = np.argmax(q_values[0]) #Choose best action

        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        
        if not done: #update model
            target = reward + gamma * np.amax(model.predict(next_state, verbose=0)[0])
            target_f = model.predict(state, verbose=0)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=20, verbose=0)            

        state = next_state


    if epsilon > min_epsilon: #epsilon modification
        if episode_reward > prior_reward:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)  # Decay epsilon
            # if episode > 10 and highest_reward < episode_reward:
            #     episode = episode-10
            #epsilon = math.pow(epsilon_decay, episode - 50000)

            # if episode % 50 == 0:
            #     print("Epsilon: " + str(epsilon))
    
    t1 = time.time() #episode has finished
    episode_total = t1 - t0 #episode total time
    total = total + episode_total

    total_reward += episode_reward #episode total reward
    prior_reward = episode_reward

    if episode % 50 == 0: #every 1000 episodes print the average time and the average reward
        mean = total / 50
        print("Time Average: " + str(mean))
        total = 0

        print(f'Epsilon: {epsilon}')

        mean_reward = total_reward / 50
        print("Mean Reward: " + str(mean_reward))
        total_reward = 0

        if mean_reward > highest_reward:
            highest_reward = mean_reward
            if episode > 50:
                 episode = episode-50

        print(f'Highest Mean Reward: {highest_reward}')

#Evaluate the trained model
evaluate = True
while evaluate == True:
    go = input('Enter Y to continue to model evaluation: ')

    if go.lower() != 'y':
        evaluate = False
        break

    t0 = time.time()
    state = np.reshape(env.reset()[0], [1, env.observation_space.shape[0]])
    done = False
    total_reward = 0
    while not done:
        q_values = model.predict(state, verbose=0)
        action = np.argmax(q_values[0]) #Choose best action
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        env.render()
        state = next_state
    t1 = time.time() #episode has finished
    episode_total = t1 - t0 #episode total time
    print(f'Survival time: {episode_total}\nReward: {total_reward}')
    env.close()