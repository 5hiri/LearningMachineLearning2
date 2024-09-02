import numpy as np
import gym, math, time

#Create the environment
env = gym.make("CartPole-v1")

#Reset the env to its initial state
state, _ = env.reset()

#Example of running one episode
# done = False
# while not done:
#     action = env.action_space.sample() #Take a random action
#     next_state, reward, done, info = env.step(action)
#     env.render()
# env.close()

#numer of actions and states
action_space = env.action_space.n
state_space = env.observation_space.shape[0]

Observation = [30, 30, 50, 50]
np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])

def get_discrete_state(state):
    discrete_state = state/np_array_win_size+ np.array([15,10,1,10])
    return tuple(discrete_state.astype(np.int64))

#initialize q-table with zeros
Q_table = np.random.uniform(low=0, high=1, size=(Observation + [action_space]))
print(Q_table.shape)
#define q learning parameters
alpha = 0.1 # Learning rate
gamma = 0.99 # Discount factor
epsilon = 1.0 # Exploration rate
epsilon_decay = 0.9995 # decay rate for epsilon
min_epsilon = 0.01 # minimum value for epsilon
episodes = 10000 # number of training episodes
total = 0
total_reward = 0
prior_reward = 0
highest_reward = 0

#Q-learning algorithm
for episode in range(episodes):
    t0 = time.time() #set the initial time
    discrete_state = get_discrete_state(env.reset()[0])
    done = False
    episode_reward = 0 #reward starts as 0 for each episode

    avg_step_time = 0
    step_count = 0

    if episode % 2000 == 0: 
        print("Episode: " + str(episode))
    while not done:
        step_count += 1
        t4 = time.time()
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() #Explore: select random action
        else:
            action = np.argmax(Q_table[discrete_state]) #Exploit: select the action with max Q-value
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward #add the reward
        new_discrete_state = get_discrete_state(next_state)
        if episode % 2000 == 0: #render
            env.render()

        if not done: #update q-table
            max_future_q = np.max(Q_table[new_discrete_state])

            current_q = Q_table[discrete_state + (action,)]

            new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)

            Q_table[discrete_state + (action,)] = new_q
        else:
            t6 = time.time()
            avg_step_time += t6 - t4
            print(f'Avg Step Time: {avg_step_time/step_count}')
        
        discrete_state = new_discrete_state
        t5 = time.time()
        avg_step_time += t5 - t4
    
    if epsilon > min_epsilon: #epsilon modification
        if episode_reward > prior_reward:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)  # Decay epsilon
            if episode > 2000 and highest_reward < episode_reward:
                episode = episode-2000
            #epsilon = math.pow(epsilon_decay, episode - 50000)

            if episode % 500 == 0:
                print("Epsilon: " + str(epsilon))
    
    t1 = time.time() #episode has finished
    episode_total = t1 - t0 #episode total time
    total = total + episode_total

    total_reward += episode_reward #episode total reward
    prior_reward = episode_reward

    if episode % 1000 == 0: #every 1000 episodes print the average time and the average reward
        mean = total / 1000
        print("Time Average: " + str(mean))
        total = 0

        mean_reward = total_reward / 1000
        print("Mean Reward: " + str(mean_reward))
        total_reward = 0

        if mean_reward > highest_reward:
            highest_reward = mean_reward

        print(f'Highest Mean Reward: {highest_reward}')

#Evaluate the trained model
state, _ = env.reset()
done = False
discrete_state = get_discrete_state(env.reset()[0])
t0 = time.time() #set the initial time
total_reward = 0
while not done:
    action = np.argmax(Q_table[discrete_state])
    next_state, reward, done, _, _ = env.step(action)
    total_reward = total_reward+reward
    env.render()
    discrete_state = get_discrete_state(next_state)
t1 = time.time() #episode has finished
episode_total = t1 - t0 #episode total time
print(f'Survival time: {episode_total}\nReward: {total_reward}')
env.close()