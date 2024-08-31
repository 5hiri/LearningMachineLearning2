import numpy as np
import gym, time

env = gym.make("LunarLander-v2")

state = env.reset()

# done = False
# while not done:
#     action = env.action_space.sample()
#     next_observation, reward, done, trunicated = env.step(action)
#     env.render()
# env.close()

action_space = env.action_space.n
state_space = env.observation_space.shape[0]


# Fine Tuning the Q_table

# episodes = 5000
# np_array_win_size = np.array([1.01882666, 1.10743685, 2.46568954, 1.35596186, 4.91044807, 7.98997498, 0.5, 0.5])

# state_min = np.inf * np.ones(8)
# state_max = -np.inf * np.ones(8)

# for _ in range(episodes):
#     state = env.reset()
#     done = False
#     while not done:
#         action = env.action_space.sample()
#         next_state, _, done, _ = env.step(action)
#         state_min = np.minimum(state_min, next_state)
#         state_max = np.maximum(state_max, next_state)
#         state = next_state

# normalized_min = state_min / np_array_win_size
# normalized_max = state_max / np_array_win_size


# print("State space ranges:")
# for i in range(8):
#     print(f"State {i}: [{state_min[i]:.2f}, {state_max[i]:.2f}]")

# print("Normalized state ranges:")
# for i in range(8):
#     print(f"State {i}: [{normalized_min[i]:.2f}, {normalized_max[i]:.2f}]")

# shift_values = -normalized_min
# scale_factor = 2 / (normalized_max - normalized_min)

# print("\nSuggested np_array_win_size:")
# suggested_win_size = (state_max - state_min) / 2
# print(suggested_win_size)

# print("\nSuggested observation (number of bins):")
# suggested_bins = np.ceil((state_max - state_min) / suggested_win_size).astype(int)
# print(suggested_bins)

# print("\nSuggested shift values:")
# print(shift_values)

# print("\nSuggested scale factors:")
# print(scale_factor)


# After fine tuning

observation = [23, 23, 28, 28, 16, 16, 4, 4]
np_array_win_size = np.array([1.01882666, 1.10743685, 2.46568954, 1.35596186, 4.91044807, 7.98997498, 0.5, 0.5])

shift_values = np.array([1.00086287, 0.3788708, 0.99582072, 1.60049672, 0.83125564, 1.11562138, 0, 0])
scale_factor = np.array([0.99873127, 0.96432542, 1.00194545, 0.9720442, 1.08859767, 0.99440361, 1, 1])

def get_discrete_state(state):
    normalized_shifted = state/np_array_win_size + shift_values
    scaled = normalized_shifted * scale_factor
    return tuple(np.clip(scaled, 0, 1).astype(np.int64))
    #return tuple((state/np_array_win_size + np.array([20, 20, 20, 20, 2, 2, 2, 2])).astype(np.int64))

#initialize q-table with zeros
Q_table = np.random.uniform(low=0, high=0, size=(observation + [action_space]))

#define q learning parameters
gamma = 0.99 # Discount factor
epsilon = 1.0 # Exploration rate
epsilon_decay = 0.9995 # decay rate for epsilon
min_epsilon = 0.01 # minimum value for epsilon
episodes = 500000 # number of training episodes
total = 0
total_reward = 0
prior_reward = 0
highest_reward = 0
initial_alpha = 0.8  # Initial learning rate
min_alpha = 0.001     # Minimum learning rate
alpha_decay = 0.9995   # Decay factor for learning rate
alpha = initial_alpha
all_times = []

#Q-learning algorithm
for episode in range(episodes):
    t0 = time.time()
    state = get_discrete_state(env.reset())
    done = False
    episode_reward = 0

    if episode % 2000 == 0:
        print(f'Episode: {episode}')
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() #Explore: select random action
        else:
            action = np.argmax(Q_table[state]) #Exploit: select the action with max Q-value
        
        next_state, reward, done, _ = env.step(action)
        next_state = get_discrete_state(next_state)
        episode_reward += reward

        if episode % 2000 == 0: #render
            env.render()
        
        if not done: #update q-table
            max_future_q = np.max(Q_table[state])
            current_q = Q_table[state + (action,)]
            new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
            Q_table[state + (action,)] = new_q
        
        state = next_state
    
    if epsilon > min_epsilon: #epsilon modification
        if episode_reward > prior_reward:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

            if episode % 2000 == 0:
                print(f'Epsilon: {epsilon}')

    if highest_reward < episode_reward:
        if episode%2 != 0:
            episode -= 1
        episode = episode/2

    # Calculate the current learning rate
    if alpha > min_alpha:
        alpha = max(min_alpha, alpha * alpha_decay)
        if episode % 2000 == 0:
                print(f'Alpha: {alpha}')
    
    t1 = time.time()
    episode_total = t1 - t0
    total = total + episode_total

    total_reward += episode_reward
    prior_reward = episode_reward

    if episode % 1000 == 0:
        mean = total / 1000
        print("Time Average: " + str(mean))
        total = 0

        mean_reward = total_reward / 1000
        print("Mean Reward: " + str(mean_reward))
        total_reward = 0

        if mean_reward > highest_reward:
            highest_reward = mean_reward

        print(f'Highest Mean Reward: {highest_reward}')
    
    all_times.append(episode_reward)

#Evaluate the trained model
state = env.reset()
done = False
state = get_discrete_state(env.reset())
t0 = time.time()
total_reward = 0
while not done:
    action = np.argmax(Q_table[state])
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()
    state = get_discrete_state(next_state)
t1 = time.time()
episode_total = t1 - t0
print(f'Survival time: {episode_total}\nReward: {total_reward}')
env.close()