import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('df_history.xlsx')

# Function to select the action with the highest Q value (randomly if there are multiple).
def dict_max(d):
    max_val = max(d.values())
    max_keys = [key for key, val in d.items() if val == max_val]
    return np.random.choice(max_keys), max_val

# Various Hyperparameters.
GAMMA = 0.99
ALPHA = 0.1
ACTION_SPACE = df[~df['action'].isnull()]['action'].unique()
EPISODES = df.episode.unique()
STATES = df.state.unique()
NON_TERMINAL_STATES = df[~df['action'].isnull()]['state'].unique()

# Initialising Q dictionary - this will be used to store the Q values for each state-action pair.
Q = {}
for s in STATES:
    Q[s] = {}
    for a in ACTION_SPACE:
        Q[s][a] = 0

# Initialising various other dictionaries and lists.
update_counts = {}
reward_per_episode = []
reward_mean_log = []
sum_max_q = []

# Main Q learning loop.
n_episodes = 10
for it in range(n_episodes):

    # We periodically print out the current iteration number, as well as the sum of the Q function.
    if it % 1 == 0:
        print("current iteration no:", it)
        iteration_max_q = []
        for st in STATES:
            iteration_max_q.append(sum(Q[st].values()))
        sum_max_q.append(sum(iteration_max_q))
        print("\n")
        print("current sum of q function:", sum_max_q[-1])

    # We randomly select an episode from the dataset.
    ep = np.random.choice(EPISODES)
    s = df[df['episode'] == ep]['state'].iloc[0]
    episode_reward = 0
    last_timestep = df[df['episode'] == ep]['timestep'].max()

    # We then iterate through the episode, updating the Q values as we go.
    index = 1
    step = 1

    # We use `step` to keep track of the current timestep, and `index` to keep track of the current index
    # in the episode. `last_timestep` is the last timestep in the episode.
    while step < last_timestep:

        a = df[(df['episode'] == ep)]['action'].iloc[index - 1]
        r = df[(df['episode'] == ep)]['reward'].iloc[index]
        s_next = df[df['episode'] == ep]['state'].iloc[index]
        episode_reward += r

        maxQ = dict_max(Q[s_next])[1]
        oldQ = (Q[s][a])
        Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * maxQ - Q[s][a])
        newQ = (Q[s][a])

        update_counts[s] = update_counts.get(s, 0) + 1

        s = s_next

        next_step_no = df[(df['episode'] == ep)]['timestep'].iloc[index]
        step_diff = next_step_no - step
        step += step_diff
        index += 1

    reward_per_episode.append(episode_reward)
    running_mean = round(np.mean(reward_per_episode[-100:]), 2)
    reward_mean_log.append(running_mean)

plt.plot(reward_mean_log)
plt.title("mean reward of last 100 episodes")
plt.xlabel("iterations")
plt.show()

plt.plot(sum_max_q)
plt.title("sum of q function")
plt.xlabel("iterations")
plt.show()

policy = {}
V = {}
for s in NON_TERMINAL_STATES:
    a, max_q = dict_max(Q[s])
    policy[s] = a
    V[s] = max_q

total = np.sum(list(update_counts.values()))
for k, v in update_counts.items():
    update_counts[k] = float(v) / total

dq = pd.DataFrame.from_dict(Q, orient='index')
dp = pd.DataFrame.from_dict(policy, orient='index')
dv = pd.DataFrame.from_dict(V, orient='index')
ds = pd.DataFrame.from_dict(update_counts, orient='index')
dd = pd.DataFrame(sum_max_q)
dr = pd.DataFrame(reward_per_episode)

dq.to_csv('dq_vals.csv')
dp.to_csv('dp_vals.csv')
dv.to_csv('dv_vals.csv')
ds.to_csv('ds_vals.csv')
dd.to_csv('dd_vals.csv')
dr.to_csv('dr_vals.csv')

with open('dict_exp.pkl', 'wb') as f:
    pickle.dump(Qd, f)
