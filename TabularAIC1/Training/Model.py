import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from DQNUtils import *

df = pd.read_csv('../Preprocessing/df_final.csv')
EPISODES = df.episode.unique()
STATES = df.state.unique()
ACTION_SPACE = df[~df['action'].isnull()]['action'].unique()

N_ITERATIONS = 1000
BATCH_SIZE = 1
GAMMA = 0.995

policy = offlineDQN(1, 25, layers=[])
target = copy.deepcopy(policy)

optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

# Main Q learning loop.
for i in range(N_ITERATIONS):

    optimizer.zero_grad()

    batch = df.sample(BATCH_SIZE)

    predicted, targets = [], []

    for index, experience in batch.iterrows():
        state, action, reward, next_state, done \
            = (experience['state'], experience['action'], experience['reward'],
               experience['next_state'], experience['done'])

        action_tensor = policy.compute_predicted_qs(state)
        rel_action = action_tensor[int(action)-1]
        predicted.append(rel_action)

        targets.append(target.compute_target_qs(reward, next_state, done, GAMMA))

    predicted_tensor = torch.tensor(predicted, dtype=torch.float32, requires_grad=True)
    targets_tensor = torch.tensor(targets, dtype=torch.float32, requires_grad=True)

    policy.back_prop(predicted_tensor, targets_tensor)

    optimizer.step()