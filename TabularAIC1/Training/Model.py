import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import torch.optim

from DQNUtils import *

df = pd.read_csv('Preprocessing/df_history.csv')
EPISODES = df.episode.unique()
STATES = df.state.unique()
ACTION_SPACE = df[~df['action'].isnull()]['action'].unique()

N_ITERATIONS = 1000
BATCH_SIZE = 128
GAMMA = 0.995

policy = offlineDQN(1, 25)
target = copy.deepcopy(policy)

# Main Q learning loop.
for i in range(N_ITERATIONS):

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

    batch = df.sample(n=BATCH_SIZE)

    predicted, targets = [], []

    for index, experience in batch.iterrows():
        state, action, reward, next_state, done \
            = (experience['state'], experience['action'], experience['reward'],
               experience['next_state'], experience['done'])

        predicted.append(policy.compute_predicted_qs(state))
        targets.append(target.compute_target_qs(reward, next_state, done, GAMMA))

    policy.back_prop(predicted, targets)