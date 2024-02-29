import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DQNUtils import *

df = pd.read_csv('Preprocessing/df_history.csv')
EPISODES = df.episode.unique()
STATES = df.state.unique()
ACTION_SPACE = df[~df['action'].isnull()]['action'].unique()
NON_TERMINAL_STATES = df[~df['action'].isnull()]['state'].unique()
TERMINAL_STATES = df[df['action'].isnull()]['state'].unique()


# Instead of initialising a Q dictionary, we will use a DQN model to approximate the Q function, and
# to predict state-action values. Input: 750 (OHE States), Output: 25 (Actions).
agent = offlineDQN(750, 25)

N_ITERATIONS = 1000
BATCH_SIZE = 128

# Main Q learning loop.
for i in range(N_ITERATIONS):

    batch = df.sample(n=BATCH_SIZE)

    predicted, targets = [], []

    #for
