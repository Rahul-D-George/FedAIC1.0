import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DQNUtils import DQN

df = pd.read_csv('Preprocessing/df_history.csv')


# Instead of initialising a Q dictionary, we will use a DQN model to approximate the Q function, and
# to predict state-action values. Input: 750 (OHE States), Output: 25 (Actions).
model = DQN(750, 25)

n_episodes = len(df['icustayid'].unique())

# SKELETON

# while episodes:
#  while timesteps:
#    if not terminal:
#      create OHE state tensor
#      pass state tensor through model to get action tensor
#      see what action was actually performed by the model, and what the reward was
#      store state-action-rewar-next_state-done_flag tuple in memory
#      sample a batch from memory
#      compute target Q values for the batch
#      compute loss between predicted and target Q values using smooth L1 loss
#      update model parameters using the loss