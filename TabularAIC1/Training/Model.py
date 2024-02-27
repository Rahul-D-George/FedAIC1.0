import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DQNUtils import DQN

df = pd.read_csv('Preprocessing/df_history.csv')
EPISODES = df.episode.unique()
STATES = df.state.unique()
ACTION_SPACE = df[~df['action'].isnull()]['action'].unique()
NON_TERMINAL_STATES = df[~df['action'].isnull()]['state'].unique()
TERMINAL_STATES = df[df['action'].isnull()]['state'].unique()

# Instead of initialising a Q dictionary, we will use a DQN model to approximate the Q function, and
# to predict state-action values. Input: 750 (OHE States), Output: 25 (Actions).
agent = DQNAgent(750, 25)

n_episodes = 1000 # Arbitrary for now.
reward_per_episode = []

for ep in range(len(n_episodes)):

    # Choose random episode to train on.
    ep = np.random.choice(EPISODES)

    # Set initial state and reward to 0.
    s = df[df['episode'] == ep]['state'].iloc[0]
    episode_reward = 0

    # Loop through the episode while time-state is not terminal.
    while s not in TERMINAL_STATES:

        # Create OHE state tensor.
        state_tensor = torch.zeros(750)
        state_tensor[s] = 1
        state = state.unsqueeze(0)

        # Pass state tensor through model to get action tensor.
        action_tensor = agent.select_action(state)
        dqn_action = action_tensor.item()  # This is your action

        # I DON'T KNOW HOW I SHOULD BE USING EITHER OF THE ABOVE

        # See what state was actually reached performed by the model, and what the reward was.
        reward = df[(df['episode'] == ep) & (df['state'] == s)]['reward'].iloc[0]
        next_state = df[(df['episode'] == ep)
                        & (df['timestep'] == df[(df['episode'] == ep)
                                                & (df['state'] == s)]['timestep'].iloc[0] + 1)]['state'].iloc[0]


        # Store state-action-reward-next_state-done_flag tuple in memory.
        agent.memory.push(state, dqn_action, reward, next_state, next_state in TERMINAL_STATES)

        # Sample a batch from memory, compute target Q values for the batch, and compute loss.
        agent.optimize_model()

        # Increment episode reward and set state to next state.
        s = next_state
        episode_reward += reward

    # Append episode reward to list of rewards.
    reward_per_episode.append(episode_reward)