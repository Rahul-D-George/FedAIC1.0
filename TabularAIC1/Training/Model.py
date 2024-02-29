import pandas as pd
import datetime

from DQNUtils import *
from Tuner import values_to_tune


def to_one_hot(num):

    one_hot = torch.zeros(754)

    intn = int(num)

    if intn == 751:
        one_hot[750] = 1
    elif intn == 752:
        one_hot[751] = 1
    elif intn == 777:
        one_hot[752] = 1
    elif intn < 751:
        one_hot[intn] = 1
    else:
        one_hot[753] = 1

    return one_hot


df = pd.read_csv('../Preprocessing/df_final.csv')

N_ITERATIONS, BATCH_SIZE, GAMMA, LAYERS, ADAM_LR, TARGET_FREQ = values_to_tune()

policy = offlineDQN(754, 25, layers=LAYERS)
target = offlineDQN(754, 25, layers=LAYERS)
target.load_state_dict(policy.state_dict())

batch = df.sample(BATCH_SIZE)

# Main Q learning loop.
for i in range(N_ITERATIONS):

    if i % TARGET_FREQ == 0:
        target.load_state_dict(policy.state_dict())

    states = torch.stack(batch['state'].apply(to_one_hot).tolist())
    next_states = torch.stack(batch['next_state'].apply(to_one_hot).tolist())
    actions = torch.tensor(batch['action'].values, dtype=torch.long) - 1
    rewards = torch.tensor(batch['reward'].values, dtype=torch.float)
    dones = torch.tensor(batch['done'].values, dtype=torch.float)

    predicted_qs = policy(states)
    next_state_qs = target(next_states).detach()

    predicted_qs = predicted_qs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    targets = rewards + GAMMA * next_state_qs.max(1)[0] * (1 - dones)

    policy.optim.zero_grad()
    loss = F.mse_loss(predicted_qs, targets)
    loss.backward()
    policy.optim.step()

    if (i + 1) % 100 == 0:
        print(f'Iteration {i + 1}')
        print(f"Loss = {loss}")

torch.save(policy.state_dict(), f"policy_{datetime.datetime.now().strftime('%Y%m%d-%H')}")
