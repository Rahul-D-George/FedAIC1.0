import torch.nn as nn
import torch.nn.functional as F

BASE_OPTIM = torch.optim.Adam(self.parameters(), lr=0.01)
BASE_LAYERS = [128]


# Define network used by policy and target
class offlineDQN(nn.Module):
    def __init__(self, input_dim, output_dim, layers=BASE_LAYERS, optim=BASE_OPTIM):
        super(DQN, self).__init__()

        nn_layers = []

        layers.append(nn.Linear(input_dim, layers[0]))

        for i in range(1, len(layers)):
            in_features = layer_sizes[i-1]
            out_features = layer_sizes[i]
            nn_layers.append(nn.Linear(in_features, out_features))

        layers.append(nn.Linear(layers[-1], output_dim))

        self.layers = nn.ModuleList(layers)
        self.optimizer = optim

    # Forward Propagation
    def forward(self, state_vector):

        x = state_vector

        for num, layer in enumerate(self.layers):
            if num == len(self.layers) - 1:
                x = layer(x)
                break
            x = F.relu(layer(x))

        return x

    # Backward Propagation
    def backward(self, target_qs, predicted_qs):

        loss = F.mse_loss(target_qs, predicted_qs)
        loss.backward()
        self.optimizer.step()

    # Compute target Qs
    def compute_target_qs(self, reward, next_state, done, gamma):
        if done:
            return reward
        else:
            return reward + gamma * torch.max(self.forward(next_state))

    # Compute predicted Qs
    def compute_predicted_qs(self, state):
        return self.forward(state)

    # Optimize
    def optimize(self, target_qs, predicted_qs):
        self.backward(target_qs, predicted_qs)
