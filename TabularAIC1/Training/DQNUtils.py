import torch.nn as nn
import torch.nn.functional as F
import torch.optim


BASE_LAYERS = [128]


# Define network used by policy and target
class offlineDQN(nn.Module):
    def __init__(self, input_dim, output_dim, layers=BASE_LAYERS):
        super().__init__()

        if len(layers) == 0:
            layers.append(output_dim)
            layers.append(input_dim)

        nn_layers = [nn.Linear(input_dim, layers[0])]


        for i in range(1, len(layers)):
            in_features = layers[i-1]
            out_features = layers[i]
            nn_layers.append(nn.Linear(in_features, out_features))

        nn_layers.append(nn.Linear(layers[-1], output_dim))

        self.layers = nn.ModuleList(nn_layers)

    def get_parameters(self):
        return self.layers

    def set_parameters(self, parameters):
        self.layers = parameters


    # Forward Propagation
    def forward(self, state_num):

        x = torch.tensor([state_num], dtype=torch.float32)

        for num, layer in enumerate(self.layers):
            if num == len(self.layers) - 1:
                x = layer(x)
                break
            x = F.relu(layer(x))

        print(x)

        return x

    # Backward Propagation
    def back_prop(self, target_qs, predicted_qs):
        loss = F.mse_loss(target_qs, predicted_qs)
        loss.backward()

    # Compute target Qs
    def compute_target_qs(self, reward, next_state, done, gamma):
        if done:
            return reward
        else:
            return reward + gamma * torch.max(self.forward(next_state))

    # Compute predicted Qs
    def compute_predicted_qs(self, state):
        return self.forward(state)
