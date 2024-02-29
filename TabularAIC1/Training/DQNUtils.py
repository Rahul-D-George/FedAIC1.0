import torch.nn as nn
import torch.nn.functional as F
import torch.optim


# Define network used by policy and target
class offlineDQN(nn.Module):
    def __init__(self, input_dim, output_dim, layers=None, ADAM_LR=0.01, BETAS=(0.9, 0.999)):
        super().__init__()

        if layers is None:
            layers = [128]
        if len(layers) == 0:
            layers.append(input_dim)

        nn_layers = [nn.Linear(input_dim, layers[0])]

        for i in range(1, len(layers)):
            nn_layers.append(nn.Linear(layers[i-1], layers[i]))

        nn_layers.append(nn.Linear(layers[-1], output_dim))

        self.layers = nn.ModuleList(nn_layers)
        self.optim = torch.optim.Adam(self.parameters(), lr=ADAM_LR, betas=BETAS)

    # Forward Propagation
    def forward(self, state):
        x = state.float()
        for num, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if num < len(self.layers) - 1 else layer(x)
        return x