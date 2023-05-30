import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, h1_neurons=128, h2_neurons=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            h1_neurons (int): Number of neurons in the first layer
            h2_neurons (int): Number of neurons in the second layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.fc1 = nn.Linear(state_size, h1_neurons)
        self.fc2 = nn.Linear(h1_neurons, h2_neurons)
        self.fc3 = nn.Linear(h2_neurons, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
