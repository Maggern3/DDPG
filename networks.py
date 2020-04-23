import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(2)
        self.fc1 = nn.Linear(state_size, 256)        
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 4)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(2)
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256+4, 256)
        self.fc3 = nn.Linear(256, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x, actions):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(torch.cat((x, actions), dim=1)))
        x = self.fc3(x)
        return x