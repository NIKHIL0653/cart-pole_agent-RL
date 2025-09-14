import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden1=128, hidden2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim)
        )

    def forward(self, x):
        return self.net(x)
