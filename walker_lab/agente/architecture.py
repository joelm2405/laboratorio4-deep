import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    # misma forma que el actor de PPO (MlpPolicy, pi=[256, 256], Tanh)
    def __init__(self, obs_dim=24, act_dim=4, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.clamp(self.net(x), -1.0, 1.0)
