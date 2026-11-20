import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Red neuronal de la política (Actor) para control continuo en BipedalWalker."""
    def __init__(self, obs_dim=24, act_dim=4, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return torch.clamp(out, -1.0, 1.0)
