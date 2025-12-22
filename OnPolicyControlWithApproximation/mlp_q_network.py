import torch
from torch import nn


class MLPQNetwork(nn.Module):
    def __init__(self, observation_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.network(s)
