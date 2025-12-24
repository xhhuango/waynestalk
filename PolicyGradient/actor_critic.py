import numpy as np
import torch
import torch.nn as nn
from torch import optim


class Actor(nn.Module):
    def __init__(self, observation_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor):
        logits = self.network(obs)
        return torch.distributions.Categorical(logits=logits)


class Critic(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor):
        return self.critic(obs).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        n_actions: int,
        *,
        hidden_dim: int,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        grad_clip=2.0,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.actor = Actor(observation_dim, n_actions, hidden_dim)
        self.critic = Critic(observation_dim, hidden_dim)
        self.grad_clip = grad_clip
        self.gamma = gamma
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def act(self, s: np.ndarray):
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        pi_s = self.actor(s_t)
        a = pi_s.sample()
        return int(a.item()), pi_s

    def update(
        self, s: np.ndarray, a: int, r: float, s_prime: np.ndarray, pi_s: torch.distributions.Categorical, terminated: bool
    ) -> tuple[float, float]:
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        a_t = torch.tensor([a], dtype=torch.long).unsqueeze(0)
        s_tp1 = torch.tensor(s_prime, dtype=torch.float32).unsqueeze(0)
        v_t = self.critic(s_t)

        with torch.no_grad():
            if terminated:
                v_prime = torch.tensor(0.0)
            else:
                v_prime = self.critic(s_tp1)
            td_target = r + self.gamma * v_prime

        td_error = td_target - v_t

        critic_loss = 0.5 * td_error.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        actor_loss = -(td_error.detach() * pi_s.log_prob(a_t))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        return float(actor_loss.item()), float(critic_loss.item())
