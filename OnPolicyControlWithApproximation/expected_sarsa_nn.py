import numpy as np
import torch
from torch import nn, optim

from mlp_q_network import MLPQNetwork


class MLPExpectedSarsa:
    def __init__(
        self,
        observation_dim: int,
        n_actions: int,
        *,
        hidden_dim: int,
        lr=1e-4,
        grad_clip=5.0,
        gamma=0.99,
        epsilon=0.1,
    ):
        self.observation_dim = observation_dim
        self.n_actions = n_actions
        self.grad_clip = grad_clip
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_network = MLPQNetwork(observation_dim, n_actions, hidden_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    @torch.no_grad()
    def q_values(self, s: np.ndarray) -> np.ndarray:
        x = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        q = self.q_network(x).squeeze(0)
        return q.numpy()

    def act(self, s: np.ndarray, greedy: bool = False) -> tuple[int, np.ndarray]:
        q = self.q_values(s)
        if not greedy and np.random.rand() < self.epsilon:
            a = np.random.randint(self.n_actions)
        else:
            ties = np.flatnonzero(np.isclose(q, q.max()))
            a = np.random.choice(ties)
        p = np.ones(self.n_actions) * (self.epsilon / self.n_actions)
        greedy_a = np.argmax(q)
        p[greedy_a] += 1.0 - self.epsilon
        return a, p

    def update(self, s: np.ndarray, a: int, r: float, s_prime: np.ndarray, terminated: bool) -> float:
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        s_tp1 = torch.tensor(s_prime, dtype=torch.float32).unsqueeze(0)

        q_t_s = self.q_network(s_t).squeeze(0)
        q_t_sa = q_t_s[a]

        with torch.no_grad():
            if terminated:
                target = torch.tensor(r, dtype=torch.float32)
            else:
                q_tp1_s = self.q_network(s_tp1).squeeze(0)
                p = torch.ones(self.n_actions, dtype=torch.float32) * (self.epsilon / self.n_actions)
                a_start = torch.argmax(q_tp1_s).item()
                p[a_start] += 1.0 - self.epsilon

                expected_q_tp1 = torch.sum(q_tp1_s * p)
                target = r + self.gamma * expected_q_tp1

        loss = 0.5 * (target - q_t_sa) ** 2

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        self.optimizer.step()

        return float(loss.item())
