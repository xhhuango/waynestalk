import gymnasium as gym
import numpy as np


class QLearning:
    def __init__(self, env: gym.Env, alpha=0.5, gamma=1.0, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _random_argmax(self, q_s: np.ndarray) -> int:
        ties = np.flatnonzero(np.isclose(q_s, q_s.max()))
        return np.random.choice(ties)

    def _epsilon_greedy_action(self, Q: np.ndarray, s: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            return self._random_argmax(Q[s])

    def _create_pi_by_Q(self, Q: np.ndarray) -> np.ndarray:
        pi = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=float)
        for s in range(self.env.observation_space.n):
            A_start = self._random_argmax(Q[s])
            pi[s][A_start] = 1.0
        return pi

    def run_control(self, n_episode=5000) -> tuple[np.ndarray, np.ndarray]:
        Q = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=float)

        for i_episode in range(1, n_episode + 1):
            print(f"\rEpisode: {i_episode}/{n_episode}", end="", flush=True)

            s, _ = self.env.reset()

            terminated = False
            truncated = False
            while not (terminated or truncated):
                a = self._epsilon_greedy_action(Q, s)
                s_prime, r, terminated, truncated, _ = self.env.step(a)
                if terminated or truncated:
                    Q[s, a] = Q[s, a] + self.alpha * (r + self.gamma * 0 - Q[s, a])
                else:
                    Q[s, a] = Q[s, a] + self.alpha * (r + self.gamma * np.max(Q[s_prime]) - Q[s, a])
                s = s_prime

        pi = self._create_pi_by_Q(Q)
        print()
        return pi, Q
