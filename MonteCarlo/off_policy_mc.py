import gymnasium as gym
import numpy as np


class OffPolicyMonteCarlo:
    def __init__(self, env: gym.Env, gamma=1.0, b_epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.b_epsilon = b_epsilon

    def _random_argmax(self, q_s: np.ndarray) -> int:
        ties = np.flatnonzero(np.isclose(q_s, q_s.max()))
        return np.random.choice(ties)

    def _create_b(self, Q: np.ndarray, epsilon: float) -> np.ndarray:
        pi = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=float)
        for s in range(self.env.observation_space.n):
            A = np.ones(self.env.action_space.n, dtype=float) * epsilon / self.env.action_space.n
            A_start = self._random_argmax(Q[s])
            A[A_start] += 1.0 - epsilon
            pi[s] = A
        return pi

    def _generate_episode(self, policy: np.ndarray) -> np.ndarray:
        episode = []
        s, _ = self.env.reset()
        while True:
            a = np.random.choice(np.arange(self.env.action_space.n), p=policy[s])
            s_prime, r, terminated, truncated, _ = self.env.step(a)
            episode.append((s, a, r))

            if terminated or truncated:
                break
            s = s_prime
        return episode

    def _create_pi_by_Q(self, Q: np.ndarray) -> np.ndarray:
        pi = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=float)
        for s in range(self.env.observation_space.n):
            A_star = self._random_argmax(Q[s])
            pi[s][A_star] = 1.0
        return pi

    def run_control(self, n_episodes=1000) -> tuple[np.ndarray, np.ndarray]:
        Q = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=float)
        C = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=float)

        for i_episode in range(1, n_episodes + 1):
            print(f"\rEpisode: {i_episode}/{n_episodes}, generating a episode ...", end="", flush=True)

            b = self._create_b(Q, self.b_epsilon)
            episode = self._generate_episode(b)
            print(f"\rEpisode: {i_episode}/{n_episodes}, episode length is {len(episode)})", end="", flush=True)

            G = 0.0
            W = 1.0
            for t in range(len(episode) - 1, -1, -1):
                s, a, r = episode[t]
                G = self.gamma * G + r
                C[s, a] += W
                Q[s, a] += (W / C[s, a]) * (G - Q[s, a])
                if a not in np.flatnonzero(np.isclose(Q[s], Q[s].max())):
                    break
                W *= 1.0 / b[s, a]

        pi = self._create_pi_by_Q(Q)
        print()
        return pi, Q