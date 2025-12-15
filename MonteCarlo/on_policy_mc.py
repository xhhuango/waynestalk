import gymnasium as gym
import numpy as np


class OnPolicyMonteCarlo:
    def __init__(self, env: gym.Env, epsilon=0.1, gamma=1.0):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma

    def _random_argmax(self, q_s: np.ndarray) -> int:
        ties = np.flatnonzero(np.isclose(q_s, q_s.max()))
        return np.random.choice(ties)

    def _update_policy_for_state(self, pi: np.ndarray, Q: np.ndarray, s: int):
        A = np.ones(self.env.action_space.n, dtype=float) * self.epsilon / self.env.action_space.n
        A_start = self._random_argmax(Q[s])
        A[A_start] += 1.0 - self.epsilon
        pi[s] = A

    def _update_policy_by_epsilon_greedy(self, pi: np.ndarray, Q: np.ndarray):
        for s in range(self.env.observation_space.n):
            A = np.ones(self.env.action_space.n, dtype=float) * self.epsilon / self.env.action_space.n
            A_star = np.argmax(Q[s])
            A[A_star] += 1.0 - self.epsilon
            pi[s] = A

    def _generate_episode(self, pi: np.ndarray) -> list[tuple[int, int, float]]:
        episode = []
        s, _ = self.env.reset()
        while True:
            a = np.random.choice(np.arange(self.env.action_space.n), p=pi[s])
            s_prime, r, terminated, truncated, _ = self.env.step(a)
            episode.append((s, a, r))

            if terminated or truncated:
                break
            s = s_prime
        return episode

    def run_control(self, n_episodes=5000) -> tuple[np.ndarray, np.ndarray]:
        Q = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=float)
        Returns_sum = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=float)
        Returns_count = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=int)

        pi = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=float)
        self._update_policy_by_epsilon_greedy(pi, Q)

        for i_episode in range(1, n_episodes + 1):
            print(f"\rEpisode: {i_episode}/{n_episodes}, generating a episode ...", end="", flush=True)

            episode = self._generate_episode(pi)
            print(f"\rEpisode: {i_episode}/{n_episodes}, episode length is {len(episode)})", end="", flush=True)

            sa_first_visit = {}
            for t, (s, a, _) in enumerate(episode):
                if (s, a) not in sa_first_visit:
                    sa_first_visit[(s, a)] = t

            G = 0.0
            for t in range(len(episode) - 1, -1, -1):
                s, a, r = episode[t]
                G = self.gamma * G + r

                if t == sa_first_visit[(s, a)]:
                    Returns_sum[s, a] += G
                    Returns_count[s, a] += 1
                    Q[s, a] = Returns_sum[s, a] / Returns_count[s, a]
                    self._update_policy_for_state(pi, Q, s)

        print()
        return pi, Q
