import gymnasium as gym
import numpy as np


class DynamicProgramming:
    def __init__(self, env: gym.Env, gamma=1, theta=1e-9):
        self.env = env
        self.gamma = gamma
        self.theta = theta

    def _policy_evaluation(self, pi: np.ndarray, V: np.ndarray) -> np.ndarray:
        while True:
            delta = 0
            for s in range(self.env.observation_space.n):
                v = 0
                for a, pi_as in enumerate(pi[s]):
                    q_sa = 0
                    for p, s_prime, r, terminated in self.env.unwrapped.P[s][a]:
                        q_sa += p * (r + self.gamma * V[s_prime] * (not terminated))
                    v += pi_as * q_sa
                delta = max(delta, np.abs(v - V[s]))
                V[s] = v

            if delta < self.theta:
                break

        return V

    def _policy_improvement(self, pi: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, bool]:
        policy_stable = True

        for s in range(self.env.observation_space.n):
            old_a = np.argmax(pi[s])
            pi[s] = self._greedify_policy(V, s)
            new_a = np.argmax(pi[s])

            if old_a != new_a:
                policy_stable = False

        return pi, policy_stable

    def _greedify_policy(self, V: np.ndarray, s: int) -> np.ndarray:
        q_s = np.zeros(self.env.action_space.n)
        for a in range(self.env.action_space.n):
            for p, s_prime, r, terminated in self.env.unwrapped.P[s][a]:
                q_s[a] += p * (r + self.gamma * V[s_prime] * (not terminated))

        new_a = np.argmax(q_s)
        pi_s = np.eye(self.env.action_space.n)[new_a]
        return pi_s

    def policy_iteration(self) -> tuple[np.ndarray, np.ndarray]:
        V = np.zeros(self.env.observation_space.n)
        pi = np.ones((self.env.observation_space.n, self.env.action_space.n)) / self.env.action_space.n
        policy_stable = False

        while not policy_stable:
            V = self._policy_evaluation(pi, V)
            pi, policy_stable = self._policy_improvement(pi, V)

        return pi, V

    def value_iteration(self) -> tuple[np.ndarray, np.ndarray]:
        V = np.zeros(self.env.observation_space.n)

        while True:
            delta = 0
            for s in range(self.env.observation_space.n):
                q_s = np.zeros(self.env.action_space.n)
                for a in range(self.env.action_space.n):
                    for p, s_prime, r, terminated in self.env.unwrapped.P[s][a]:
                        q_s[a] += p * (r + self.gamma * V[s_prime] * (not terminated))

                new_a = np.max(q_s)
                delta = max(delta, np.abs(new_a - V[s]))
                V[s] = new_a

            if delta < self.theta:
                break

        pi = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        for s in range(self.env.observation_space.n):
            pi[s] = self._greedify_policy(V, s)

        return pi, V
