import gymnasium as gym
import numpy as np


class SemiGradientTD0:
    def __init__(self, feature_fn, alpha=0.1, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.feature_fn = feature_fn
        self.W = np.zeros(feature_fn.d, dtype=float)

    def vhat(self, s: int) -> float:
        x = self.feature_fn(s)
        return float(self.W @ x)

    def update(self, s: int, r: float, s_prime: int, terminated: bool) -> None:
        v = self.vhat(s)
        v_prime = self.vhat(s_prime) if not terminated else 0.0
        delta = r + self.gamma * v_prime - v
        x = self.feature_fn(s)
        self.W = self.W + self.alpha * delta * x

    def run(self, env: gym.Env, policy, n_episodes=3000, max_steps=10_000) -> np.ndarray:
        start_values = []

        for i_episode in range(1, n_episodes + 1):
            print(f"\rEpisode: {i_episode}/{n_episodes}", end="", flush=True)

            s, _ = env.reset()
            terminated = False
            truncated = False
            steps = 0

            start_values.append(self.vhat(s))
            while not terminated and not truncated and steps < max_steps:
                a = policy(s)
                s_prime, r, terminated, truncated, _ = env.step(a)
                self.update(s, r, s_prime, terminated)
                s = s_prime
                steps += 1

        print()
        return np.array(start_values)
