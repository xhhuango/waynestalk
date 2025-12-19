import gymnasium as gym
import numpy as np


class GradientMonteCarlo:
    def __init__(self, feature_fn, alpha=0.1, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.feature_fn = feature_fn
        self.W = np.zeros(feature_fn.d, dtype=float)

    def vhat(self, s: int) -> float:
        x = self.feature_fn(s)
        return float(self.W @ x)

    def update(self, states: list[int], rewards: list[float]) -> None:
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            s = states[t]
            x = self.feature_fn(s)
            v = self.vhat(s)
            self.W = self.W + self.alpha * (G - v) * x

    def run(self, env: gym.Env, policy, n_episodes=3000, max_steps=10_000) -> np.ndarray:
        start_values = []

        for i_episode in range(1, n_episodes + 1):
            print(f"\rEpisode: {i_episode}/{n_episodes}", end="", flush=True)

            s, _ = env.reset()
            terminated = False
            truncated = False
            steps = 0
            states = []
            rewards = []

            start_values.append(self.vhat(s))
            while not terminated and not truncated and steps < max_steps:
                states.append(s)
                a = policy(s)
                s_prime, r, terminated, truncated, _ = env.step(a)
                rewards.append(r)
                s = s_prime
                steps += 1

            self.update(states, rewards)

        print()
        return np.array(start_values)
