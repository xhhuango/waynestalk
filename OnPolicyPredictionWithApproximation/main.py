import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from gradient_monte_carlo import GradientMonteCarlo
from semi_gradient_td0 import SemiGradientTD0
from state_aggregation_features import StateAggregationFeatures

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3


def epsilon_soft_policy(s: int, epsilon: float) -> int:
    if np.random.random() < epsilon:
        return np.random.randint(4)
    else:
        row, col = divmod(s, 12)
        if row == 3 and col < 11:
            return UP
        if col < 11:
            return RIGHT
        if row < 3:
            return DOWN
        return RIGHT


if __name__ == "__main__":
    env = gym.make("CliffWalking-v1")
    features = StateAggregationFeatures(n_rows=4, n_cols=12, col_bin=2)
    policy = lambda s: epsilon_soft_policy(s, epsilon=0.1)

    print("Start Gradient Monte Carlo")
    gradient_monte_carlo = GradientMonteCarlo(features)
    values1 = gradient_monte_carlo.run(env, policy, n_episodes=10000, max_steps=100)

    print("Start Semi-Gradient TD(0)")
    semi_gradient_td0 = SemiGradientTD0(features)
    values2 = semi_gradient_td0.run(env, policy, n_episodes=10000, max_steps=100)

    plt.plot(values1, label="Gradient Monte Carlo", alpha=0.7)
    plt.plot(values2, label="Semi-Gradient TD(0)", alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Estimated V(start)")
    plt.legend()
    plt.title("On-policy Prediction on CliffWalking")
    plt.show()
