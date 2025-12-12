import time

import gymnasium as gym
import numpy as np

from dp import DynamicProgramming

# GYM_ID = "Taxi-v3"
GYM_ID = "CliffWalking-v1"


def play_game(policy, episodes=1):
    visual_env = gym.make(GYM_ID, render_mode="human")

    for episode in range(episodes):
        state, _ = visual_env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0

        print(f"Episode {episode + 1} starts")
        while not terminated and not truncated:
            action = np.argmax(policy[state])
            state, reward, terminated, truncated, _ = visual_env.step(action)
            total_reward += reward
            step_count += 1
            time.sleep(0.3)

        print(f"Episode {episode + 1} is finished: Total reward is {total_reward}, steps = {step_count}")
        time.sleep(1)

    visual_env.close()


if __name__ == "__main__":
    env = gym.make(GYM_ID)
    dp = DynamicProgramming(env)

    print(f"Gym environment: {GYM_ID}")

    print("Start DP Policy Iteration")
    pi_policy, pi_V = dp.policy_iteration()
    play_game(pi_policy)

    print("\n")

    print("Start DP Value Iteration")
    vi_policy, vi_V = dp.value_iteration()
    play_game(vi_policy)

    env.close()
