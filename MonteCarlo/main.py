import time

import gymnasium as gym
import numpy as np

from off_policy_mc import OffPolicyMonteCarlo
from on_policy_mc import OnPolicyMonteCarlo

GYM_ID = "CliffWalking-v1"
# GYM_ID = "Taxi-v3"


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

    print(f"Gym environment: {GYM_ID}")

    print("Start On-Policy First-Visit Monte Carlo (for epsilon-soft policies)")
    on_policy_mc = OnPolicyMonteCarlo(env)
    on_policy_pi, on_policy_Q = on_policy_mc.run_control(n_episodes=5000)
    play_game(on_policy_pi)

    print("\n")

    print("Start Off-Policy Monte Carlo")
    off_policy_mc = OffPolicyMonteCarlo(env)
    off_policy_pi, off_policy_Q = off_policy_mc.run_control(n_episodes=5000)
    play_game(off_policy_pi)

    env.close()
