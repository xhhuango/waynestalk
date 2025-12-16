import time

import gymnasium as gym
import numpy as np

from expected_sarsa import ExpectedSarsa
from q_learning import QLearning
from sarsa import Sarsa

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

        for i in range(10):
            print(f"Sleep {i}")
            time.sleep(1)

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

    print("Start Sarsa")
    sarsa = Sarsa(env)
    sarsa_policy, sarsa_Q = sarsa.run_control()
    play_game(sarsa_policy)
    print("\n")

    print("Start Q-learning")
    q_learning = QLearning(env)
    ql_policy, ql_Q = q_learning.run_control()
    play_game(ql_policy)
    print("\n")

    print("Start Expected Sarsa with epsilon = 0.1")
    esarsa_01 = ExpectedSarsa(env, epsilon=0.1)
    esarsa_01_policy, esarsa_01_Q = esarsa_01.run_control()
    play_game(esarsa_01_policy)
    print("\n")

    print("Start Expected Sarsa with epsilon = 0.001")
    esarsa_0001 = ExpectedSarsa(env, epsilon=0.001)
    esarsa_0001_policy, esarsa_0001_Q = esarsa_0001.run_control()
    play_game(esarsa_0001_policy)
    print("\n")

    env.close()
