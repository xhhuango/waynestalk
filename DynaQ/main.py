import time

import gymnasium as gym
import numpy as np

from dyna_q_plus import DynaQPlus
from non_stationary_cliff_walking import NonStationaryCliffWalking

GYM_ID = "CliffWalking-v1"


def play_game(policy, episodes=1):
    base_visual_env = gym.make(GYM_ID, render_mode="human")
    visual_env = NonStationaryCliffWalking(base_visual_env)
    visual_env.switch()

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
    print(f"Gym environment: {GYM_ID}")

    print("Start Dyna-Q+ (kappa=0.05)")
    env = NonStationaryCliffWalking(gym.make(GYM_ID))
    dynaq_plus = DynaQPlus(env, kappa=0.05)  # Dyna-Q (kappa = 0)
    dynaq_plus.run()
    dynaq_plus_policy = dynaq_plus.create_pi_by_Q()
    play_game(dynaq_plus_policy)
    env.close()
    print("\n")
