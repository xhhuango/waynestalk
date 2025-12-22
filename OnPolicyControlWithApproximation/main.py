import time

import gymnasium as gym

from expected_sarsa_nn import MLPExpectedSarsa
from sarsa_nn import MLPSarsa

GYM_ID = "LunarLander-v3"

N_EPISODES = 1000
MAX_STEPS = 10_000

ALGO = "Sarsa"
# ALGO = "Expected Sarsa"


def play_game(agent: MLPExpectedSarsa, episodes=1):
    visual_env = gym.make(GYM_ID, render_mode="human")

    for episode in range(episodes):
        state, _ = visual_env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        step_count = 0

        input("Press Enter to play game")

        print(f"Episode {episode + 1} starts")
        while not terminated and not truncated:
            action, _ = agent.act(state, greedy=True)
            state, reward, terminated, truncated, _ = visual_env.step(action)
            total_reward += reward
            step_count += 1
            visual_env.render()

        print(f"Episode {episode + 1} is finished: Total reward is {total_reward}, steps = {step_count}")
        time.sleep(1)

    visual_env.close()


def train(env: gym.Env, agent: MLPExpectedSarsa):
    for i_episode in range(1, N_EPISODES + 1):
        print(f"\rEpisode: {i_episode}/{N_EPISODES}", end="", flush=True)

        s, _ = env.reset()
        done = False
        G = 0.0
        steps = 0

        a, _ = agent.act(s)

        while not done and steps < MAX_STEPS:
            s_prime, r, terminated, truncated, _ = env.step(a)
            G += r
            done = terminated or truncated
            if done:
                a_prime = 0
            else:
                a_prime, _ = agent.act(s_prime)

            agent.update(s, a, r, s_prime, done)
            s, a = s_prime, a_prime
            steps += 1


if __name__ == "__main__":
    env = gym.make(GYM_ID)
    if ALGO == "Sarsa":
        agent = MLPSarsa(
            observation_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            hidden_dim=256,
        )
    else:
        agent = MLPExpectedSarsa(
            observation_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            hidden_dim=256,
        )
    train(env, agent)
    env.close()
    play_game(agent)
