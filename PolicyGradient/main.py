import time

import gymnasium as gym

from actor_critic import ActorCritic

GYM_ID = "LunarLander-v3"
N_EPISODES = 1000
MAX_STEPS = 1000


def play_game(agent: ActorCritic, episodes=1):
    visual_env = gym.make(GYM_ID, render_mode="human")  # UI window

    for episode in range(episodes):
        state, _ = visual_env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        step_count = 0

        print(f"Episode {episode + 1} starts")
        while not terminated and not truncated:
            action, _ = agent.act(state)
            state, reward, terminated, truncated, _ = visual_env.step(action)
            total_reward += reward
            step_count += 1
            visual_env.render()

        print(f"Episode {episode + 1} is finished: Total reward is {total_reward}, steps = {step_count}")
        time.sleep(1)

    visual_env.close()


def train() -> ActorCritic:
    env = gym.make(GYM_ID)
    agent = ActorCritic(
        observation_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_dim=256,
        actor_lr=3e-4,
    )

    returns = 0.0

    for i_episode in range(1, N_EPISODES + 1):
        print(f"\rEpisode: {i_episode}/{N_EPISODES}", end="", flush=True)

        s, _ = env.reset()
        done = False
        G = 0.0
        steps = 0

        while not done and steps < MAX_STEPS:
            a, pi_s = agent.act(s)
            s_prime, r, terminated, truncated, _ = env.step(a)
            G += r
            done = terminated or truncated
            agent.update(s, a, r, s_prime, pi_s, done)
            s = s_prime
            steps += 1

        returns += G
        if i_episode % 50 == 0:
            print(f"\nEpisode: {i_episode}/{N_EPISODES}, average return: {returns / 50:.2f}")
            returns = 0.0

    return agent


if __name__ == "__main__":
    agent = train()
    play_game(agent)
