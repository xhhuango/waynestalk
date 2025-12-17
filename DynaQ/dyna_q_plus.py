import numpy as np

from non_stationary_cliff_walking import NonStationaryCliffWalking


class DynaQPlus:
    def __init__(
        self,
        env: NonStationaryCliffWalking,
        alpha=0.1,
        gamma=1,
        epsilon=0.01,
        kappa=0.5,
        planning_steps=10,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.kappa = kappa
        self.planning_steps = planning_steps

        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=float)
        self.model = {}
        self.t = 0
        self.last_time = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=int)

    def _init_sa_if_not_exists(self, s: int) -> None:
        for a in range(self.env.action_space.n):
            if (s, a) not in self.model:
                self.model[(s, a)] = (0.0, s, False)

    def _random_argmax(self, q_s: np.ndarray) -> int:
        ties = np.flatnonzero(np.isclose(q_s, q_s.max()))
        return np.random.choice(ties)

    def _act(self, s: int):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            return self._random_argmax(self.Q[s])

    def _q_learning_update(self, s: int, a: int, r: float, s_prime: int, terminated: bool):
        """Direct RL"""
        if terminated:
            self.Q[s][a] += self.alpha * (r + self.gamma * 0 - self.Q[s][a])
        else:
            self.Q[s][a] += self.alpha * (r + self.gamma * np.max(self.Q[s_prime]) - self.Q[s][a])

    def _model_update(self, s: int, a: int, r: float, s_prime: int, terminated: bool):
        """Model Learning"""
        self.model[(s, a)] = (r, s_prime, terminated)

    def _plan(self) -> None:
        seen_sa = list(self.model.keys())
        if not seen_sa:
            return

        for _ in range(self.planning_steps):
            s, a = seen_sa[np.random.choice(len(seen_sa))]
            r, s_prime, terminated = self.model[(s, a)]

            if self.kappa > 0.0:
                tau = self.t - self.last_time[s, a]
                bonus = self.kappa * np.sqrt(tau)
            else:
                bonus = 0

            self._q_learning_update(s, a, r + bonus, s_prime, terminated)

    def run(self, n_episodes=5000, max_steps=1_000) -> None:
        for i_episode in range(1, n_episodes + 1):
            print(f"\rEpisode: {i_episode}/{n_episodes}", end="", flush=True)

            s, _ = self.env.reset()
            self._init_sa_if_not_exists(s)
            for i in range(max_steps):
                print(f"\rEpisode: {i_episode}/{n_episodes}, steps={i}", end="", flush=True)
                a = self._act(s)
                s_prime, r, terminated, truncated, _ = self.env.step(a)
                self._init_sa_if_not_exists(s_prime)
                done = terminated or truncated
                self._q_learning_update(s, a, r, s_prime, done)
                self._model_update(s, a, r, s_prime, done)
                self.t += 1
                self.last_time[s, a] = self.t
                self._plan()
                s = s_prime
                if done:
                    break

            self.env.end_episode()

        print()

    def create_pi_by_Q(self) -> np.ndarray:
        pi = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=float)
        for s in range(self.env.observation_space.n):
            A_start = self._random_argmax(self.Q[s])
            pi[s][A_start] = 1.0
        return pi
