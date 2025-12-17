import gymnasium as gym


class NonStationaryCliffWalking(gym.Wrapper):
    def __init__(self, env: gym.Env, switch_episode=3000, extra_cliff_row=2):
        super().__init__(env)
        self.switch_episode = switch_episode
        self.extra_cliff_row = extra_cliff_row
        self._n_episodes = 0
        self.switched = False

    def _to_rc(self, s: int) -> tuple[int, int]:
        return divmod(s, 12)

    def _to_s(self, r: int, c: int) -> int:
        return r * 12 + c

    def _is_cliff(self, r: int, c: int) -> bool:
        base = (r == 4 - 1) and (1 <= c <= 12 - 2)
        extra = False if self.switched else (r == self.extra_cliff_row) and (1 <= c <= 12 - 2)
        return base or extra

    def step(self, action):
        s_prime, r, terminated, truncated, info = self.env.step(action)

        row, col = self._to_rc(s_prime)
        if self._is_cliff(row, col):
            r = -100.0
            terminated = True
            s_prime = self._to_s(4 - 1, 0)

        return s_prime, r, terminated, truncated, info

    def end_episode(self) -> None:
        self._n_episodes += 1
        if self._n_episodes >= self.switch_episode:
            self.switch()

    def switch(self) -> None:
        self.switched = True
