"""Wrappers shared by training and evaluation."""

from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper


class FlattenObs(BaseParallelWrapper):
    """Flatten each agent's Dict observation (SuperSuit's flatten does not handle Dict)."""

    def __init__(self, env):
        super().__init__(env)
        self._flat = {a: spaces.flatten_space(env.observation_space(a))
                      for a in env.possible_agents}

    def observation_space(self, agent):
        return self._flat[agent]

    def _flatten(self, observations):
        return {a: spaces.flatten(self.env.observation_space(a), obs)
                for a, obs in observations.items()}

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return self._flatten(observations), infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return self._flatten(observations), rewards, terminations, truncations, infos


class SeedableSB3VecEnvWrapper(SB3VecEnvWrapper):
    """SB3VecEnvWrapper with a seed() that actually reaches the next reset.

    The underlying SuperSuit vector envs only take a seed through reset(), and
    the base wrapper neither defines seed() nor forwards it.
    """

    def __init__(self, venv):
        super().__init__(venv)
        self._pending_seed = None

    def seed(self, seed=None):
        self._pending_seed = seed
        return [seed] * self.num_envs

    def reset(self, seed=None, options=None):
        use = seed if seed is not None else self._pending_seed
        self._pending_seed = None
        observations, self.reset_infos = self.venv.reset(seed=use, options=options)
        return observations
