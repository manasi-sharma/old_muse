import gymnasium as gym
import numpy as np
from gymnasium import envs

from muse.envs.env import Env, make
from muse.envs.env_spec import EnvSpec

from attrdict import AttrDict

from muse.envs.simple.gym_spec import GymEnvSpec
from muse.utils.general_utils import value_if_none
from muse.utils.torch_utils import to_numpy


SUPPORTED = {
    env_spec.id: env_spec
    for k, env_spec in envs.registry.items()
}


class GymEnv(Env):
    """
    Wrapper for muse.Env for gym environment
    """
    def __init__(self, params, env_spec: EnvSpec):
        super().__init__(params, env_spec)

        # PARAMS
        self._render = params.render
        self._env_type = params.env_type

        # LOCAL VARIABLES
        self._env = gym.make(self._env_type, render_mode="human" if self._render else None)

        self._obs = np.zeros(self._env.observation_space.shape)
        self._reward = 0
        self._done = False
        self._truncated = False

    def _init_parameters(self, params):
        pass

    # this will be overriden
    def step(self, action, **kwargs):
        # batch input
        base_action = to_numpy(action.action[0], check=True)
        self._obs, self._reward, self._done, self._truncated, info = self._env.step(base_action)
        if self._render:
            self._env.render()
        return self.get_obs(), self.get_goal(), np.array([self._done])

    def reset(self, presets: AttrDict = None):
        self._obs, info = self._env.reset()
        self._reward = 0
        return self.get_obs(), self.get_goal()

    def get_obs(self):
        return self._env_spec.map_to_types(
            AttrDict(
                obs=self._obs.copy()[None],
                reward=np.array([[self._reward]])
            )
        )

    def get_goal(self):
        return AttrDict()

    default_params = AttrDict(env_type='BipedalWalker-v3')

    @staticmethod
    def get_default_env_spec_params(params: AttrDict = None) -> AttrDict:
        params = value_if_none(params, AttrDict())

        env_wrap = gym.make(params.env_type)
        base_nsld = [
            ('obs', tuple(env_wrap.observation_space.shape),
             (env_wrap.observation_space.low, env_wrap.observation_space.high), env_wrap.observation_space.dtype.type),
            ('reward', (1,), (-np.inf, np.inf), np.float32),
            ('action', tuple(env_wrap.action_space.shape), (env_wrap.action_space.low, env_wrap.action_space.high),
             env_wrap.action_space.dtype.type),

            # other names that might be used.
            ('goal/obs', tuple(env_wrap.observation_space.shape),
             (env_wrap.observation_space.low, env_wrap.observation_space.high), env_wrap.observation_space.dtype.type),
            ('next/obs', tuple(env_wrap.observation_space.shape),
             (env_wrap.observation_space.low, env_wrap.observation_space.high), env_wrap.observation_space.dtype.type),
        ]
        del env_wrap

        return AttrDict(cls=GymEnvSpec, names_shapes_limits_dtypes=base_nsld)


if __name__ == '__main__':
    from muse.experiments import logger
    env = make(GymEnv, AttrDict(env_type='BipedalWalker-v3', render=True))

    env.reset()

    done = [False]
    while not done[0]:
        action = env.env_spec.get_uniform(env.env_spec.action_names, 1)
        obs, goal, done = env.step(action)

    logger.debug('Done.')
