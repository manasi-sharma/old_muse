"""
This is where actions actually get executed on the robot, and observations are received.

NOTE: Why is there no reward? Are we even doing reinforcement learning???
      The reward is just another observation! Viewing it this way is much more flexible,
      especially with model-based RL

Also defines make() to create environments
    e.g., env = make('muse.envs.my_env.MyEnv')

"""
from pydoc import locate
from typing import Callable

from muse.envs.env_spec import EnvSpec
from muse.utils.abstract import BaseClass
from muse.utils.general_utils import value_if_none
from muse.utils.input_utils import UserInput
from attrdict import AttrDict


class Env(BaseClass):

    # things to define if you want to be compatible with make()
    default_params: AttrDict = AttrDict()

    @staticmethod
    def get_default_env_spec_params(params: AttrDict = None) -> AttrDict:
        raise NotImplementedError

    def __init__(self, params, env_spec):
        self._env_spec = env_spec
        self._display = None  # might get filled later
        self._params = params

        self.user_input = None  # might get filled later

    def step(self, action):
        """ Step the environment once.

        Parameters
        ----------
        action: AttrDict (B x ...)

        Returns
        -------
        obs: AttrDict (B x ...)
        goal: AttrDict (B x ...)
        done: ndarray (B)

        """
        raise NotImplementedError

    def reset(self, presets: AttrDict = None):
        """ Reset environment, returns the next episode obs, goal.

        Parameters
        ----------
        presets: some episodes support this

        Returns
        -------
        obs: AttrDict
        goal: AttrDict

        """
        raise NotImplementedError

    def user_input_reset(self, user_input: UserInput, reset_action_fn=None, presets: AttrDict = None):
        """ Reset, but when we want user input in the loop during the reset.

        Default behavior is same as reset.

        Parameters
        ----------
        user_input: UserInput
            UserInput object to query state from
        reset_action_fn: Callable
            Optional external function to run at some point in user input reset
            default behavior is to call it before calling reset
        presets: AttrDict
            presets for reset

        Returns
        -------
        obs: AttrDict
        goal: AttrDict

        """
        self.user_input = user_input
        if isinstance(reset_action_fn, Callable):
            reset_action_fn()
        return self.reset(presets)

    @property
    def env_spec(self):
        return self._env_spec

    @property
    def display(self):
        return self._display

    @property
    def params(self):
        return self._params.leaf_copy()

    def is_success(self) -> bool:
        """ Returns if the environment reached a success criteria.

        Only valid in environments that have a single known success metric.
        TODO: otherwise use muse.task or muse.reward

        Returns
        -------

        """
        return False


def make(env_cls, params=None, use_default_params=True, env_spec=None):
    """ Make an environment

    Parameters
    ----------
    env_cls: Either a class that subclasses from Env or a string defining the class to load
    params: parameters to use when creating environment
    use_default_params: will load in class defaults, treating params as overrides (True by default)
    env_spec: A spec to use. If none, it will use env_cls.default_env_spec, which might be None

    Returns
    -------
    env: Env

    """

    # get the class if it's a string
    if isinstance(env_cls, str):
        env_cls = locate(env_cls)

    assert issubclass(env_cls, Env), f"{env_cls} is not a subclass of Env!"

    params = value_if_none(params, AttrDict())
    if use_default_params:
        params = env_cls.default_params & params

    if env_spec is None:
        env_spec_prms = env_cls.get_default_env_spec_params(params)
        env_spec = env_spec_prms.cls(env_spec_prms)

    return env_cls(params, env_spec)
