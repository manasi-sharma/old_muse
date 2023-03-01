"""
Replays a demonstration
"""
from muse.policies.policy import Policy
from attrdict import AttrDict
from attrdict.utils import get_with_default
from muse.experiments import logger
import numpy as np

class ReplayPolicy(Policy):

    def _init_params_to_attrs(self, params):
        self._demo_file = params["demo_file"]
        self._action_names = get_with_default(params, "action_names", self._out_names)
        assert set(self._action_names).issubset(self._env_spec.all_names)

    def _init_setup(self):
        logger.debug(f'Loading demo file: {self._demo_file}')
        self._demo_data = np.load(self._demo_file, allow_pickle=True)
        
        # lazy loading since we don't need obs
        self._demo_actions = AttrDict()
        for key in self._action_names:
            self._demo_actions[key] = self._demo_data[key]
        
        self._demo_len = len(self._demo_actions.get_one())
        self._counter = 0
        logger.debug(f'Demo len: {self._demo_len}, actions = {self._action_names}')

    def warm_start(self, model, observation, goal):
        pass
    
    def reset_policy(self, **kwargs):
        self._counter = 0
        return super().reset_policy(**kwargs)

    def get_action(self, model, observation, goal, **kwargs):
        """
        :param model: (Model)
        :param observation: (AttrDict)  (1 x H x ...)
        :param goal: (AttrDict) (1 x H x ...)

        :return action: AttrDict (1 x ...)
        """
        action = self._demo_actions.leaf_apply(lambda arr: arr[self._counter, None])
        self._counter += 1
        return action

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self._counter >= self._demo_len