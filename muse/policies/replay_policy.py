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
        self._done_key = get_with_default(params, "done_key", 'done')
        self._ep_idx = get_with_default(params, "ep_idx", 0)
        assert set(self._action_names).issubset(self._env_spec.all_names)

    def _init_setup(self):
        logger.debug(f'Loading demo file: {self._demo_file}')
        self._demo_data = np.load(self._demo_file, allow_pickle=True)
        
        # lazy loading since we don't need obs
        self._demo_actions = AttrDict()
        for key in self._action_names:
            self._demo_actions[key] = self._demo_data[key]

        if self._done_key in self._demo_data:
            done = self._demo_data[self._done_key]
            # keep the first episode
            self._split_indices = np.nonzero(done)[0]
            self._start_indices = np.concatenate([[0], self._split_indices[:-1]])
            self._demo_actions.leaf_modify(
                lambda arr: arr[self._start_indices[self._ep_idx]:self._split_indices[self._ep_idx]])
        
        self._demo_len = len(self._demo_actions.get_one())
        self._counter = 0
        self._demo_counter = 0  # used for cycling
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