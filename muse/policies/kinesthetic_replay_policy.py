"""
Replays a demonstration
"""
from attrdict import AttrDict
from attrdict.utils import get_with_default

from muse.policies.policy import Policy
from muse.experiments import logger
from muse.utils.transform_utils import quat_difference, quat2euler

import numpy as np

GRIPPER_TOLERANCE = 0.01


class ReplayPolicy(Policy):

    def _init_params_to_attrs(self, params):
        self._demo_file = params["demo_file"]
        self._action_names = get_with_default(params, "action_names", self._out_names)
        self._ee_pose_keys = ['ee_position', 'ee_orientation']
        assert set(self._action_names).issubset(self._env_spec.all_names)

    def _init_setup(self):
        self.reload_data(self._demo_file)

        self._old_format = False

    def reload_data(self, demo_file):
        self._demo_file = demo_file
        logger.debug(f'Loading demo file: {self._demo_file}')
        self._demo_data = np.load(self._demo_file, allow_pickle=True)
        # for key in self._demo_data.keys(): print(key)
        # lazy loading since we don't need obs
        self._demo_actions = AttrDict()
        self._demo_poses = AttrDict()
        self._old_format = False

        for key in self._action_names:
            self._demo_actions[key] = self._demo_data[key]
            # print(key, self._demo_actions[key][0])

        for key in self._ee_pose_keys:
            self._demo_poses[key] = self._demo_data[key]

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

        ee_position = self._demo_poses['ee_position'][self._counter, None]
        ee_orientation = self._demo_poses['ee_orientation'][self._counter, None]
        delta_position = ee_position - observation['ee_position'][0]
        delta_orientation = quat2euler(quat_difference(ee_orientation, observation['ee_orientation'][0]))

        # print(ee_position, delta_position)

        new_action = np.concatenate([delta_position, delta_orientation, [[1 - action['action'][-1][-1]]]], axis=1)
        self._counter += 1

        action['action'] = new_action
        return action

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self._counter >= self._demo_len