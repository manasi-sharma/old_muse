#!/usr/bin/env python3
import argparse

import numpy as np
import sys
import pybullet as p

from muse.envs.bullet_envs.block3d.block_env_3d import BlockEnv3D
from muse.envs.bullet_envs.teleop_functions import bullet_teleop_keys_rollout
from muse.envs.bullet_envs.utils_env import RobotControllerMode
from muse.envs.env import make

from muse.policies.controllers.pid_controller import ControlType

from attrdict import AttrDict as d
from attrdict.utils import get_with_default


# plotting forces
import multiprocessing as mp


RCM = RobotControllerMode
CT = ControlType

DEFAULT_PLAT_HEIGHT = 0.16/3


class PlatformBlockEnv3D(BlockEnv3D):

    default_params = BlockEnv3D.default_params & d(
        object_start_bounds={
            'block': (np.array([-0.5, -0.5]), np.array([0.5, 0.5])),
            'mug': (np.array([-0.45, -0.45]), np.array([0.45, 0.45]))
            # % [-1, 1] is the range for each, object initializes inside this percentage of w/2
        },
    )

    def _init_params_to_attrs(self, params: d):
        # change object start bounds
        self._platform_height = get_with_default(params, "platform_height", DEFAULT_PLAT_HEIGHT)
        self._platform_extent = get_with_default(params, "platform_extent", 0.08)
        self._init_obj_on_platform = get_with_default(params, "init_obj_on_platform", False)  # TODO implement

        assert not self._init_obj_on_platform, "not implemented yet"

        super(PlatformBlockEnv3D, self)._init_params_to_attrs(params)

    def load_surfaces(self):
        super(PlatformBlockEnv3D, self).load_surfaces()

        h = self._platform_height / 2  # third of the cabinet height TODO param
        ext = self._platform_extent / 2  # protrudes this much into the table TODO param

        # four edge platforms for lifting
        w1 = self._create_cabinet_fn(halfExtents=[self.surface_bounds[0] / 2, ext, h],
                                     location=self.surface_center + np.array([0, self.surface_bounds[1] / 2 - ext, h / 2]))
        w2 = self._create_cabinet_fn(halfExtents=[self.surface_bounds[0] / 2, ext, h],
                                     location=self.surface_center + np.array([0, -self.surface_bounds[1] / 2 + ext, h / 2]))
        w3 = self._create_cabinet_fn(halfExtents=[ext, self.surface_bounds[1] / 2, h],
                                     location=self.surface_center + np.array([self.surface_bounds[0] / 2 - ext, 0, h / 2]))
        w4 = self._create_cabinet_fn(halfExtents=[ext, self.surface_bounds[1] / 2, h],
                                     location=self.surface_center + np.array([-self.surface_bounds[0] / 2 + ext, 0, h / 2]))

        _, aabbmax = p.getAABB(w1, -1, physicsClientId=self.id)
        self.platform_z = aabbmax[2]

        self.cabinet_obj_ids.extend([w1, w2, w3, w4])

    def get_nearest_platform(self, obj_obs, return_all=True, margin=0):
        # distances to each platform
        obj_pos = obj_obs["position"][0]
        obj_aabb = obj_obs["aabb"][0]
        obj_height = obj_aabb[5] - obj_aabb[2]
        closest_points = np.array([
            [self.surface_center[0] + self.surface_bounds[0] / 2 - self._platform_extent/2 - margin, obj_pos[1], obj_pos[2]],
            [self.surface_center[0] - self.surface_bounds[0] / 2 + self._platform_extent/2 + margin, obj_pos[1], obj_pos[2]],
            [obj_pos[0], self.surface_center[1] + self.surface_bounds[1] / 2 - self._platform_extent/2 - margin, obj_pos[2]],
            [obj_pos[0], self.surface_center[1] - self.surface_bounds[1] / 2 + self._platform_extent/2 + margin, obj_pos[2]],
        ])
        closest_dist = np.abs(closest_points - obj_pos).sum(-1)

        close_idx = np.argmin(closest_dist)
        cp = closest_points[close_idx]

        # z will be such that object is flush with platform
        cp[2] = obj_height/2 + self.platform_z

        if return_all:
            # close_pt (3,), idx, distances (4,), points (4,3)
            return cp, close_idx, closest_dist, closest_points
        return cp


class PlatformMugEnv3D(PlatformBlockEnv3D):
    default_params = PlatformBlockEnv3D.default_params & d(
        object_spec=['mug'],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_mug', action='store_true')
    args = parser.parse_args()

    if sys.platform != "linux":
        mp.set_start_method('spawn')  # macos thing i think

    max_steps = 10000

    params = d(
        render=True,
        debug_cam_dist=0.35,
        debug_cam_p=-45,
        debug_cam_y=0,
        debug_cam_target_pos=[0.4, 0, 0.45],
    )

    cls = PlatformMugEnv3D if args.use_mug else PlatformBlockEnv3D
    env = make(cls, params)

    bullet_teleop_keys_rollout(env, max_steps=max_steps)
