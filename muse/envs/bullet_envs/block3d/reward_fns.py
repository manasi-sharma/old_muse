"""
Versions of the Block3D environment with different success metrics
"""

from muse.envs.bullet_envs.block3d.block_env_3d import BlockEnv3D


def get_lift_reward(lift_distance=0.15):
    def lift_reward_fn(curr_obs, goal, action, next_obs=None, done=None, env=None):
        assert isinstance(env, BlockEnv3D)
        obj_z = curr_obs['objects/position'][0, 0, 2]
        table_z = env.table_aabb[1, 2]

        return obj_z > table_z + lift_distance

    return lift_reward_fn
