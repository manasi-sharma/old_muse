from attrdict import AttrDict as d

from configs.fields import GroupField
from muse.envs.bullet_envs.block3d.block_env_3d import BlockEnv3D
from cfgs.env import block3d
from muse.envs.bullet_envs.block3d.reward_fns import get_lift_reward
from muse.models.model import Model
from muse.policies.meta_policy import MetaPolicy

from muse.policies.scripted.pybullet_policies import get_lift_block_policy_params, WaypointPolicy

env_extra = d(
    env_reward_fn=get_lift_reward(lift_distance=0.15)
)

export = d(
    exp_name='affordance/collect',
    cls=BlockEnv3D,
    env_spec=GroupField('env_train', BlockEnv3D.get_default_env_spec_params),
    env_train=block3d.export & env_extra,

    policy=d(
        cls=MetaPolicy,
        num_policies=1,
        max_policies_per_reset=1,  # one policy then terminate
        policy_0=d(cls=WaypointPolicy),
        next_param_fn=lambda idx, _, obs, goal, env=None, **kwargs:
            (0, get_lift_block_policy_params(obs, goal, env=env, random_ee_ori=True, random_ee_offset=True))
    ),
    model=d(
        cls=Model,
        ignore_inputs=True
    )
)
