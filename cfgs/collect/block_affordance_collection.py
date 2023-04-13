from muse.envs.bullet_envs.block3d.block_env_3d import BlockEnv3D
from cfgs.env import block3d
from muse.models.model import Model

from muse.policies.random_policy_bullet import RandomPolicyBullet
from attrdict import AttrDict as d

export = d(
    exp_name='affordance',
    cls=BlockEnv3D,
    env_spec=BlockEnv3D.get_default_env_spec_params(block3d.export),
    env_train=block3d.export,
    num_blocks=1,
    horizon=100,

    policy=d(
        cls=RandomPolicyBullet,
    ),
    model=d(
        cls=Model,
        ignore_inputs=True
    )
)
