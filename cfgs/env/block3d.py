from muse.envs.bullet_envs.block3d.block_env_3d import BlockEnv3D
from muse.policies.random_policy import RandomPolicy

from configs.fields import Field as F
from attrdict import AttrDict as d

export = d(
    cls=BlockEnv3D,
    env_name='BlockEnv3D',
    num_blocks=1,
    do_random_ee_position=False,
    img_width=256,
    img_height=256,
    render=True
)
