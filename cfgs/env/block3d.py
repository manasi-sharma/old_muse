from muse.envs.bullet_envs.block3d.block_env_3d import BlockEnv3D

from configs.fields import Field as F
from attrdict import AttrDict as d

export = BlockEnv3D.default_params & d(
    cls=BlockEnv3D,
    num_blocks=1,
    do_random_ee_position=False,
    img_width=128,
    img_height=128,
    render=True,
    compute_images=F('render'),
)
