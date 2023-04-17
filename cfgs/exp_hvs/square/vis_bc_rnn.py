from attrdict import AttrDict as d

from cfgs.dataset import np_img_base_seq
from cfgs.env import square
from cfgs.exp_hvs.square import bc_rnn as sq_bc_rnn
from cfgs.model import vis_bc_rnn
from configs.fields import Field as F
from muse.envs.robosuite.robosuite_env import RobosuiteEnv


export = sq_bc_rnn.export.leaf_filter(lambda k, v: 'dataset' not in k) & d(
    batch_size=16,
    dataset='human_square_30k_eimgs',
    env_spec=RobosuiteEnv.get_default_env_spec_params(square.export & d(imgs=True, ego_imgs=True)),
    env_train=square.export & d(imgs=True, ego_imgs=True),
    # sequential dataset modifications (adding input file)
    model=vis_bc_rnn.export & d(
        state_names=['robot0_eef_pos', 'robot0_gripper_qpos'],
        vision_encoder=d(
            image_shape=[84, 84, 3],
            img_embed_size=64,
        ),
    ),

    # sequential dataset modifications (adding input file)
    dataset_train=np_img_base_seq.export & d(
        load_episode_range=[0.0, 0.9],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/hvsBlock3D/{x}.npz'),
        batch_names_to_get=['robot0_eef_pos', 'robot0_gripper_qpos', 'action', 'image', 'ego_image'],
    ),
    dataset_holdout=np_img_base_seq.export & d(
        load_from_base=True,
        load_episode_range=[0.9, 1.0],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/hvsBlock3D/{x}.npz'),
        batch_names_to_get=['robot0_eef_pos', 'robot0_gripper_qpos', 'action', 'image', 'ego_image'],
    ),
    trainer=d(
        max_steps=600000,
    ),
)
