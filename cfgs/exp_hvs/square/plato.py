from attrdict import AttrDict as d

from cfgs.dataset import np_interaction
from cfgs.exp_hvs.square import lmp
from cfgs.model import plato

from configs.fields import Field as F
from muse.envs.robosuite.robosuite_utils import rs_gripper_width_as_contact_fn
from muse.models.bc.lmp.play_helpers import get_parse_interaction_from_episode_fn

parse_fn = get_parse_interaction_from_episode_fn(rs_gripper_width_as_contact_fn, max_contact_len=0, bounds=True)

export = lmp.export.leaf_filter(lambda k, v: 'dataset_' not in k) & d(
    model=plato.export & d(
        device=F('device'),
        horizon=F('horizon'),
    ),

    # sequential dataset modifications (adding input file)
    dataset_train=np_interaction.export & d(
        parse_interaction_bounds_from_episode_fn=parse_fn,
        load_episode_range=[0.0, 0.9],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/hvsBlock3D/{x}.npz'),
        batch_names_to_get=['policy_type', 'robot0_eef_pos', 'object', 'robot0_gripper_qpos', 'policy_switch',
                            'robot0_eef_quat', 'action'],
    ),
    dataset_holdout=np_interaction.export & d(
        parse_interaction_bounds_from_episode_fn=parse_fn,
        load_episode_range=[0.9, 1.0],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/hvsBlock3D/{x}.npz'),
        batch_names_to_get=['policy_type', 'robot0_eef_pos', 'object', 'robot0_gripper_qpos', 'policy_switch',
                            'robot0_eef_quat', 'action'],
    ),

)
