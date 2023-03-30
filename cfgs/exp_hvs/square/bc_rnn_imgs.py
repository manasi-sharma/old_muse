from attrdict import AttrDict as d

from cfgs.env import square
from cfgs.exp_hvs.square import bc_rnn
from muse.envs.robosuite.robosuite_env import RobosuiteEnv

export = bc_rnn.export & d(
    env_spec=RobosuiteEnv.get_default_env_spec_params(square.export & d(imgs=True)),
    env_train=square.export & d(imgs=True),
    # sequential dataset modifications (adding input file)
    dataset_train=d(
        batch_names_to_get=['policy_type', 'robot0_eef_pos', 'object', 'robot0_gripper_qpos', 'policy_switch',
                            'robot0_eef_quat', 'image', 'action'],
    ),
    dataset_holdout=d(
        batch_names_to_get=['policy_type', 'robot0_eef_pos', 'object', 'robot0_gripper_qpos', 'policy_switch',
                            'robot0_eef_quat', 'image', 'action'],
    ),
)
