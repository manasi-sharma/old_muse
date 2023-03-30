from attrdict import AttrDict as d

from cfgs.env import tool_hang
from muse.envs.robosuite.robosuite_env import RobosuiteEnv

# use this as a starting point.
from cfgs.exp_hvs.tool_hang import bc_rnn as th_bc_rnn
from muse.envs.robosuite.robosuite_utils import modify_spec_prms

env_params = tool_hang.export & d(
    parse_objects=True,
)
env_spec = modify_spec_prms(RobosuiteEnv.get_default_env_spec_params(env_params))

export = th_bc_rnn.export & d(
    dataset='human_tool_hang_pobj_100k',
    env_spec=env_spec,
    env_train=env_params,
    model=d(
        goal_names=['objects/position', 'objects/orientation'],
        state_names=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'objects/position',
                     'objects/orientation'],
    ),
    dataset_train=d(
        batch_names_to_get=['policy_type', 'robot0_eef_pos', 'objects/position', 'objects/orientation',
                            'robot0_gripper_qpos', 'policy_switch', 'robot0_eef_quat', 'action'],
    ),
    dataset_holdout=d(
        batch_names_to_get=['policy_type', 'robot0_eef_pos', 'objects/position', 'objects/orientation',
                            'robot0_gripper_qpos', 'policy_switch', 'robot0_eef_quat', 'action'],
    ),
)
