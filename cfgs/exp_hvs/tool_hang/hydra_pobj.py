import numpy as np
from attrdict import AttrDict as d

from cfgs.env import tool_hang
from cfgs.exp_hvs.tool_hang import hydra as th_hydra
from muse.envs.robosuite.robosuite_env import RobosuiteEnv
from muse.envs.robosuite.robosuite_utils import modify_spec_prms

env_params = tool_hang.export & d(
    parse_objects=True,
)
env_spec = modify_spec_prms(RobosuiteEnv.get_default_env_spec_params(env_params),
                            include_mode=True, include_target_names=True)
env_spec.names_shapes_limits_dtypes.append(("smooth_mode", (1,), (0, 255), np.uint8))
env_spec.names_shapes_limits_dtypes.append(("mask", (1,), (0, 1), np.uint8))

# same as tool hang but replace "object" with parsed objects (e.g. pos and ori)
export = th_hydra.export & d(
    dataset='mode_clean_real2_v3_fast_human_tool_hang_pobj_100k',
    env_train=env_params,
    env_spec=env_spec,
    model=d(
        goal_names=['objects/position', 'objects/orientation'],
        state_names=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'objects/position',
                     'objects/orientation'],
    ),
    dataset_train=d(
        batch_names_to_get=['policy_type', 'robot0_eef_pos', 'objects/position', 'objects/orientation',
                            'robot0_gripper_qpos', 'policy_switch',
                            'robot0_eef_quat', 'action', 'mode', 'target/position', 'target/orientation',
                            'target/orientation_eul', 'mask', 'smooth_mode'],
    ),
    dataset_holdout=d(
        batch_names_to_get=['policy_type', 'robot0_eef_pos', 'objects/position', 'objects/orientation',
                            'robot0_gripper_qpos', 'policy_switch',
                            'robot0_eef_quat', 'action', 'mode', 'target/position', 'target/orientation',
                            'target/orientation_eul', 'mask', 'smooth_mode'],
    ),
)
