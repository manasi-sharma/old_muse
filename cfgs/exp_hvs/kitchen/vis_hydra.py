import numpy as np

from cfgs.env import kitchen
from cfgs.exp_hvs.kitchen import vis_bc_rnn as kitchen_vis_bcrnn
from cfgs.model import vis_hydra

from attrdict import AttrDict as d

from configs.fields import Field as F

from muse.envs.robosuite.robosuite_env import RobosuiteEnv
from muse.envs.robosuite.robosuite_utils import modify_spec_prms, get_rs_online_action_postproc_fn, get_wp_dynamics_fn
from muse.policies.bc.hydra.hydra_policy import HYDRAPolicy
from muse.policies.memory_policy import get_timeout_terminate_fn

env_spec_prms = RobosuiteEnv.get_default_env_spec_params(kitchen.export)
env_spec_prms = modify_spec_prms(env_spec_prms, no_object=True, include_mode=True, include_target_names=True)
env_spec_prms.names_shapes_limits_dtypes.append(("smooth_mode", (1,), (0, 255), np.uint8))
env_spec_prms.names_shapes_limits_dtypes.append(("mask", (1,), (0, 1), np.uint8))
env_spec_prms.wp_dynamics_fn = get_wp_dynamics_fn(fast_dynamics=True, no_ori=True)

export = kitchen_vis_bcrnn.export & d(
    dataset='mode_clean_r2v2_human_buds-kitchen_60k_eimgs',
    env_spec=env_spec_prms,
    env_train=kitchen.export,
    model=vis_hydra.export & d(
        state_names=['robot0_eef_pos', 'robot0_gripper_qpos'],
        sparse_action_names=['target/position'],
        vision_encoder=d(
            image_shape=[128, 128, 3],
            img_embed_size=128,
        ),
        device=F('device'),
    ),

    # sequential dataset modifications (adding input file)
    dataset_train=d(
        batch_names_to_get=['robot0_eef_pos', 'robot0_gripper_qpos', 'action', 'image', 'ego_image',
                            'target/position', "mode"],
    ),
    dataset_holdout=d(
        batch_names_to_get=['robot0_eef_pos', 'robot0_gripper_qpos', 'action', 'image', 'ego_image',
                            'target/position', "mode"],
    ),
    policy=d(
        cls=HYDRAPolicy,
        sparse_policy_out_names=['target/position'],
        velact=True,
        recurrent=True,
        replan_horizon=F('horizon'),
        policy_out_names=['target/position', 'action'],
        policy_out_norm_names=['target/position'],
        mode_key="mode",
        fill_extra_policy_names=True,
        online_action_postproc_fn=get_rs_online_action_postproc_fn(no_ori=True, fast_dynamics=True),
        is_terminated_fn=get_timeout_terminate_fn(1200),
    ),
)
