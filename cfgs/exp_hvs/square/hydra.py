import numpy as np

from cfgs.dataset import np_seq
from cfgs.exp_hvs import smooth_dpp, mode2mask_dpp
from cfgs.env import square
from cfgs.model import hydra
from cfgs.trainer import rm_goal_trainer
from configs.fields import Field as F
from muse.datasets.preprocess.data_augmentation import DataAugmentation

from muse.envs.robosuite.robosuite_env import RobosuiteEnv
from muse.envs.robosuite.robosuite_utils import get_rs_online_action_postproc_fn, get_wp_dynamics_fn, modify_spec_prms
from muse.policies.basic_policy import BasicPolicy
from muse.policies.bc.hydra.hydra_policy import HYDRAPolicy
from muse.policies.memory_policy import get_timeout_terminate_fn
from attrdict import AttrDict as d

from muse.utils.torch_utils import get_masked_augment_fn

env_spec = modify_spec_prms(RobosuiteEnv.get_default_env_spec_params(square.export),
                            include_mode=True, include_target_names=True)
env_spec.names_shapes_limits_dtypes.append(("smooth_mode", (1,), (0, 255), np.uint8))
env_spec.names_shapes_limits_dtypes.append(("mask", (1,), (0, 1), np.uint8))
env_spec.wp_dynamics_fn = get_wp_dynamics_fn(fast_dynamics=True)


export = d(
    augment=True,
    device="cuda",
    batch_size=256,
    horizon=10,
    dataset='mode_real2_v2_fast_human_square_30k_imgs',
    exp_name='hvsBlock3D/velact_{?augment:aug_}b{batch_size}_h{horizon}_{dataset}',
    env_spec=env_spec,
    env_train=square.export,
    model=hydra.export & d(
        device=F('device'),
        use_smooth_mode=True,
    ),
    # sequential dataset modifications (adding input file)
    dataset_train=np_seq.export & d(
        load_episode_range=[0.0, 0.9],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/hvsBlock3D/{x}.npz'),
        batch_names_to_get=['policy_type', 'robot0_eef_pos', 'object', 'robot0_gripper_qpos', 'policy_switch',
                            'robot0_eef_quat', 'action', 'mode', 'target/position', 'target/orientation',
                            'target/orientation_eul', 'mask', 'smooth_mode'],
        data_preprocessors=[mode2mask_dpp.export, smooth_dpp.export],
    ),
    dataset_holdout=np_seq.export & d(
        load_episode_range=[0.9, 1.0],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/hvsBlock3D/{x}.npz'),
        batch_names_to_get=['policy_type', 'robot0_eef_pos', 'object', 'robot0_gripper_qpos', 'policy_switch',
                            'robot0_eef_quat', 'action', 'mode', 'target/position', 'target/orientation',
                            'target/orientation_eul', 'mask', 'smooth_mode'],
        data_preprocessors=[mode2mask_dpp.export, smooth_dpp.export],
    ),

    policy=d(
        cls=HYDRAPolicy,
        sparse_policy_out_names=['target/position', 'target/orientation'],
        velact=True,
        recurrent=True,
        replan_horizon=F('horizon'),
        policy_out_names=['target/position', 'target/orientation', 'action'],
        policy_out_norm_names=['target/position', 'target/orientation'],
        mode_key="mode",
        fill_extra_policy_names=True,
        online_action_postproc_fn=get_rs_online_action_postproc_fn(no_ori=False, fast_dynamics=True),
        is_terminated_fn=get_timeout_terminate_fn(400),
    ),
    goal_policy=d(
        cls=BasicPolicy,
        policy_model_forward_fn=lambda m, o, g, **kwargs: d(),
        timeout=2,
    ),
    trainer=rm_goal_trainer.export & d(
        train_do_data_augmentation=F('augment'),
        data_augmentation_params=d(
            cls=DataAugmentation,
            post_mask_name='mask',
            read_keys=['mask'],
            augment_keys=['robot0_eef_pos', 'robot0_eef_quat'],
            augment_fns=[get_masked_augment_fn(0.02, corr=True), get_masked_augment_fn(0.04, corr=True)],
        ),
    ),
)
