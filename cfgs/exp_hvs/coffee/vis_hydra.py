from attrdict import AttrDict as d

from cfgs.dataset import np_img_base_seq
from cfgs.env import polymetis_panda
from cfgs.model import vis_hydra

from cfgs.trainer import real_goal_trainer
from configs.fields import Field as F
from muse.envs.polymetis.polymetis_utils import get_polymetis_online_action_postproc_fn, modify_spec_prms, \
    get_wp_dynamics_fn
from muse.policies.basic_policy import BasicPolicy
from muse.policies.bc.hydra.hydra_policy import HYDRAPolicy
from muse.policies.memory_policy import get_timeout_terminate_fn

export = d(
    augment=True,
    device="cuda",
    batch_size=8,
    horizon=10,
    dataset='mode_real_fast_make-coffee-v3_eimgs_100ep',
    exp_name='real/velact_{?augment:aug_}b{batch_size}_h{horizon}_{dataset}',
    # utils=utils,
    env_spec=modify_spec_prms(polymetis_panda.export.cls.get_default_env_spec_params(polymetis_panda.export),
                              include_mode=True, include_target_names=True) & d(
        wp_dynamics_fn=get_wp_dynamics_fn(fast_dynamics=True),
    ),
    env_train=polymetis_panda.export,
    model=vis_hydra.export & d(
        state_names=['ee_position', 'ee_orientation', 'gripper_pos'],
        goal_names=[],
        sparse_action_names=['target/ee_position', 'target/ee_orientation'],
        device=F('device'),
    ),

    # sequential dataset modifications (adding input file)
    dataset_train=np_img_base_seq.export & d(
        load_episode_range=[0.0, 0.9],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/real/{x}.npz'),
        batch_names_to_get=['ee_orientation', 'ee_position', 'gripper_pos', 'policy_type', 'policy_switch',
                            'action', 'image', 'ego_image', 'mode',
                            'target/ee_position', 'target/ee_orientation', 'target/ee_orientation_eul'],
    ),

    dataset_holdout=np_img_base_seq.export & d(
        load_from_base=True,
        load_episode_range=[0.9, 1.0],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/real/{x}.npz'),
        batch_names_to_get=['ee_orientation', 'ee_position', 'gripper_pos', 'policy_type', 'policy_switch',
                            'action', 'image', 'ego_image', 'mode',
                            'target/ee_position', 'target/ee_orientation', 'target/ee_orientation_eul'],
    ),

    policy=d(
        cls=HYDRAPolicy,
        state_keys=['ee_position', 'ee_orientation', 'ee_orientation_eul'],
        sparse_policy_out_names=['target/ee_position', 'target/ee_orientation'],
        velact=True,
        recurrent=True,
        replan_horizon=10,
        policy_out_names=['target/ee_position', 'target/ee_orientation', 'action'],
        policy_out_norm_names=['target/ee_position', 'target/ee_orientation'],
        mode_key="mode",
        fill_extra_policy_names=True,
        online_action_postproc_fn=get_polymetis_online_action_postproc_fn(no_ori=False, fast_dynamics=True),
        is_terminated_fn=get_timeout_terminate_fn(1200),
    ),
    goal_policy=d(
        cls=BasicPolicy,
        policy_model_forward_fn=lambda m, o, g, **kwargs: d(),
        timeout=2,
    ),
    trainer=real_goal_trainer.export & d(
        train_do_data_augmentation=F('augment'),
    ),
)
