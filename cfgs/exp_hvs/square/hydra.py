import torch

from cfgs.dataset import np_seq_train_val
from cfgs.exp_hvs import smooth_dpp, mode2mask_dpp
from cfgs.env import square
from cfgs.model import hydra
from cfgs.trainer import reward_tracker

from configs.exp_hvs.rm_utils import Robot3DLearningUtils
from muse.envs.param_spec import ParamEnvSpec
from muse.envs.robosuite.robosuite_env import get_rs_example_spec_params
from muse.policies.basic_policy import BasicPolicy
from muse.policies.bc.gcbc_policy import GCBCPolicy
from muse.trainers.goal_trainer import GoalTrainer
from muse.trainers.optimizers.optimizer import SingleOptimizer
from muse.utils.config_utils import parse_cfgs, get_config_args
from muse.utils.python_utils import AttrDict as d

utils = Robot3DLearningUtils(fast_dynamics=True)

params = np_seq_train_val.params & d(
    exp_name='hvsBlock3D/velact_STATIC_V4_rpo4-m0-sk10-corr-seq_b256_lr0_0001_dec0_h10-10_mode_real2_v2_fast_human_square_30k_'
             'imgs_split0_9_dyn_bc_qt_normgrabil_no-vel_l2err_tanh_sp-normac_sm_gma0_5_mb0_01_tqt_lstm-hs400-ps200',
    utils=utils,
    env_spec=d(
        cls=ParamEnvSpec,
        params=get_rs_example_spec_params("NutAssemblySquare", img_width=84, img_height=84,
                                          include_mode=True, include_target_names=True),
    ),
    env_train=square.params,
    model=hydra.params & d(
        params=d(
            use_smooth_mode=True,
        )
    ),

    # sequential dataset modifications (adding input file)
    dataset_train=d(
        params=d(
            horizon=10,
            batch_size=256,
            file='data/hvsBlock3D/mode_real2_v2_fast_human_square_30k_imgs.npz',
            batch_names_to_get=['robot0_eef_pos', 'policy_type', 'robot0_eef_quat', 'action', 'policy_switch', 'object',
                                'robot0_gripper_qpos', 'mode', 'target/position', 'target/orientation',
                                'target/orientation_eul', 'mask', 'smooth_mode'],
            data_preprocessors=[mode2mask_dpp.params, smooth_dpp.params],
        )
    ),
    dataset_holdout=d(
        params=d(
            horizon=10,
            batch_size=256,
            file='data/hvsBlock3D/mode_real2_v2_fast_human_square_30k_imgs.npz',
            batch_names_to_get=['robot0_eef_pos', 'policy_type', 'robot0_eef_quat', 'action', 'policy_switch', 'object',
                                'robot0_gripper_qpos', 'mode', 'target/position', 'target/orientation',
                                'target/orientation_eul', 'mask', 'smooth_mode'],
            data_preprocessors=[mode2mask_dpp.params, smooth_dpp.params],
        )
    ),

    policy=d(
        cls=GCBCPolicy,
        params=d(
            velact=True,
            recurrent=True,
            replan_horizon=10,
            policy_out_names=['action'],
            policy_out_norm_names=[],
            fill_extra_policy_names=True,
            online_action_postproc_fn=utils.default_online_action_postproc_fn,
            is_terminated_fn=lambda model, obs, goal, mem, **kwargs: False if mem.is_empty() else mem >> "count" >= 400,
        ),
    ),
    goal_policy=d(
        cls=BasicPolicy,
        params=d(
            policy_model_forward_fn=lambda m, o, g, **kwargs: d(),
            timeout=2,
        ),
    ),
    trainer=d(
        cls=GoalTrainer,
        params=d(
            max_steps=400000,
            train_every_n_steps=1,
            block_train_on_first_n_steps=0,
            block_env_on_first_n_steps=20000,
            random_policy_on_first_n_steps=0,
            step_train_env_every_n_steps=0,
            step_train_env_n_per_step=1,
            step_holdout_env_every_n_steps=0,
            step_holdout_env_n_per_step=1,
            add_to_data_train_every_n_goals=0,
            add_to_data_holdout_every_n_goals=0,
            holdout_every_n_steps=50,
            rollout_train_env_every_n_steps=20000,
            rollout_train_env_n_per_step=50,
            rollout_holdout_env_every_n_steps=0,
            rollout_holdout_env_n_per_step=1,
            no_data_saving=True,
            train_do_data_augmentation=False,
            load_statistics_initial=True,
            reload_statistics_every_n_env_steps=0,
            reload_statistics_n_times=0,
            log_every_n_steps=1000,
            save_every_n_steps=20000,
            save_checkpoint_every_n_steps=100000,
            save_data_train_every_n_steps=0,
            save_data_holdout_every_n_steps=0,
            checkpoint_model_file='model.pt',
            save_checkpoints=True,
            optimizer=d(
                cls=SingleOptimizer,
                params=d(
                    max_grad_norm=None,
                    get_base_optimizer=lambda p: torch.optim.Adam(p, lr=1e-4, betas=(0.9, 0.999), weight_decay=0),
                ),
            ),
            trackers=reward_tracker.params,
            write_average_episode_returns_every_n_env_steps=0,
            track_best_name='env_train/returns',
            track_best_key='returns',
        ),
    ),
)


# parses supported values from command line
params = parse_cfgs(params, get_config_args())
