import torch

from cfgs.dataset import np_seq_train_val
from cfgs.env import tool_hang
from cfgs.model import bc_rnn
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
    exp_name='hvsBlock3D/velact_STATIC_seq_b256_lr0_0001_dec0_h10-10_' \
             'human_tool_hang_100k_split0_9_bc_qt_normgrabil_no-vel_l2err_tanh_lstm-hs1000-ps0',
    utils=utils,
    env_spec=d(
        cls=ParamEnvSpec,
        params=get_rs_example_spec_params("ToolHang", img_width=240, img_height=240),
    ),
    env_train=tool_hang.params,
    model=bc_rnn.params & d(
        params=d(
            hidden_size=1000
        )
    ),

    # sequential dataset modifications (adding input file)
    dataset_train=d(
        params=d(
            horizon=10,
            batch_size=256,
            file='data/hvsBlock3D/human_tool_hang_100k.npz',
            batch_names_to_get=['policy_type', 'robot0_eef_pos', 'object', 'robot0_gripper_qpos', 'policy_switch',
                                'robot0_eef_quat', 'action'],
        )
    ),
    dataset_holdout=d(
        params=d(
            horizon=10,
            batch_size=256,
            file='data/hvsBlock3D/human_tool_hang_100k.npz',
            batch_names_to_get=['policy_type', 'robot0_eef_pos', 'object', 'robot0_gripper_qpos', 'policy_switch',
                                'robot0_eef_quat', 'action'],
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
            is_terminated_fn=lambda model, obs, goal, mem, **kwargs: False if mem.is_empty() else mem >> "count" >= 700,
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
            holdout_every_n_steps=36,
            rollout_train_env_every_n_steps=20000,
            rollout_train_env_n_per_step=36,
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
