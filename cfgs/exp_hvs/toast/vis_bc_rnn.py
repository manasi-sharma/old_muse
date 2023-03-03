import torch

from cfgs.dataset import np_seq_train_val
from cfgs.env import polymetis_panda
from cfgs.model import vis_bc_rnn
from configs.exp_hvs.rm_utils import Robot3DLearningUtils
from muse.datasets.preprocess.data_augmentation import DataAugmentation
from muse.envs.param_spec import ParamEnvSpec
from muse.envs.polymetis.polymetis_panda_env import get_polymetis_panda_example_spec_params
from muse.policies.basic_policy import BasicPolicy
from muse.policies.bc.gcbc_policy import GCBCPolicy
from muse.trainers.trainer import Trainer
from muse.utils.config_utils import parse_cfgs, get_config_args, default_process_env_step_output_fn
from muse.utils.python_utils import AttrDict as d
from muse.utils.torch_utils import get_augment_fn

utils = Robot3DLearningUtils(fast_dynamics=True)

params = np_seq_train_val.params & d(
    exp_name='real/velact_STATIC_rpo-seq_b8_lr0_0001_dec0_h10-10_raw_mode_toast-bread_80ep_'
             'split0_9_bc-imobs-eimobs_qt_normgrabil_no-vel_l2err_tanh_imcrop_spatmax_lstm-hs1000-ps0',
    env_spec=d(
        cls=ParamEnvSpec,
        params=get_polymetis_panda_example_spec_params(action_space='ee-euler-delta',
                                                       use_imgs=True, img_height=240, img_width=320,
                                                       use_ego_imgs=True, ego_img_height=240, ego_img_width=320,
                                                       use_ego_depth=False, )
    ),
    env_train=polymetis_panda.params,
    model=vis_bc_rnn.params & d(
        params=d(
            state_names=['ee_position', 'ee_orientation', 'gripper_pos'],
        )
    ),
    dataset_train=d(
        params=d(
            file='data/real/raw_mode_toast-bread_80ep.npz',
            index_all_keys=False,
            capacity=100000,
            batch_size=8,
            horizon=10,
            batch_names_to_get=['policy_type', 'gripper_pos', 'policy_switch', 'ee_position', 'action',
                                'ee_orientation', 'image', 'ego_image'],
        ),
    ),
    dataset_holdout=d(
        params=d(
            file='data/real/raw_mode_toast-bread_80ep.npz',
            index_all_keys=False,
            capacity=100000,
            batch_size=8,
            horizon=10,
            batch_names_to_get=['policy_type', 'gripper_pos', 'policy_switch', 'ee_position', 'action',
                                'ee_orientation', 'image', 'ego_image'],
        ),
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
            is_terminated_fn=lambda model, obs, goal, mem,
                                    **kwargs: False if mem.is_empty() else mem >> "count" >= 1200,
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
        cls=Trainer,
        params=d(
            max_steps=600000,
            train_every_n_steps=1,
            block_train_on_first_n_steps=0,
            step_train_env_every_n_steps=0,
            step_holdout_env_every_n_steps=0,
            holdout_every_n_steps=50,
            episode_return_buffer_len=1,
            write_average_episode_returns_every_n_env_steps=20,
            max_grad_norm=None,
            data_augmentation_params=d(
                cls=DataAugmentation,
                params=d(
                    augment_keys=['ee_position', 'ee_orientation'],
                    augment_fns=[get_augment_fn(0.005), get_augment_fn(0.0005)],
                )
            ),
            train_do_data_augmentation=True,
            torchify_dataset=False,
            torchify_device='cpu',
            load_statistics_initial=True,
            reload_statistics_every_n_env_steps=0,
            log_every_n_steps=1000,
            save_every_n_steps=20000,
            save_checkpoint_every_n_steps=20000,
            save_data_train_every_n_steps=0,
            save_data_holdout_every_n_steps=0,
            checkpoint_model_file='model.pt',
            save_checkpoints=True,
            base_optimizer=lambda p: torch.optim.Adam(p, lr=1e-4, betas=(0.9, 0.999), weight_decay=0),
            process_env_step_output_fn=default_process_env_step_output_fn,
        ),
    ),
)


# parses supported values from command line
params = parse_cfgs(params, get_config_args())
