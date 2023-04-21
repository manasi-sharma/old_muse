from attrdict import AttrDict as d

from cfgs.dataset import np_img_base_seq
from cfgs.env import polymetis_panda
from cfgs.model import vis_bc_rnn

#from cfgs.trainer import real_trainer, real_goal_trainer
from cfgs.trainer import real_goal_trainer
from configs.fields import Field as F
from muse.envs.polymetis.polymetis_utils import get_polymetis_online_action_postproc_fn
from muse.policies.basic_policy import BasicPolicy
from muse.policies.bc.gcbc_policy import GCBCPolicy
from muse.policies.memory_policy import get_timeout_terminate_fn

export = d(
    augment=True,
    device="cuda",
    batch_size=8,
    horizon=10,
    dataset='raw_mode_make-coffee-v3_eimgs_100ep',
    exp_name='real/velact_{?augment:aug_}b{batch_size}_h{horizon}_{dataset}',
    # utils=utils,
    env_spec=polymetis_panda.export.cls.get_default_env_spec_params(polymetis_panda.export),
    env_train=polymetis_panda.export,
    model=vis_bc_rnn.export & d(
        state_names=['ee_position', 'ee_orientation', 'gripper_pos'],
        vision_encoder=d(
            image_shape=[240, 320, 3],
            img_embed_size=128,
        ),
        device=F('device'),
    ),

    # sequential dataset modifications (adding input file)
    dataset_train=np_img_base_seq.export & d(
        load_episode_range=[0.0, 0.9],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/real/{x}.npz'),
        batch_names_to_get=['policy_type', 'gripper_pos', 'policy_switch', 'ee_position', 'action',
                            'ee_orientation', 'image', 'ego_image'],
    ),
    dataset_holdout=np_img_base_seq.export & d(
        load_from_base=True,
        load_episode_range=[0.9, 1.0],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/real/{x}.npz'),
        batch_names_to_get=['policy_type', 'gripper_pos', 'policy_switch', 'ee_position', 'action',
                            'ee_orientation', 'image', 'ego_image'],
    ),

    policy=d(
        cls=GCBCPolicy,
        velact=True,
        recurrent=True,
        replan_horizon=F('horizon'),
        policy_out_names=['action'],
        policy_out_norm_names=[],
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
