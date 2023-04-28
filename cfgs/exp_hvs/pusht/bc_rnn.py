from attrdict import AttrDict as d

from cfgs.dataset import np_seq
from cfgs.env import pusht
from cfgs.model import bc_rnn

from cfgs.trainer import rm_goal_trainer
from configs.fields import Field as F
from muse.envs.robosuite.robosuite_env import RobosuiteEnv
from muse.envs.robosuite.robosuite_utils import get_rs_online_action_postproc_fn
from muse.policies.basic_policy import BasicPolicy
from muse.policies.bc.gcbc_policy import GCBCPolicy
from muse.policies.memory_policy import get_timeout_terminate_fn

export = d(
    device="cuda",
    batch_size=256,
    horizon=10,
    seed=0,
    dataset='human_square_30k',
    exp_name='hvsBlock3D/velact_{?seed:s{seed}_}b{batch_size}_h{horizon}_{dataset}',
    # utils=utils,
    env_spec=RobosuiteEnv.get_default_env_spec_params(pusht.export),
    env_train=pusht.export,
    model=bc_rnn.export & d(
        device=F('device'),
        horizon=F('horizon'),
    ),

    # sequential dataset modifications (adding input file)
    dataset_train=np_seq.export & d(
        load_episode_range=[0.0, 0.9],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/hvsBlock3D/{x}.npz'),
        batch_names_to_get=['policy_type', 'robot0_eef_pos', 'object', 'robot0_gripper_qpos', 'policy_switch',
                            'robot0_eef_quat', 'action'],
    ),
    dataset_holdout=np_seq.export & d(
        load_episode_range=[0.9, 1.0],
        horizon=F('horizon'),
        batch_size=F('batch_size'),
        file=F('dataset', lambda x: f'data/hvsBlock3D/{x}.npz'),
        batch_names_to_get=['policy_type', 'robot0_eef_pos', 'object', 'robot0_gripper_qpos', 'policy_switch',
                            'robot0_eef_quat', 'action'],
    ),

    policy=d(
        cls=GCBCPolicy,
        velact=True,
        recurrent=True,
        policy_out_names=['action'],
        policy_out_norm_names=[],
        fill_extra_policy_names=True,
        online_action_postproc_fn=get_rs_online_action_postproc_fn(no_ori=False, fast_dynamics=True),
        is_terminated_fn=get_timeout_terminate_fn(400),
    ),
    goal_policy=d(
        cls=BasicPolicy,
        policy_model_forward_fn=lambda m, o, g, **kwargs: d(),
        timeout=2,
    ),
    trainer=rm_goal_trainer.export,
)
