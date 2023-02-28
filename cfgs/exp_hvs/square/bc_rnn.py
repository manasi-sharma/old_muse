from attrdict import AttrDict as d

from cfgs.dataset import np_seq
from cfgs.env import square
from cfgs.model import bc_rnn

# from configs.exp_hvs.rm_utils import Robot3DLearningUtils

from cfgs.trainer import rm_goal_trainer
from configs.fields import Field as F
from muse.envs.param_spec import ParamEnvSpec
from muse.envs.robosuite.robosuite_env import get_rs_example_spec_params
from muse.policies.basic_policy import BasicPolicy
from muse.policies.bc.gcbc_policy import GCBCPolicy

# utils = Robot3DLearningUtils(fast_dynamics=True)

export = d(
    batch_size=256,
    horizon=10,
    dataset='human_square_30k',
    exp_name='hvsBlock3D/velact_b{batch_size}_h{horizon}_{dataset}',
    # utils=utils,
    env_spec=d(
        cls=ParamEnvSpec,
    ) & get_rs_example_spec_params("NutAssemblySquare", img_width=84, img_height=84),
    env_train=square.export,
    model=bc_rnn.export,

    # sequential dataset modifications (adding input file)
    dataset_train=np_seq.export & d(
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
        replan_horizon=F('horizon'),
        policy_out_names=['action'],
        policy_out_norm_names=[],
        fill_extra_policy_names=True,
        # TODO online_action_postproc_fn=utils.default_online_action_postproc_fn,
        is_terminated_fn=lambda model, obs, goal, mem, **kwargs: False if mem.is_empty() else mem >> "count" >= 400,
    ),
    goal_policy=d(
        cls=BasicPolicy,
        policy_model_forward_fn=lambda m, o, g, **kwargs: d(),
        timeout=2,
    ),
    trainer=rm_goal_trainer.export,
)
