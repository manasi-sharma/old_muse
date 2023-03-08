import numpy as np
from attrdict import AttrDict as d

from cfgs.env import tool_hang
from muse.envs.robosuite.robosuite_env import RobosuiteEnv
from muse.envs.robosuite.robosuite_utils import modify_spec_prms

# use this as a starting point.
from cfgs.exp_hvs.square import hydra as sq_hydra

from muse.policies.memory_policy import get_timeout_terminate_fn

env_spec = modify_spec_prms(RobosuiteEnv.get_default_env_spec_params(tool_hang.export),
                            include_mode=True, include_target_names=True)
env_spec.names_shapes_limits_dtypes.append(("smooth_mode", (1,), (0, 255), np.uint8))
env_spec.names_shapes_limits_dtypes.append(("mask", (1,), (0, 1), np.uint8))


export = sq_hydra.export & d(
    augment=False,
    dataset='mode_clean_real2_v3_fast_human_tool_hang_100k',
    env_spec=env_spec,
    env_train=tool_hang.export,
    horizon=20,
    model=d(
        hidden_size=1000,
        inner_hidden_size=1000,
        sparse_mlp_size=400,
        policy_size=400,
        use_smooth_mode=False,
        mode_beta=0.1,
    ),

    policy=d(
        is_terminated_fn=get_timeout_terminate_fn(700),
    ),
    trainer=d(
        holdout_every_n_steps=36,
        rollout_train_env_n_per_step=36,
    ),
)
