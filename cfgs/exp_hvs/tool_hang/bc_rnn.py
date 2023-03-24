from attrdict import AttrDict as d

from cfgs.env import tool_hang
from configs.fields import Field as F
from muse.envs.robosuite.robosuite_env import RobosuiteEnv

# use this as a starting point.
from cfgs.exp_hvs.square import bc_rnn as sq_bc_rnn

from muse.policies.memory_policy import get_timeout_terminate_fn

export = sq_bc_rnn.export & d(
    dataset='human_tool_hang_100k',
    env_spec=RobosuiteEnv.get_default_env_spec_params(tool_hang.export),
    env_train=tool_hang.export,
    model=d(
        action_decoder=d(
            hidden_size=1000,
        )
    ),

    policy=d(
        is_terminated_fn=get_timeout_terminate_fn(700),
    ),
    trainer=d(
        holdout_every_n_steps=36,
        rollout_train_env_n_per_step=36,
    ),
)
