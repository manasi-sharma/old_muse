from attrdict import AttrDict as d

from cfgs.env import square
from cfgs.exp_hvs.square import bc_rnn
from configs.fields import Field as F
from muse.envs.robosuite.robosuite_env import RobosuiteEnv
from muse.utils.loss_utils import get_default_nll_loss_fn, get_default_mae_action_loss_fn, mse_err_fn

export = bc_rnn.export & d(
    # this dataset has position actions relabeled..
    dataset='human_square_abs_30k',
    exp_name='hvsBlock3D/posact_{?seed:s{seed}_}b{batch_size}_h{horizon}_{dataset}',
    # change in the env
    env_spec=RobosuiteEnv.get_default_env_spec_params(square.export & d(use_delta=False)),
    env_train=square.export & d(use_delta=False),
    # stuff for normalization of actions, during training
    model=d(
        save_action_normalization=True,
        loss_fn=F('use_policy_dist',
                  lambda x: get_default_nll_loss_fn(['action'], policy_out_norm_names=['action'], vel_act=True)
                  if x else
                  get_default_mae_action_loss_fn(['action'], max_grab=None,
                                                 err_fn=mse_err_fn, vel_act=True,
                                                 policy_out_norm_names=['action'])
                  )
    ),
    # policy also needs to normalize actions, during inference
    policy=d(
        policy_out_norm_names=['action']
    ),
)
