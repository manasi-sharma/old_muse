from attrdict import AttrDict as d

from cfgs.env import square
from cfgs.exp_hvs.square import dp_conv_1d
from muse.envs.robosuite.robosuite_env import RobosuiteEnv

export = dp_conv_1d.export & d(
    # this dataset has position actions relabeled..
    dataset='human_square_abs_30k',
    exp_name='hvsBlock3D/posact_{?seed:s{seed}_}b{batch_size}_h{horizon}_{dataset}',
    # change in the env
    env_spec=RobosuiteEnv.get_default_env_spec_params(square.export & d(use_delta=False)),
    env_train=square.export & d(use_delta=False),
    # stuff for normalization of actions, during training
    model=d(
        normalize_actions=True,
        save_action_normalization=True,
    ),
    # policy also needs to normalize actions, during inference
    policy=d(
        policy_out_norm_names=['action']
    ),
)
