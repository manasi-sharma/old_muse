import numpy as np
from attrdict import AttrDict as d

from cfgs.env import square
from cfgs.exp_hvs.square import dp_conv_1d
from configs.fields import Field as F
from muse.envs.robosuite.robosuite_env import RobosuiteEnv

export = dp_conv_1d.export & d(
    # this dataset has position actions relabeled..
    dataset='human_square_abs_30k',
    exp_name='hvsBlock3D/posact_{?seed:s{seed}_}b{batch_size}_h{horizon}_{dataset}{?use_pose_norm:_pn}',
    # change in the env
    env_spec=RobosuiteEnv.get_default_env_spec_params(square.export & d(use_delta=False)),
    env_train=square.export & d(use_delta=False),
    # stuff for normalization of actions, during training

    # True = use of fixed scale norms (1 for pos, 10 for ori, and 1 for gripper)
    use_pose_norm=True,
    model=d(
        norm_overrides=F('use_pose_norm', lambda pn: (d(action=d(mean=np.zeros(7),
                                                                 std=np.array([1., 1., 1., 10., 10., 10., 1.])))
                         if pn else d())),
        normalize_actions=True,
        save_action_normalization=True,
        action_decoder=d(
            use_tanh_out=False,  # actions are not -1 to 1
        ),
    ),
    # policy also needs to normalize actions, during inference
    policy=d(
        policy_out_norm_names=['action']
    ),
)
