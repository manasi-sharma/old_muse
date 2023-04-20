from attrdict import AttrDict as d

from cfgs.env import square
from cfgs.exp_hvs.square import vis_bc_rnn
from muse.envs.robosuite.robosuite_env import RobosuiteEnv

export = vis_bc_rnn.export & d(
    # this dataset has position actions relabeled, but using 6D rotations..
    dataset='human_square_abs6d_30k_eimgs',
    exp_name='hvsBlock3D/posact_{?seed:s{seed}_}b{batch_size}_h{horizon}_{dataset}',
    # change in the env
    env_spec=RobosuiteEnv.get_default_env_spec_params(square.export &
                                                      d(use_delta=False, use_rot6d=True, imgs=True, ego_imgs=True)),
    env_train=square.export & d(use_delta=False, use_rot6d=True, imgs=True, ego_imgs=True),
    # NO ACTION NORMALIZING
    model=d(
        action_decoder=d(
            use_tanh_out=False,  # actions are not -1 to 1
        ),
    ),
)
