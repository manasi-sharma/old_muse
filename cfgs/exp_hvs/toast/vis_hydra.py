from attrdict import AttrDict as d

from cfgs.exp_hvs.coffee import vis_hydra as coffee_vis_hydra

export = coffee_vis_hydra.export & d(
    dataset='mode_real_fast_toast-bread-v2_eimgs_100ep',
)
