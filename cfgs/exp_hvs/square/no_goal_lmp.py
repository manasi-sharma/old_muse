from attrdict import AttrDict as d

from cfgs.exp_hvs.square import lmp

export = lmp.export & d(
    model=d(
        use_goal=False,
        goal_names=[],
    )
)
