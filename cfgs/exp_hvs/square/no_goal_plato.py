from attrdict import AttrDict as d

from cfgs.exp_hvs.square import plato

export = plato.export & d(
    model=d(
        use_goal=False,
        goal_names=[],
    ),
    dataset_train=d(
        sample_goals=False,
    ),
    dataset_holdout=d(
        sample_goals=False,
    ),
)
