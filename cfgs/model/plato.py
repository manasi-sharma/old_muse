from attrdict import AttrDict as d

from cfgs.model import lmp
from configs.fields import Field as F
from muse.models.bc.lmp.plato import PLATO

export = lmp.export & d(
    exp_name='_{?use_goal::ng}plato-l2-p{plan_size}_hs{hidden_size}_bt{beta}_btp{beta_pre}',

    cls=PLATO,



    # goal should be passed in by dataset, we don't want to separately load it.
    use_last_state_goal=False,

    # weighting of interaction loss
    do_pre_policy=True,
    beta_pre=1.,

    # other macros to use
    object_names=['object'],

    # prior only sees object states.
    prior=d(
        model_inputs=F(['../object_names', '../goal_names'], lambda s, g: s + g),
    ),
)