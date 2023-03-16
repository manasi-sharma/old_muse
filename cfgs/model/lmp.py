from attrdict import AttrDict as d

from muse.models.bc.lmp.lmp_grouped import LMPGroupedModel

export = d(
    cls=LMPGroupedModel,
    plan_size=16,
    hidden_size=64,


)