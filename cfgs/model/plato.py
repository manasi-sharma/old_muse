from attrdict import AttrDict as d

from muse.models.bc.lmp.plato_grouped import PLATOGroupedModel

export = d(
    cls=PLATOGroupedModel,
    plan_size=16,
    hidden_size=64,

)