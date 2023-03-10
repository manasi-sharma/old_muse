from attrdict import AttrDict as d

from cfgs.exp_hvs.collection import vr
from cfgs.policy import sm_teleop

export = vr.export & d(
    policy=sm_teleop.export,
)
