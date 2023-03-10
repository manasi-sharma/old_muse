from attrdict import AttrDict as d

from cfgs.env import polymetis_panda
from cfgs.policy import vr_teleop
from muse.envs.polymetis.polymetis_utils import modify_spec_prms, get_wp_dynamics_fn
from muse.models.model import Model

export = d(
    exp_name='real/collection',
    env_spec=modify_spec_prms(polymetis_panda.export.cls.get_default_env_spec_params(polymetis_panda.export),
                              include_click_state=True) & d(
        wp_dynamics_fn=get_wp_dynamics_fn(fast_dynamics=True),
    ),
    env_train=polymetis_panda.export,
    model=d(cls=Model, ignore_inputs=True),
    policy=vr_teleop.export,
)
