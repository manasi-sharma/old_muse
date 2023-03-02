from muse.envs.polymetis.polymetis_panda_env import DualCamera240PolymetisPandaEnv
from attrdict import AttrDict as d

export = d(
    cls=DualCamera240PolymetisPandaEnv,
    home=[0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, -0.7853981633974483],
    camera=d(
        cv_cam_id=-1,
    ),
    ego_camera=d(
        do_depth=False,
        do_decimate=False,
    ),
    use_ego_depth=False,
)
