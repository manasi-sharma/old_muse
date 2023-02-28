from muse.envs.polymetis.polymetis_panda_env import PolymetisPandaEnv
from muse.envs.sensor.camera import USBCamera, RSDepthCamera, RSSeries
from attrdict import AttrDict as d

export = d(
    cls=PolymetisPandaEnv,
    home=[0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, -0.7853981633974483],
    hz=10,
    use_gripper=True,
    franka_ip='172.16.0.1',
    action_space='ee-euler-delta',
    delta_pivot='ground-truth',
    camera=d(
        cls=USBCamera,
        cv_cam_id=-1,
        img_width=320,
        img_height=240,
    ),
    ego_camera=d(
        cls=RSDepthCamera,
        config_json='muse/sandbox/rs_short_range_config_240.json',
        series=RSSeries.d400,
        check_width=320,
        check_height=240,
        do_depth=False,
        do_decimate=False,
    ),
    use_ego_depth=False,
)
