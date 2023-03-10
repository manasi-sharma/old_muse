import numpy as np
from attrdict import AttrDict as d
from scipy.spatial.transform import Rotation as R
from configs.fields import Field as F
from muse.policies.sm_teleop_policy import SpaceMouseTeleopPolicy

MAX_VEL = 0.25
MAX_ORI_VEL = 2.5


def get_oculus_to_robot(yaw):
    o2r = np.asarray([[0, 0, 1, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]])
    r2nr = R.from_euler('z', np.deg2rad(yaw)).as_matrix()
    o2r[:3, :3] = r2nr @ o2r[:3, :3]
    return o2r


export = d(
    cls=SpaceMouseTeleopPolicy,
    dt=0.1,
    yaw=-90,  # degrees
    action_name='action',
    gripper_pos_name='gripper_pos',
    gripper_tip_pos_name='ee_position',
    postproc_fn=None,
    use_gripper=True,
    use_click_state=False,
    gripper_action_space='normalized',
    clip_ori_max=None,
    oculus_to_robot_mat_4d=F('yaw', lambda x: get_oculus_to_robot(x)),
)
