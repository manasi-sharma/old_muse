import threading
import time
from collections import namedtuple

import numpy as np
from attrdict import AttrDict
from attrdict.utils import get_with_default

from muse.envs.env_interfaces import VRInterface
from muse.policies.policy import Policy
from muse.utils.torch_utils import cat_any, to_numpy
from muse.utils.transform_utils import fast_euler2quat, quat2mat, rotation_matrix

try:
    import os
    os.environ["LD_LIBRARY_PATH"] = os.getcwd()  # or whatever path you want
    import hid
except ModuleNotFoundError as exc:
    raise ImportError("Unable to load module hid, required to interface with SpaceMouse. \n"
                      "Installation: https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/using_teleoperation_devices.html?highlight=spacemouse") from exc


AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

SPACE_MOUSE_SPEC = {
    "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
    "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "roll": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
    "pitch": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
    "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
}


def to_int16(y1, y2):
    """
    Convert two 8 bit bytes to a signed 16 bit integer.
    Args:
        y1 (int): 8-bit byte
        y2 (int): 8-bit byte
    Returns:
        int: 16-bit integer
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """
    Normalize raw HID readings to target range.
    Args:
        x (int): Raw reading from HID
        axis_scale (float): (Inverted) scaling factor for mapping raw input value
        min_v (float): Minimum limit after scaling
        max_v (float): Maximum limit after scaling
    Returns:
        float: Clipped, scaled input from HID
    """
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x


def convert(b1, b2):
    """
    Converts SpaceMouse message to commands.
    Args:
        b1 (int): 8-bit byte
        b2 (int): 8-bit byte
    Returns:
        float: Scaled value from Spacemouse message
    """
    return scale_to_control(to_int16(b1, b2))


class SpaceMouseInterface:
    """
    A minimalistic driver class for SpaceMouse with HID library.
    Note: Use hid.enumerate() to view all USB human interface devices (HID).
    Make sure SpaceMouse is detected before running the script.
    You can look up its vendor/product id from this method.
    Args:
        vendor_id (int): HID device vendor id
        product_id (int): HID device product id
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling

    See https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/using_teleoperation_devices.html?highlight=spacemouse
    """

    def __init__(self, vendor_id=9583, product_id=50734, pos_sensitivity=1.0, rot_sensitivity=2.0, action_scale=0.08):
        print("Opening SpaceMouse device")
        # print(hid.enumerate())
        # print(vendor_id, product_id)
        self.device = hid.device()
        self.device.open(vendor_id, product_id)  # SpaceMouse

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.action_scale = action_scale

        self.gripper_is_closed = False

        print("Manufacturer: %s" % self.device.get_manufacturer_string())
        print("Product: %s" % self.device.get_product_string())

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._display_controls()

        self.single_click_and_hold = False
        self.elapsed_time = 0

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])


        # launch a new listener thread to listen to SpaceMouse
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Right button", "reset simulation")
        print_command("Left button (hold)", "toggle gripper")
        print_command("Move mouse laterally", "move arm horizontally in x-y plane")
        print_command("Move mouse vertically", "move arm vertically")
        print_command("Twist mouse about an axis", "rotate arm about a corresponding axis")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._control = np.zeros(6)

        self.single_click_and_hold = False
        self.t_last_click = time.time()

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0

    def get_controller_state(self):
        """
        Grabs the current state of the 3D mouse.
        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        dpos = self.control[:3] * 0.005 * self.pos_sensitivity
        roll, pitch, yaw = self.control[3:] * 0.005 * self.rot_sensitivity

        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            reset=self._reset_state,
        )

    def run(self):
        """Listener method that keeps pulling new messages."""

        # t_last_click = -1

        while True:
            d = self.device.read(13)

            if d is not None:

                # readings from 6-DoF sensor
                if d[0] == 1:
                    self.y = convert(d[1], d[2])
                    self.x = convert(d[3], d[4])
                    self.z = convert(d[5], d[6]) * -1.0

                    self.roll = -1.0 * convert(d[11], d[12])
                    self.pitch = convert(d[7], d[8])
                    self.yaw = convert(d[9], d[10])

                    self._control = [
                        self.x,
                        self.y,
                        self.z,
                        self.yaw,
                        self.pitch,
                        self.roll,
                    ]

                # readings from the side buttons

                elif d[0] == 3:

                    # press left button
                    if d[1] == 1:
                        t_click = time.time()
                        self.elapsed_time = t_click - self.t_last_click
                        self.t_last_click = t_click
                        if self.elapsed_time > 0.5:
                            self.single_click_and_hold = True
                            self.gripper_is_closed = not self.gripper_is_closed

                    # release button
                    if d[1] == 0:
                        self.single_click_and_hold = False
                        self._reset_state = 0

                    # right button is for reset
                    if d[1] == 2:
                        t_click = time.time()
                        self.elapsed_time = t_click - self.t_last_click
                        self.t_last_click = t_click
                        if self.elapsed_time > 0.5:
                            self._reset_state = 1
                        #self._enabled = False

                else:
                    if self._reset_state:
                        self.single_click_and_hold = False
                        self._reset_state = 0

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse
        Returns:
            np.array: 6-DoF control value
        """

        return np.array(self._control)

    @property
    def control_gripper(self):
        """
        Maps internal states into gripper commands.
        Returns:
            float: Whether we're using single click and hold or not
        """
        return self.gripper_is_closed

        return None

    def get_action(self):
        if sum(abs(self.control)) > 0.0 or self.control_gripper is not None:
            return self.action_scale * self.control, self.control_gripper, self._reset_state
        else:
            return None, self.control_gripper, self._reset_state


class SpaceMouseTeleopPolicy(Policy):
    """
    The policy uses the model to select actions using the current observation and goal.
    The policy will vary significantly depending on the algorithm.
    If the model is a policy that maps observation/goal directly to actions, then all you have to do is call the model.
    If the model is a dynamics model, the policy will need to be a planner (e.g., random shooting, CEM).
    Having the policy be separate from the model is advantageous since it allows you to easily swap in
    different policies for the same model.
    spacemouse.py
    Interface for 3DConnexion spacemouse controller (code from robosuite).
    Installation:
        1. Give access to spacemouse without root:
           put `SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c62e", MODE="0666"` in /etc/udev/rules.d/50-3dmouse.rules
           sudo udevadm control --reload-rules
        2. Install hdi library: pip install hidapi
    """

    def _init_params_to_attrs(self, params):
        # this will be the pose (euler)
        self.action_name = get_with_default(params, "action_name", "action")
        self.gripper_pos_name = get_with_default(params, "gripper_pos_name", "gripper_pos")
        self.gripper_tip_pos_name = get_with_default(params, "gripper_tip_pos_name", "gripper_tip_pos")

        # remaps B -> mode label
        self.use_click_state = get_with_default(params, "use_click_state", False)

        self.use_gripper = get_with_default(params, "use_gripper", True)
        # default = 0->255
        # normalized = 0->1
        self.gripper_action_space = get_with_default(params, "gripper_action_space", "default")
        # will use clipped delta gripper
        if self.gripper_action_space == "default":
            # max = closed
            self._gripper_max = 250.
        elif self.gripper_action_space == "normalized":
            self._gripper_max = 1.

        # parses obs -> pose 7d
        self.get_pose_from_obs_fn = get_with_default(params, "get_pose_from_obs_fn", lambda obs: cat_any(
            [obs["ee_position"].reshape(-1), fast_euler2quat(obs["ee_orientation_eul"].reshape(-1))],
            dim=-1))
        self.get_gripper_from_obs_fn = get_with_default(params, "get_gripper_from_obs_fn",
                                                        lambda obs: obs[self.gripper_pos_name].reshape(-1))
        self.get_gripper_tip_pose_from_obs_fn = get_with_default(params, "get_gripper_tip_pose_from_obs_fn",
                                                                 lambda obs: cat_any(
                                                                     [obs[self.gripper_tip_pos_name].reshape(-1),
                                                                      fast_euler2quat(
                                                                          obs["ee_orientation_eul"].reshape(-1))],
                                                                     dim=-1))
        # parses obs -> base pose 7d (usually doesn't change, so set this to return a constant)
        self.get_base_pose_from_obs_fn = get_with_default(params, "get_base_pose_from_obs_fn", lambda obs: np.array(
            [0, 0, 0, 0, 0, 0, 1]))  # this will be the pose.


        assert issubclass(type(self._env), VRInterface), "Env must inherit from VR interface!"

        # TODO safenet
        self.tip_safe_bounds = np.asarray(
            get_with_default(params, "tip_safe_bounds", self._env.get_safenet()))
        assert list(self.tip_safe_bounds.shape) == [2, 3], self.tip_safe_bounds.shape

        # TODO conical clipping of orientation
        # if non None, determines the max allowable angle of the end effector from downward-z
        self._clip_ori_max = params << "clip_ori_max"

        super(SpaceMouseTeleopPolicy, self)._init_params_to_attrs(params)

    def reset_policy(self, **kwargs):
        self._done = False
        self._step = 0

    def _init_setup(self):
        super(SpaceMouseTeleopPolicy, self)._init_setup()

        # TODO TESTING
        self.spacemouse = SpaceMouseInterface()
        self.spacemouse.start_control()

        self._done = False
        self._trig_pressed = False

        self.yaw = 0  # pybullet only

    def warm_start(self, model, observation, goal):
        pass

    def _set_robot_orientation(self, obs):
        # For all of them Det == 1 => proper rotation matrices.
        # sets internal pose estimate.
        base_pose = self.get_base_pose_from_obs_fn(obs).reshape(-1)
        self.robot_to_global_rmat = quat2mat(base_pose[3:])[:3, :3]
        self.robot_to_global_mat_4d = np.eye(4)
        self.robot_to_global_mat_4d[:3, :3] = self.robot_to_global_rmat
        self.global_to_robot_mat_4d = np.linalg.inv(
            self.robot_to_global_mat_4d)
        self.global_to_robot_mat_rmat = self.global_to_robot_mat_4d[:3, :3]

    def _read_observation(self, obs):
        # Read Environment Observation
        self.robot_7d_pose = to_numpy(self.get_pose_from_obs_fn(obs).reshape(-1), check=True)
        if self.use_gripper:
            self.gripper = to_numpy(self.get_gripper_from_obs_fn(obs).reshape(-1), check=True)[0]
        else:
            self.gripper = 0
        self.gripper_tip_pose = to_numpy(self.get_gripper_tip_pose_from_obs_fn(obs).reshape(-1), check=True)
        return self.robot_7d_pose, self.gripper, self.gripper_tip_pose

    def get_action(self, model, observation, goal, **kwargs):
        self._set_robot_orientation(observation)

        # Read Sensor TODO TESTING

        delta_action, gripper_action, reset = self.spacemouse.get_action()

        # Read Observation
        robot_pose, curr_gripper, curr_gripper_tip_pose = self._read_observation(observation)
        gripper = self._gripper_max * np.array([gripper_action], dtype=np.float32)  # press toggle gripper

        # TODO find a button for mode labeling!
        curr_click_label = 0.
        '''
        if buttons['B']:
            if self.use_click_state:
                curr_click_label = 1.
            else:
                # make sure this is implemented
                self._env.change_view(delta_yaw=-self.yaw)
                self.yaw = 0
        '''

        if reset:
            # RESET
            self._done = True

        if delta_action is None:
            pos_action = np.zeros(6)
        else:
            pos_action = delta_action

        command = pos_action

        # postprocess
        return self._postproc_fn(model, observation, goal, AttrDict.from_dict({
            self.action_name: np.concatenate([command, gripper]),
            'target': {
                'ee_position': robot_pose[:3] + pos_action[:3],
                'ee_orientation_eul': pos_action[3:],
                self.gripper_pos_name: gripper,
            },
            'policy_type': np.array([255]),  # spacemouse
            'policy_name': np.array(["spacemouse_teleop"]),  # spacemouse
            'policy_switch': np.array([False]),  # spacemouse
            'click_state': np.array([curr_click_label])
        }).leaf_apply(lambda arr: arr[None]))

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self._done
