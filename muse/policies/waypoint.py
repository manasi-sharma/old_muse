import numpy as np

from muse.utils.geometry_utils import CoordinateFrame, world_frame_3D


class Waypoint:
    """
    Flexible specification of absolute / relative waypoints
    """

    def __init__(self, pose, gripper, timeout=np.inf,
                 relative_to_parent=True,
                 relative_to_robot=-1,
                 relative_to_object=-1,
                 relative_pos=True,
                 relative_ori=True,
                 relative_gripper=False,
                 check_reach=True,
                 max_pos_vel=None,
                 max_ori_vel=None,):
        # if False, will use the last state, not the last waypoint of the given robot/object
        self.relative_to_parent = relative_to_parent
        # ... -1 = robot0, 0 = None, 1 = object0 ... ignored if relative_to_parent == True
        self.relative_to_robot = relative_to_robot
        self.relative_to_object = relative_to_object
        # False = absolute in this dimension
        self.relative_pos = relative_pos
        self.relative_ori = relative_ori
        self.relative_gripper = relative_gripper

        assert relative_to_robot < 0 or relative_to_object < 0, "Cannot be relative to both..."
        if relative_to_parent:
            assert relative_to_robot >=0 or relative_to_object >=0, "If relative to parent, must specify either robot or object"
        if relative_to_object >=0 or relative_to_robot >=0:
            assert relative_pos or relative_ori or relative_gripper, "Must be relative to something..."

        self._base_pose = pose.copy()
        self._base_gripper = gripper
        self.timeout = timeout
        self.check_reach = check_reach  # will check reached, otherwise wait til timeout

        self.max_pos_vel = max_pos_vel  # policy can choose to use this to limit vel
        self.max_ori_vel = max_ori_vel  # policy can choose to use this to limit vel

        self.cf = None
        self.gripper = None

    def update(self, parent, robot_poses, object_poses, gripper):
        # get the source
        if self.relative_to_robot < 0 and self.relative_to_object < 0:
            # don't keep creating things if not relative to anything.
            self.cf = CoordinateFrame.from_pose(self._base_pose, world_frame_3D) if self.cf is None else self.cf
            self.gripper = self._base_gripper
        else:
            if self.relative_to_parent:
                rel_source_pose, rel_gripper = parent.pose_and_gripper  # last cf,gripper of the parent
            else:
                rel_source_pose = robot_poses[self.relative_to_robot] if self.relative_to_robot >= 0 else object_poses[self.relative_to_object]
                rel_gripper = gripper  # last gripper

            rel_source_pose = rel_source_pose.copy()  # since we will be modifying
            if not self.relative_pos:
                rel_source_pose[:3] = 0
            if not self.relative_ori:
                rel_source_pose[3:6] = 0
            if not self.relative_gripper:
                rel_gripper = 0

            rel_source = CoordinateFrame.from_pose(rel_source_pose, world_frame_3D)

            self.cf = CoordinateFrame.from_pose(self._base_pose, rel_source)
            self.gripper = self._base_gripper + rel_gripper

        return self.pose_and_gripper

    @property
    def pose_and_gripper(self):
        return self.cf.as_pose(world_frame_3D), self.gripper

    @property
    def max_pos_ori_vel(self):
        return self.max_pos_vel, self.max_ori_vel