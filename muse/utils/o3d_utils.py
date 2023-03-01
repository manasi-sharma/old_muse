import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# for panda rw env, estimation of position
from muse.utils.geometry_utils import world_frame_3D, CoordinateFrame, ee_true_frame_from_state
from attrdict import AttrDict


def get_bbox_axes_from_rgbd(rgb_img, depth_img, pinhole_camera_intrinsic):
    color_raw_filtered = o3d.geometry.Image(rgb_img.astype(np.uint8))
    depth_raw_filtered = o3d.geometry.Image(depth_img)

    rgbd_image_filtered = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw_filtered, depth_raw_filtered)

    pcd_filtered = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_filtered, pinhole_camera_intrinsic)
    #
    # o3d.visualization.draw_geometries([pcd_filtered])
    plane_model, inliers = pcd_filtered.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)


    inlier_cloud = pcd_filtered.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # outlier_cloud = pcd_filtered.select_by_index(inliers, invert=True)

    oriented_bounding_box = inlier_cloud.get_oriented_bounding_box()
    oriented_bounding_box.color = (0, 1, 0)

    center = np.asarray(oriented_bounding_box.center)
    extent = np.asarray(oriented_bounding_box.extent)
    idx_order = np.argsort(extent)  # least to greatest
    width = extent[idx_order[2]]  # biggest
    height = extent[idx_order[1]]  # medium
    depth = extent[idx_order[0]]  # smallest
    ee2cam = oriented_bounding_box.R

    # # # TODO parameterize: this is just for the case of blue square, where cam looks directly at it
    width_axis = ee2cam[:, idx_order[2]].copy()  # x in ee
    height_axis = ee2cam[:, idx_order[1]].copy()  # z in ee
    depth_axis = ee2cam[:, idx_order[0]].copy()  # y in ee

    flip = np.ones(3)  # (width, height, depth)
    if width_axis[0] > 0:  #
        width_axis *= -1
    if height_axis[1] > 0:  # facing the opposite direction to camera y
        height_axis *= -1
    if depth_axis[2] > 0:  # facing away from camera
        depth_axis *= -1
    #
    # box2ee = np.array([[flip[0], 0.,      0      ],
    #                    [0,       0.,      flip[2]],
    #                    [0,       flip[1], 0      ]])
    #
    # cam_frame = world_frame_3D
    # ee_frame = CoordinateFrame(cam_frame, R.from_matrix(ee2cam), center)
    # robot_frame = CoordinateFrame(ee_frame, R.identity(), np.array([0, -0.5, 0.]))

    # new basis in cam frame (orientation of box is same as end effector
    ee_xyz_basis_in_cam_frame = np.column_stack([width_axis, depth_axis, height_axis])
    return inlier_cloud, oriented_bounding_box, ee_xyz_basis_in_cam_frame, center, np.array([width, height, depth])


def estimate_camera_position_subroutine(ee_position, ee_orientation, sensor, pinhole_camera_intrinsic, num_iters=30):
    robot_frame = world_frame_3D
    ee_z_forward_to_y_forward = np.array([0.5, 0.5, 0.5, 0.5])  # rotating a body by this brings it from z forward, x up to y forward, z up frame

    # position of blue square center in ee_frame
    shape_position_in_ee_frame = np.array([0.101, 0.019, 0.0])
    ee_to_shape_rot = R.from_rotvec(np.array([0,0,np.deg2rad(95)]))

    actual_shape_wh = np.array([0.0565, 0.033])

    # this will be "facing" the same way as the end effector

    # end effector orientation, coordinate frame adjusted for consistency
    # ee_delta_orientation = R.from_quat(ee_z_forward_to_y_forward) * R.from_quat(ee_orientation)

    # true_ee_orientation_R = R.from_quat(ee_orientation)  # * R.from_matrix(np.array([[0,0,1], [1,0,0], [0,1,0]]))
    # * R.from_quat(quaternion_base)

    ee_true_frame = ee_true_frame_from_state(ee_position, ee_orientation, robot_frame)
    ee_frame = CoordinateFrame(ee_true_frame, R.from_quat(ee_z_forward_to_y_forward).inv(), np.array([0,0,0]))  # adjusted frame
    shape_frame = CoordinateFrame(ee_frame, ee_to_shape_rot, shape_position_in_ee_frame)

    cam_frame_buffer = []
    obb_buffer = []
    pcd_buffer = []
    shape_buffer = []

    def pipeline():
        dc = sensor.read_state()
        rgb_img = dc.rgb
        depth_img = dc.depth
        # image processing
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
        target_mask = cv2.inRange(hsv_img, np.array([165 * 180 / 360, 150, 20]), np.array([195 * 180 / 360, 255, 255]))
        targ = cv2.bitwise_and(rgb_img, rgb_img, mask=target_mask)

        targ_bw = cv2.cvtColor(targ, cv2.COLOR_BGR2GRAY)
        targ_bin = cv2.inRange(targ_bw, 10, 255)  # binary of fruit

        # erode before finding contours
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        erode_target = cv2.erode(targ_bin, kernel, iterations=1)
        # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # erode_target = cv2.dilate(erode_target, kernel2, iterations=1)

        img_th = cv2.adaptiveThreshold(erode_target, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
        contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask_targ = np.zeros(targ_bin.shape, np.uint8)
        largest_areas = sorted(contours, key=cv2.contourArea)
        if (len(largest_areas) == 1):
            targ_contour = largest_areas[-1]
        else:
            targ_contour = largest_areas[-2]
        cv2.drawContours(mask_targ, [targ_contour], 0, (255, 255, 255), -1)

        # #dilate now
        # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # mask_targ = cv2.dilate(mask_targ, kernel2, iterations=1)
        res = cv2.bitwise_and(targ_bin, targ_bin, mask=mask_targ)
        rgb_img_filtered = cv2.bitwise_and(rgb_img, rgb_img, mask=mask_targ)
        depth_img_filtered = cv2.bitwise_and(depth_img, depth_img, mask=mask_targ)
        target_depth = cv2.inRange(depth_img_filtered, 0, 1500)
        depth_img_filtered = cv2.bitwise_and(depth_img_filtered, depth_img_filtered, mask=target_depth)
        # dif_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_img_filtered, alpha=0.03), cv2.COLORMAP_JET)

        cv2.imshow("RGB", rgb_img_filtered)
        cv2.waitKey(1)

        return rgb_img_filtered.copy(), depth_img_filtered.copy()

    for i in range(num_iters):
        rgb_img_filtered, depth_img_filtered = pipeline()
        box_points, oriented_bounding_box, shape_xyz_basis_in_cam_frame, center, dims = get_bbox_axes_from_rgbd(
            rgb_img_filtered, depth_img_filtered, pinhole_camera_intrinsic)

        cam2shape = np.linalg.inv(shape_xyz_basis_in_cam_frame)
        cam_in_shape = - cam2shape @ center  # cam2ee @ (cam_in_cam - ee_in_cam)

        # frames
        cam_frame = CoordinateFrame(shape_frame, R.from_matrix(shape_xyz_basis_in_cam_frame), cam_in_shape)

        cam_frame_buffer.append(cam_frame)
        obb_buffer.append(oriented_bounding_box)
        shape_buffer.append(dims)
        pcd_buffer.append(box_points)

    volumes = np.array([obb.volume() for obb in obb_buffer])
    shapes = np.stack(shape_buffer)[:, :2]  # N x 3 -> 2
    closeness = np.sqrt(((shapes - actual_shape_wh[None]) ** 2).mean(-1))
    idx = np.argmin(closeness)
    print("closest dimension to %s |  (w/h): %s", (actual_shape_wh, shapes[idx]))

    cam_frame = cam_frame_buffer[idx]
    obb = obb_buffer[idx]
    pcd = pcd_buffer[idx]

    cv2.destroyAllWindows()

    return AttrDict(robot_frame=robot_frame, cam_frame=cam_frame, ee_frame=ee_frame, ee_true_frame=ee_true_frame, shape_frame=shape_frame, obb=obb, pcd=pcd)


# draw coordinate frames in open3d
def get_o3d_geometries_from_coordinate_frames(frame_list, base_frame, size_initial, size_increment=0.):
    line_list = []
    size = size_initial
    for frame in frame_list:
        f2base, f_origin_in_base = CoordinateFrame.transform_from_a_to_b(frame, base_frame)
        v3d = o3d.utility.Vector3dVector(np.stack(
            [f_origin_in_base, f_origin_in_base + size * f2base.as_matrix()[:, 0],
             f_origin_in_base + size * f2base.as_matrix()[:, 1], f_origin_in_base + size * f2base.as_matrix()[:, 2]]))
        idxs = o3d.utility.Vector2iVector(np.array([[0, 1], [0, 2], [0, 3]]))
        lines = o3d.geometry.LineSet(v3d, idxs)
        lines.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0],  # red for ee x
                                                            [0, 1, 0],  # green for ee y
                                                            [0, 0, 1]]))  # blue for ee z
        size += size_increment  # size to distinguish between frames
        line_list.append(lines)

    return line_list

# we want fork -> cam = cam2robot.inv() * fork2robot


def draw_frame(frame, size=0.02):
    cframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    cframe.rotate(frame.c2g_R.as_matrix())
    cframe.translate(frame.c_origin_in_g)
    return cframe


def create_arrow(scale=10):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = scale/10
    cylinder_radius = scale/20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=1,
        cone_height=cone_height,
        cylinder_radius=0.5,
        cylinder_height=cylinder_height)
    return mesh_frame


def get_arrow(origin=[0, 0, 0], end=None, vec=None):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    scale = 10
    rot = np.eye(3)
    # T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = np.linalg.norm(vec) + 1e-11
        angle = np.arccos(1/scale * vec[2])  # vec/|vec| * z
        axis = np.cross(vec, [0,0,1])
        axis = axis / (np.linalg.norm(axis) + 1e-11)
        rot = R.from_rotvec(axis * angle).as_matrix()
    mesh = create_arrow(scale)
    # Create the arrow
    mesh.rotate(rot, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return(mesh)

# def stitch_pcds()