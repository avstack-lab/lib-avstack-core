import os
import sys

import numpy as np
from cv2 import imread

from avstack.calibration import (
    Calibration,
    CameraCalibration,
    LidarCalibration,
    RadarCalibration,
)
from avstack.geometry import (
    BoundingBox3D,
    BoxSize,
    FrameTransform,
    PointMatrix3D,
    Pose,
    Rotation,
    Transform,
    TransformManager,
    Vector,
    WorldFrame,
    conversions,
    q_stan_to_cam,
)
from avstack.modules.perception import detections
from avstack.objects import VehicleState
from avstack.sensors import ImageData, LidarData, RadarDataRazelRRT


# -- calibration data
tm = TransformManager()

lidar_tform = Transform(x=np.array([0, 0, 1.73]), q=np.quaternion(1))
lidar_frame = FrameTransform(
    from_frame="world", to_frame="lidar", transform=lidar_tform
)
tm.add_transform(lidar_frame)

camera_tform = Transform(x=np.array([0.27, 0.06, 1.65]), q=q_stan_to_cam)
camera_frame = FrameTransform(
    from_frame="world", to_frame="camera", transform=lidar_tform
)
tm.add_transform(camera_frame)

P_cam = np.array(
    [
        [7.215377000000e02, 0.000000000000e00, 6.095593000000e02, 4.485728000000e01],
        [0.000000000000e00, 7.21537000000e02, 1.728540000000e02, 2.163791000000e-01],
        [0.000000000000e00, 0.000000000000e00, 1.000000000000e00, 2.745884000000e-03],
    ]
)

img_shape = (375, 1242, 3)
camera_calib = CameraCalibration(camera_frame, P_cam, img_shape)
box_calib = Calibration(camera_frame)
lidar_calib = LidarCalibration(lidar_frame)
radar_calib = RadarCalibration(
    lidar_frame, fov_horizontal=np.pi, fov_vertical=np.pi / 2
)

KITTI_data_dir = os.path.join(os.getcwd(), "data/test_data/object/training")


def get_lane_lines():
    pt_pairs_left = [(i, 4) for i in range(20)]
    pt_pairs_right = [(i + 1, -3) for i in range(20)]
    pts_left = [Vector([x, y, 0], WorldFrame) for x, y in pt_pairs_left]
    lane_left = detections.LaneLineInSpace(pts_left)
    pts_right = [Vector([x, y, 0], WorldFrame) for x, y in pt_pairs_right]
    lane_right = detections.LaneLineInSpace(pts_right)
    return [lane_left, lane_right]


def get_ego(seed, reference=camera_frame):
    np.random.seed(seed)
    rot = Rotation(q_stan_to_cam, reference)
    pos = Vector(np.random.rand(3), reference)
    hwl = [1, 2, 4]
    pose = Pose(pos, rot)
    box_size = BoxSize(hwl)
    box = BoundingBox3D(pose, box_size)  # box in local coordinates
    vel = Vector(np.random.rand(3), reference)
    acc = Vector(np.random.rand(3), reference)
    ang = Vector(np.zeros(3), reference)
    ego_init = VehicleState("car")
    ego_init.set(0, pos, box, vel, acc, rot, ang)
    return ego_init


def get_object_global(seed, reference=camera_frame):
    np.random.seed(seed)
    pos_obj = Vector(10 * np.random.rand(3), reference)
    rot_obj = Rotation(q_stan_to_cam, reference)
    box_obj = BoundingBox3D(pos_obj, rot_obj, [2, 2, 5])  # box in local coordinates
    vel_obj = Vector(10 * np.random.rand(3), reference)
    acc_obj = Vector(np.random.rand(3), reference)
    ang_obj = Vector(np.zeros(3), reference)
    obj = VehicleState("car")
    obj.set(0, pos_obj, box_obj, vel_obj, acc_obj, rot_obj, ang_obj)
    return obj


def get_object_local(ego, seed):
    obj = get_object_global(seed)
    obj.change_reference(ego, inplace=True)
    return obj


def get_lidar_data(t, frame, lidar_ID=0):
    pc_fname = os.path.join(KITTI_data_dir, "velodyne", "%06d.bin" % frame)
    pc = PointMatrix3D(
        np.fromfile(pc_fname, dtype=np.float32).reshape((-1, 4)), lidar_calib
    )
    pc = LidarData(t, frame, pc, lidar_calib, lidar_ID)
    return pc


def get_image_data(t, frame, camera_ID=0):
    img_fname = os.path.join(KITTI_data_dir, "image_2", "%06d.png" % frame)
    img = imread(img_fname)[:, :, ::-1]
    img = ImageData(t, frame, img, camera_calib, camera_ID)
    return img


def get_radar_data(t, frame, radar_ID=0):
    pc = get_lidar_data(t=t, frame=frame)
    nrows = 50
    rad = pc.data[np.random.randint(pc.data.shape[0], size=nrows), :]
    rad[:, :3] = conversions.matrix_cartesian_to_spherical(
        rad[:, :3]
    )  # change to spherical coordinates
    rad[:, 3] = 0  # range rate set to zero artificially...
    rad = RadarDataRazelRRT(t, frame, rad, radar_calib, radar_ID)
    return rad


def get_test_sensor_data(frame=1000, reference=camera_frame):
    sys.path.append(KITTI_data_dir)
    obj = VehicleState("car", ID=1)

    # -- vehicle data
    t = 0
    p = [6.27, -1.45, 14.55]
    position = Vector(p, reference=reference)
    yaw = 3.09
    h = 1.47
    w = 1.77
    l = 4.49
    attitude = Rotation(
        conversions.transform_orientation([0, 0, yaw], "euler", "quat"),
        reference=reference,
    )
    pose = Pose(position, attitude)
    box_size = BoxSize(h, w, l)
    box_3d = BoundingBox3D(pose, box_size)
    vel = acc = ang = None
    obj.set(
        t=t,
        position=position,
        box=box_3d,
        velocity=vel,
        acceleration=acc,
        attitude=attitude,
        angular_velocity=ang,
    )

    # -- sensor data
    box_2d = camera_calib.project_3d_box_to_2d(box_3d)
    pc = get_lidar_data(t, frame)
    img = get_image_data(t, frame)
    rad = get_radar_data(t, frame)

    return (
        obj,
        box_calib,
        lidar_calib,
        pc,
        camera_calib,
        img,
        radar_calib,
        rad,
        box_2d,
        box_3d,
    )
