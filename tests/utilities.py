# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-08-08
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-27
# @Description:
"""

"""
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
from avstack.environment.objects import VehicleState
from avstack.geometry import (
    Acceleration,
    AngularVelocity,
    Attitude,
    Box3D,
    GlobalOrigin3D,
    PointMatrix3D,
    Position,
    ReferenceFrame,
    Vector,
    Velocity,
    q_stan_to_cam,
)
from avstack.geometry import transformations as tforms
from avstack.modules.perception import detections
from avstack.sensors import ImageData, LidarData, RadarDataRazelRRT


# -- calibration data
ref_lidar = ReferenceFrame(
    x=np.array([0, 0, 1.73]), q=np.quaternion(1), reference=GlobalOrigin3D
)
ref_camera = ReferenceFrame(
    x=np.array([0.27, 0.06, 1.65]), q=q_stan_to_cam, reference=GlobalOrigin3D
)
P_cam = np.array(
    [
        [7.215377000000e02, 0.000000000000e00, 6.095593000000e02, 4.485728000000e01],
        [0.000000000000e00, 7.21537000000e02, 1.728540000000e02, 2.163791000000e-01],
        [0.000000000000e00, 0.000000000000e00, 1.000000000000e00, 2.745884000000e-03],
    ]
)

img_shape = (375, 1242, 3)
camera_calib = CameraCalibration(ref_camera, P_cam, img_shape)
box_calib = Calibration(ref_camera)
lidar_calib = LidarCalibration(ref_lidar)
radar_calib = RadarCalibration(ref_lidar)

KITTI_data_dir = os.path.join(os.getcwd(), "data/test_data/object/training")


def get_lane_lines():
    pt_pairs_left = [(i, 4) for i in range(20)]
    pt_pairs_right = [(i + 1, -3) for i in range(20)]
    pts_left = [Vector([x, y, 0], GlobalOrigin3D) for x, y in pt_pairs_left]
    lane_left = detections.LaneLineInSpace(pts_left)
    pts_right = [Vector([x, y, 0], GlobalOrigin3D) for x, y in pt_pairs_right]
    lane_right = detections.LaneLineInSpace(pts_right)
    return [lane_left, lane_right]


def get_ego(seed, reference=ref_camera):
    np.random.seed(seed)
    rot = Attitude(q_stan_to_cam, reference)
    pos = Position(np.random.rand(3), reference)
    hwl = [1, 2, 4]
    box = Box3D(pos, rot, hwl)  # box in local coordinates
    vel = Velocity(np.random.rand(3), reference)
    acc = Acceleration(np.random.rand(3), reference)
    ang = AngularVelocity(np.quaternion(1), reference)
    ego_init = VehicleState("car")
    ego_init.set(0, pos, box, vel, acc, rot, ang)
    return ego_init


def get_object_global(seed, reference=ref_camera):
    np.random.seed(seed)
    pos_obj = Position(10 * np.random.rand(3), reference)
    rot_obj = Attitude(q_stan_to_cam, reference)
    box_obj = Box3D(pos_obj, rot_obj, [2, 2, 5])  # box in local coordinates
    vel_obj = Velocity(10 * np.random.rand(3), reference)
    acc_obj = Acceleration(np.random.rand(3), reference)
    ang_obj = AngularVelocity(np.quaternion(1), reference)
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
    rad[:, :3] = tforms.matrix_cartesian_to_spherical(
        rad[:, :3]
    )  # change to spherical coordinates
    rad[:, 3] = 0  # range rate set to zero artificially...
    rad = RadarDataRazelRRT(t, frame, rad, radar_calib, radar_ID)
    return rad


def get_test_sensor_data(frame=1000, reference=ref_camera):
    sys.path.append(KITTI_data_dir)
    obj = VehicleState("car", ID=1)

    # -- vehicle data
    t = 0
    p = [6.27, -1.45, 14.55]
    position = Position(p, reference=reference)
    yaw = 3.09
    h = 1.47
    w = 1.77
    l = 4.49
    attitude = Attitude(
        tforms.transform_orientation([0, 0, yaw], "euler", "quat"), reference=reference
    )
    box_3d = Box3D(position, attitude, [h, w, l])
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
    box_2d = box_3d.project_to_2d_bbox(camera_calib)
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
