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
import quaternion
from cv2 import imread

from avstack import transformations as tforms
from avstack.calibration import Calibration, CameraCalibration
from avstack.geometry import (
    Box3D,
    NominalOriginCamera,
    NominalOriginStandard,
    Origin,
    Rotation,
    Transform,
    Translation,
    q_cam_to_stan,
    q_stan_to_cam,
)
from avstack.modules.perception import detections
from avstack.objects import VehicleState
from avstack.sensors import ImageData, LidarData


# -- calibration data
Tr_lid = Translation([0, 0, 1.73], origin=NominalOriginStandard)
Tr_cam = Translation([-0.06, -1.65, 0], origin=NominalOriginCamera)
P_cam = np.array(
    [
        [7.215377000000e02, 0.000000000000e00, 6.095593000000e02, 4.485728000000e01],
        [0.000000000000e00, 7.21537000000e02, 1.728540000000e02, 2.163791000000e-01],
        [0.000000000000e00, 0.000000000000e00, 1.000000000000e00, 2.745884000000e-03],
    ]
)
CameraOrigin = Origin(Tr_cam.vector, q_stan_to_cam)
LidarOrigin = Origin(Tr_lid.vector, np.eye(3))

img_shape = (375, 1242, 3)
camera_calib = CameraCalibration(CameraOrigin, P_cam, img_shape)
box_calib = Calibration(CameraOrigin)
lidar_calib = Calibration(LidarOrigin)

KITTI_data_dir = os.path.join(os.getcwd(), "data/test_data/object/training")


def get_lane_lines():
    pt_pairs_left = [(i, 4) for i in range(20)]
    pt_pairs_right = [(i + 1, -3) for i in range(20)]
    pts_left = [
        Translation([x, y, 0], origin=NominalOriginStandard) for x, y in pt_pairs_left
    ]
    lane_left = detections.LaneLineInSpace(pts_left)
    pts_right = [
        Translation([x, y, 0], origin=NominalOriginStandard) for x, y in pt_pairs_right
    ]
    lane_right = detections.LaneLineInSpace(pts_right)
    return [lane_left, lane_right]


def get_ego(seed, frame=CameraOrigin):
    np.random.seed(seed)
    q = q_stan_to_cam  # np.quaternion(1)
    pos = np.random.rand(3)
    box = Box3D([1, 2, 4, pos, q], frame)  # box in local coordinates
    vel = np.random.rand(3)
    acc = np.random.rand(3)
    rot = np.eye(3)
    ang = np.random.rand(3)
    ego_init = VehicleState("car")
    ego_init.set(0, pos, box, vel, acc, rot, ang, origin=frame)
    return ego_init


def get_object_global(seed):
    np.random.seed(seed)
    q = q_stan_to_cam  # np.quaternion(1)
    pos_obj = 10 * np.random.rand(3)
    box_obj = Box3D([2, 2, 5, pos_obj, q], CameraOrigin)  # box in local coordinates
    vel_obj = 10 * np.random.rand(3)
    acc_obj = np.random.rand(3)
    rot_obj = np.eye(3)
    ang_obj = np.random.rand(3)
    obj = VehicleState("car")
    obj.set(
        0, pos_obj, box_obj, vel_obj, acc_obj, rot_obj, ang_obj, origin=CameraOrigin
    )
    return obj


def get_object_local(ego, seed):
    return ego.global_to_local(get_object_global(seed))


def get_lidar_data(t, frame, lidar_ID=0):
    pc_fname = os.path.join(KITTI_data_dir, "velodyne", "%06d.bin" % frame)
    pc = np.fromfile(pc_fname, dtype=np.float32).reshape((-1, 4))
    pc = LidarData(t, frame, pc, lidar_calib, lidar_ID)
    return pc


def get_image_data(t, frame, camera_ID=0):
    img_fname = os.path.join(KITTI_data_dir, "image_2", "%06d.png" % frame)
    img = imread(img_fname)[:, :, ::-1]
    img = ImageData(t, frame, img, camera_calib, camera_ID)
    return img


def get_test_sensor_data():
    sys.path.append(KITTI_data_dir)
    obj = VehicleState("car", ID=1)

    # -- vehicle data
    t = 0
    frame = 1000
    p = [6.27, -1.45, 14.55]
    position = Translation(p, origin=CameraOrigin)
    yaw = 3.09
    h = 1.47
    w = 1.77
    l = 4.49
    q = tforms.transform_orientation([0, 0, yaw], "euler", "quat")
    box_3d = Box3D([h, w, l, position, q], CameraOrigin)
    vel = acc = rot = ang = None
    obj.set(t, position, box_3d, vel, acc, rot, ang, origin=CameraOrigin)

    # -- sensor data
    box_2d = box_3d.project_to_2d_bbox(camera_calib)
    pc = get_lidar_data(t, frame)
    img = get_image_data(t, frame)

    return obj, box_calib, lidar_calib, pc, camera_calib, img, box_2d, box_3d
