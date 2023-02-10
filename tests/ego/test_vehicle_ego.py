# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-06
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-08-25
# @Description:
"""

"""


import sys

import numpy as np
import quaternion

from avstack import GroundTruthInformation
from avstack import datastructs as ds
from avstack import ego
from avstack.environment import objects
from avstack.geometry import Rotation, Transform, Translation, Vector, bbox


sys.path.append("tests/")
from utilities import get_image_data, get_lane_lines, get_lidar_data, get_object_global


# def test_level_2_gt_percep_gt_loc():
#     dt = 0.1
#     t_init = 0.0
#     ego_init = get_object_global(1)
#     player = ego.vehicle.Level2GtPerceptionGtLocalization(t_init, ego_init)
#     for i in range(10):
#         ll = get_lane_lines()
#         objs = [ego_init.global_to_local(get_object_global(2))]
#         ground_truth = GroundTruthInformation(
#             i, i * dt, ego_init, objects=objs, lane_lines=ll, lane_id=0
#         )
#         player.tick(
#             frame=i, timestamp=dt * i, data_manager=None, ground_truth=ground_truth
#         )


def run_analysis_vehicle_test(player, n_frames=3):
    data_manager = ds.DataManager()
    iframe = 3
    for i in range(n_frames):
        t = i * 0.1
        pc = get_lidar_data(t=t, frame=1)
        pc.frame = i
        img = get_image_data(t=t, frame=1)
        img.frame = i
        data_manager.push(pc)
        data_manager.push(img)
        tracks, objects = player.tick(frame=i, timestamp=t, data_manager=data_manager)
    return tracks, objects


def test_LidarPerceptionAndTrackingVehicle():
    dt = 0.1
    t_init = 0.0
    ego_init = get_object_global(1)
    try:
        import mmdet3d
    except ModuleNotFoundError as e:
        print("Cannot run vehicle test without the mmdet module")
    else:
        player = ego.vehicle.LidarPerceptionAndTrackingVehicle(t_init, ego_init)
        tracks, objects = run_analysis_vehicle_test(player, n_frames=4)
        assert len(tracks) > 0


def test_LidarCameraPerceptionAndTrackingVehicle():
    dt = 0.1
    t_init = 0.0
    ego_init = get_object_global(1)
    try:
        import mmdet
        import mmdet3d
    except ModuleNotFoundError as e:
        print("Cannot run vehicle test without the mmdet module")
    else:
        player = ego.vehicle.LidarCameraPerceptionAndTrackingVehicle(t_init, ego_init)
        tracks, objects = run_analysis_vehicle_test(player, n_frames=4)
