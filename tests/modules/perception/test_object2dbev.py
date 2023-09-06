# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-08
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-08-09
# @Description:
"""

"""
import sys

from avstack.geometry import GlobalOrigin3D
from avstack.modules.perception import object2dbev
from avstack.modules.perception.detections import CentroidDetection
from avstack.sensors import LidarData


sys.path.append("tests/")
from utilities import get_test_sensor_data


(
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
) = get_test_sensor_data()
pc_bev = LidarData(pc.timestamp, pc.frame, pc.data[:, [1, 2]], lidar_calib, 100)

alg_ID = 0
alg_name = "detector-1"


def test_lidar_2d_centroid_detector():
    platform = GlobalOrigin3D
    D = object2dbev.Lidar2dCentroidDetector()
    dets = D(pc_bev, platform, alg_name)
    assert isinstance(dets, list)
    assert isinstance(dets[0], CentroidDetection)
