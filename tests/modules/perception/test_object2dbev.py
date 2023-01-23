# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-08
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-08-09
# @Description:
"""

"""
import os
import sys

import numpy as np

from avstack import sensors
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
    box_2d,
    box_3d,
) = get_test_sensor_data()
pc_bev = LidarData(pc.timestamp, pc.frame, pc.data[:, [1, 2]], lidar_calib, 100)

alg_ID = 0
alg_name = "detector-1"


def test_lidar_2d_centroid_detector():
    idx_frame = 100
    D = object2dbev.Lidar2dCentroidDetector()
    dets = D(pc_bev, alg_name)
    assert isinstance(dets, list)
    assert isinstance(dets[0], CentroidDetection)
