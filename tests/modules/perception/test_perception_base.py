# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-09
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-08-11
# @Description:
"""

"""

import os
import sys

import numpy as np

from avstack.calibration import Calibration, CameraCalibration
from avstack.datastructs import DataManager
from avstack.geometry import (
    Box3D,
    CameraCoordinates,
    Rotation,
    StandardCoordinates,
    Transform,
    Translation,
    bbox,
)
from avstack.modules import perception
from avstack.objects import VehicleState
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


def test_percep_base():
    percep = perception.object2dbev.Lidar2dCentroidDetector()
    output = percep(pc_bev, "lidar-detector")
