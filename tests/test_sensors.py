# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-09
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-28
# @Description:
"""

"""

import sys

from avstack import sensors
from avstack.datastructs import DataBucket, DataManager


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


def test_sensor_data_bucket():
    SDB = DataBucket(source_identifier="sensor-1")
    S1 = sensors.SensorData(
        frame=2,
        timestamp=10,
        data="a",
        calibration=lidar_calib,
        source_ID=1,
        source_name="sensor",
    )
    S2 = sensors.SensorData(
        frame=3,
        timestamp=11,
        data="b",
        calibration=lidar_calib,
        source_ID=1,
        source_name="sensor",
    )
    S3 = sensors.SensorData(
        frame=1,
        timestamp=9,
        data="c",
        calibration=lidar_calib,
        source_ID=1,
        source_name="sensor",
    )
    S4 = sensors.SensorData(
        frame=0,
        timestamp=8,
        data="d",
        calibration=lidar_calib,
        source_ID=0,
        source_name="sensor",
    )
    for S in [S1, S2, S3, S4]:
        try:
            SDB.push(S)
        except Exception as e:
            if S == S4:
                pass
            else:
                raise e
        else:
            if S == S4:
                raise
    assert len(SDB) == 3
    assert SDB.pop() == S3
    assert SDB.pop() == S1
    assert SDB.pop() == S2


def test_sensor_data_manager():
    SDM = DataManager()
    S1 = sensors.SensorData(
        frame=2,
        timestamp=10,
        data="a",
        calibration=lidar_calib,
        source_ID=1,
        source_name="sensor",
    )
    S2 = sensors.SensorData(
        frame=3,
        timestamp=11,
        data="b",
        calibration=lidar_calib,
        source_ID=1,
        source_name="sensor",
    )
    S3 = sensors.SensorData(
        frame=1,
        timestamp=9,
        data="c",
        calibration=lidar_calib,
        source_ID=1,
        source_name="sensor",
    )
    S4 = sensors.SensorData(
        frame=0,
        timestamp=8,
        data="d",
        calibration=lidar_calib,
        source_ID=0,
        source_name="sensor",
    )
    for S in [S1, S2, S3, S4]:
        SDM.push(S)
    assert SDM.n_buckets == 2
    assert SDM.n_data == 4


def test_lidar_spherical_matrix():
    AE_M = pc.as_spherical_matrix(rate=10, sensor="kitti")


# def test_lidar_data_pseudo_packets():
#     # -- pc is KITTI data
#     PP = pc.as_pseudo_packets(rate=10, sensor='kitti')
