# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-11
# @Last Modified by:   spencer@primus
# @Last Modified date: 2022-09-08
# @Description:
"""

"""

import os
import sys

import numpy as np

from avstack import GroundTruthInformation
from avstack.modules import perception


sys.path.append("tests/")
from utilities import get_ego, get_object_global, get_test_sensor_data


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
frame = 0


def test_groundtruth_perception():
    # -- set up ego and objects
    ego_init = get_ego(seed=3)
    obj1 = get_object_global(seed=4)
    obj_local = ego_init.global_to_local(obj1)
    assert obj_local.origin == ego_init.origin == obj1.origin
    assert not np.allclose(obj1.position.vector, obj_local.position.vector)

    # GT information
    frame = timestamp = 0
    ground_truth = GroundTruthInformation(
        frame, timestamp, ego_state=ego_init, objects=[obj1]
    )

    # -- test update
    percep = perception.object3d.GroundTruth3DObjectDetector()
    detections = percep(ground_truth, frame=frame, identifier="percep-1")
    assert np.allclose(detections[0].box.t.vector, obj_local.position.vector)


class LidarMeasurement():
    """To emulate the carla measurements"""
    def __init__(self, raw_data: memoryview) -> None:
        assert isinstance(raw_data, memoryview)
        self.raw_data = raw_data


def run_mmdet(datatype, model, dataset, as_memoryview=False):
    try:
        import mmdet3d
    except ModuleNotFoundError as e:
        print("Cannot run mmdet test without the module")
    else:
        if datatype == 'lidar':
            data = pc
            if as_memoryview:
                data.data = LidarMeasurement(memoryview(data.data))
        elif datatype == 'image':
            data = img
        else:
            raise NotImplementedError(datatype)
        detector = perception.object3d.MMDetObjectDetector3D(model=model, dataset=dataset)
        detections = detector(data, frame=frame, identifier="lidar_objects_3d")
        assert len(detections) > 2


def test_mmdet_3d_pgd_kitti():
    run_mmdet('image', 'pgd', 'kitti')


def test_mmdet_3d_pillars_nuscenes():
    run_mmdet('lidar', 'pointpillars', 'nuscenes')
    

def test_mmdet_3d_pillars_nuscenes_memoryview():
    run_mmdet('lidar', 'pointpillars', 'nuscenes', as_memoryview=True)
