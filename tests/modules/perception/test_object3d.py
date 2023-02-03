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
from avstack.geometry import bbox
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


def test_mmdet_3d_perception_kitti():
    try:
        import mmdet3d
    except ModuleNotFoundError as e:
        print("Cannot run mmdet test without the module")
    else:
        detector = perception.object3d.MMDetObjectDetector3D()
        detections = detector(pc, frame=frame, identifier="lidar_objects_3d")
        for det in detections:
            if det.box.t.distance(obj.box.t):
                break
        else:
            raise
        # assert len(detections) == len(labels)


def test_mmdete_3d_perception_nuscenes():
    pass
