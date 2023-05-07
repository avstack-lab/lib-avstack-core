# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-11
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-08-08
# @Description:
"""

"""

import os
import sys

from avstack.modules import perception


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


def test_mmdet_2d_perception():
    try:
        import mmdet
    except ModuleNotFoundError as e:
        print("Cannot run mmdet test without the module")
    else:
        frame = 0
        model_dataset_pairs = [('fasterrcnn', 'kitti'),
                               ('fasterrcnn', 'cityscapes'),
                               ('fasterrcnn', 'coco-person')]
                               #('rtmdet', 'coco')]
        
        for model, dataset in model_dataset_pairs:
            detector = perception.object2dfv.MMDetObjectDetector2D(model=model, dataset=dataset)
            detections = detector(img, frame=frame, identifier="camera_objects_2d")


def test_jetson_2d_perception():
    try:
        import jetson_inference
    except ModuleNotFoundError as e:
        print("Cannot run jetson test without the jetson module")
    else:
        frame = 0
        models = ["dashcamnet"]
        for model in models:
            detector = perception.object2dfv.JetsonInference2D(model=model)
            import pdb; pdb.set_trace()
            detections = detector(img, frame=frame, identifier="camera_objects_2d")
