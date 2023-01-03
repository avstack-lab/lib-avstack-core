# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-11
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-08-08
# @Description:
"""

"""

import os, sys
from avstack.modules import perception

sys.path.append('tests/')
from utilities import get_test_sensor_data

obj, box_calib, lidar_calib, pc, camera_calib, img, box_2d, box_3d = get_test_sensor_data()


def test_mmdet_2d_perception():
    frame = 0
    try:
        import mmdet
    except ModuleNotFoundError as e:
        print('Cannot run mmdet test without the module')
    else:
        detector = perception.object2dfv.MMDetObjectDetector2D()
        detections = detector(frame, img, 'camera_objects_2d')
