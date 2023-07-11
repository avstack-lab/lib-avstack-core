# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-08-07
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""
import json

import numpy as np

from avstack import calibration
from avstack.geometry.refchoc import GlobalOrigin3D, ReferenceFrame


def test_encode_decode_calibation():
    ref = ReferenceFrame(
        x=np.random.rand(3), q=np.quaternion(1), reference=GlobalOrigin3D
    )
    calib_1 = calibration.Calibration(ref)
    calib_2 = json.loads(calib_1.encode(), cls=calibration.CalibrationDecoder)
    assert calib_1.allclose(calib_2)


def test_encode_decode_camera_calibation():
    ref = ReferenceFrame(
        x=np.random.rand(3), q=np.quaternion(1), reference=GlobalOrigin3D
    )
    calib_1 = calibration.CameraCalibration(
        ref, P=np.random.rand(3, 4), img_shape=(400, 500)
    )
    calib_2 = json.loads(calib_1.encode(), cls=calibration.CalibrationDecoder)
    assert calib_1.allclose(calib_2)
