import sys

import numpy as np

from avstack.geometry import GlobalOrigin3D, Position, ReferenceFrame
from avstack.modules.perception.fov_estimator import FastRayTraceBevLidarFovEstimator


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
frame = 0


def make_far_reference():
    return ReferenceFrame(
        x=np.array([100, 100, 0]), q=np.quaternion(1), reference=GlobalOrigin3D
    )


def test_fast_raytrace():
    pc_local = pc

    # run fov estimator in a simple frame
    fov_estimator = FastRayTraceBevLidarFovEstimator()
    fov_local = fov_estimator(pc_local, in_global=False)
    pt_local = Position(
        np.array([-20, 0, 0], dtype=float), reference=pc_local.reference
    )
    assert fov_local.check_point(pt_local.x)

    # convert the fov frame using change reference
    pc_far = pc.change_calibration(make_far_reference(), inplace=False)
    pt_far = pt_local.change_reference(pc_far.reference, inplace=False)
    fov_local_as_far = fov_local.change_reference(pc_far.reference, inplace=False)
    assert fov_local_as_far.check_point(pt_far.x)

    # run fov estimator in a different frame
    fov_far = fov_estimator(pc_far, in_global=False)
    assert fov_far.check_point(pt_far.x)


def test_slow_raytrace():
    pass
