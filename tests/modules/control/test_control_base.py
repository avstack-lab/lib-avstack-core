# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-12
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-08-11
# @Description:
"""

"""

import numpy as np

from avstack.environment.objects import VehicleState
from avstack.geometry import (
    NominalTransform,
    Rotation,
    StandardCoordinates,
    Translation,
    VectorDirMag,
    bbox,
)

def get_object(seed):
    np.random.seed(seed)
    pos_obj = Translation(StandardCoordinates, 10 * np.random.rand(3))
    box_obj = bbox.Box3D(
        [2, 2, 5, [0, 0, 0], 0], NominalTransform
    )  # box in local coordinates
    vel_obj = VectorDirMag(StandardCoordinates, np.random.rand(3))
    acc_obj = VectorDirMag(StandardCoordinates, np.random.rand(3))
    rot_obj = Rotation(StandardCoordinates, np.eye(3))
    ang_obj = VectorDirMag(StandardCoordinates, np.random.rand(3))
    obj = VehicleState("car")
    obj.set(
        0,
        pos_obj,
        box_obj,
        vel_obj,
        acc_obj,
        rot_obj,
        ang_obj,
        T_reference=NominalTransform,
    )
    return obj


# def test_init_control():
#     cfg_path = './avstack/config/modules/control/pid_control.yml'
#     controller = control.init_from_cfg(read_config(cfg_path))
#     assert controller is not None
