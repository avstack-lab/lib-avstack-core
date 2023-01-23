# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-23
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-28
# @Description:
"""

"""

import numpy as np

from avstack.geometry import StandardCoordinates, bbox
from avstack.modules import tracking
from avstack.objects import VehicleState


def get_ego():
    pos = np.random.rand(3)
    box = bbox.Box3D(
        [1, 2, 4, [0, 0, 0], 0], StandardCoordinates
    )  # box in local coordinates
    vel = np.random.rand(3)
    acc = np.random.rand(3)
    rot = np.eye(3)
    ang = np.random.rand(3)
    ego_init = VehicleState("car")
    ego_init.set(0, pos, box, vel, acc, rot, ang, coordinates=StandardCoordinates)
    return ego_init


def get_object(ego):
    pos_obj = ego.position + 10 * np.random.rand(3)
    box_obj = bbox.Box3D(
        [2, 2, 5, [0, 0, 0], 0], StandardCoordinates
    )  # box in local coordinates
    vel_obj = np.random.rand(3)
    acc_obj = np.random.rand(3)
    rot_obj = np.eye(3)
    ang_obj = np.random.rand(3)
    obj = VehicleState("car")
    obj.set(
        0,
        pos_obj,
        box_obj,
        vel_obj,
        acc_obj,
        rot_obj,
        ang_obj,
        coordinates=StandardCoordinates,
    )
    return obj


# def test_tracking_class():
#     ego = get_ego()
#     obj1 = get_object(ego)
#     timestamp = frame = ID = score= 0
#     obj_type = 'Car'
#     n_updates = 10
#     COORD = StandardCoordinates
#     tracking.Track3D(timestamp, frame, ID, obj_type, COORD,
#         obj1.position, obj1.velocity, obj1.acceleration,
#         obj1.box.h, obj1.box.w, obj1.box.l, obj1.box.yaw, n_updates, score)
