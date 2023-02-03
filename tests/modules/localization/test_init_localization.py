# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-11
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""

from copy import copy, deepcopy

import numpy as np
import quaternion

from avstack import GroundTruthInformation
from avstack.environment.objects import VehicleState
from avstack.geometry import NominalOriginStandard, bbox
from avstack.modules import localization


def test_init_ground_truth():
    # -- set up ego
    pos = np.random.rand(3)
    box = bbox.Box3D([2, 2, 5, [0, 0, 0], np.quaternion(1)], NominalOriginStandard)
    vel = np.random.rand(3)
    acc = np.random.rand(3)
    rot = np.quaternion()
    ang = np.random.rand(3)
    ego_init = VehicleState("car")
    ego_init.set(0, pos, box, vel, acc, rot, ang, origin=NominalOriginStandard)

    # -- set up localizer
    localizer = localization.GroundTruthLocalizer(rate=10, t_init=0, ego_init=ego_init)
    assert localizer.position.allclose(pos)
    assert localizer.attitude.allclose(rot)

    # -- test update
    ego_state = deepcopy(ego_init)
    ego_state.position += 10
    frame = timestamp = 0
    ground_truth = GroundTruthInformation(frame, timestamp, ego_state, objects=None)
    ego_loc = localizer(timestamp, ground_truth=ground_truth)
    assert ego_loc.position.allclose(ego_state.position)
    assert ego_loc.velocity.allclose(vel)
    dt = 1e-5
    ego_loc2 = localizer(timestamp + dt, ground_truth=ground_truth)
    assert ego_loc2.velocity.allclose(ego_loc.velocity + ego_loc.acceleration * dt)
