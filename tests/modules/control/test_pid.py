# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-06
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""

import numpy as np
import quaternion

from avstack.environment.objects import VehicleState
from avstack.geometry import (
    NominalOriginStandard,
    Rotation,
    Transform,
    Translation,
    Vector,
    bbox,
)
from avstack.geometry import transformations as tforms
from avstack.modules import control
from avstack.modules.planning import Waypoint, WaypointPlan


def test_pid_base():
    # -- p only
    pid = control.PIDBase(1.0, 0, 0)
    c1 = pid(1.0, -5)
    c2 = pid(2.0, -10)
    c3 = pid(3.0, 20, 0, 4.5)
    assert c1 == -5
    assert c2 == -10
    assert c3 == 4.5


def test_vehicle_pid_control():
    box_obj = bbox.Box3D(
        [2, 2, 5, [0, 0, 0], np.quaternion(1)], NominalOriginStandard
    )  # box in local coordinates
    ego_state = VehicleState("car")
    lat = {"K_P": 1.5, "K_D": 0.02, "K_I": 0.01}
    lon = {"K_P": 1.0, "K_D": 0.01, "K_I": 0.05}
    controller = control.vehicle.VehiclePIDController(lat, lon)

    # Initialize
    t = 0
    tmax = 10
    dt = 0.1
    speed_target = 20
    imsize = [200, 200]
    lane_width = 3.7
    yaw = lambda t: 1 / 4 * np.sin(t / 3)

    pos = Translation(np.zeros(3), origin=NominalOriginStandard)
    rot = Rotation(
        tforms.transform_orientation([0, 0, yaw(t)], "euler", "dcm"),
        origin=NominalOriginStandard,
    )
    v = 10
    vel = v * rot.forward_vector
    acc = Vector(np.zeros(3), origin=NominalOriginStandard)
    ang_vel = Vector(np.zeros(3), origin=NominalOriginStandard)

    # Run looop
    pos_all = []
    WP = WaypointPlan()
    while t < tmax:
        t += dt
        # -- reference
        rot = Rotation(
            tforms.transform_orientation([0, 0, yaw(t)], "euler", "dcm"),
            origin=NominalOriginStandard,
        )
        vel_new = v * rot.forward_vector
        pos = pos + Translation(dt * (vel + vel_new) / 2, origin=NominalOriginStandard)
        vel = vel_new

        # -- ego state
        pos_ego = pos + np.random.randn(3)
        vel_ego = vel + np.random.randn(3)
        acc_ego = acc
        rot_ego = Rotation(
            tforms.transform_orientation(
                [0, 0, yaw(t) + np.random.randn(1)], "euler", "dcm"
            ),
            origin=NominalOriginStandard,
        )
        ego_state.set(t, pos_ego, box_obj, vel_ego, acc_ego, rot_ego, ang_vel)

        # -- control
        WP.update(ego_state)
        if WP.needs_waypoint():
            w = Waypoint(Transform(rot, pos), speed_target)
            d = pos.distance(pos_ego)
            WP.push(d, w)
        ctrl = controller(ego_state, WP)
        assert np.sign(ctrl.throttle - ctrl.brake) == np.sign(
            speed_target - np.linalg.norm(vel_ego)
        )
        assert (ctrl.throttle + ctrl.brake) > 0
