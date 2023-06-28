# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-12
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""

import numpy as np

from avstack.geometry import GlobalOrigin3D, Pose, Position, Attitude
from avstack.modules import planning


def test_waypoints():
    WPP = planning.WaypointPlan()
    assert WPP.needs_waypoint()
    t_rot = Attitude(np.quaternion(1), GlobalOrigin3D)
    t_pnt = Position(5 + np.random.rand(3), GlobalOrigin3D)
    target_point = Pose(t_pnt, t_rot)
    target_speed = 10
    wp = planning.Waypoint(target_point, target_speed)
    distance = t_pnt.norm()
    WPP.push(distance, wp)

    assert len(WPP) == 1
    assert not WPP.needs_waypoint()
    wp_1 = WPP.top()[1]
    wp_2 = WPP.pop()[1]
    assert wp == wp_1
    assert wp == wp_2
