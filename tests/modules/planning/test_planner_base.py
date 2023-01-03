# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-12
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""

import numpy as np
import quaternion
from avstack.modules import planning
from avstack.geometry import Translation, Rotation, Transform, NominalOriginStandard


def test_waypoints():
    WPP = planning.WaypointPlan()
    assert WPP.needs_waypoint()
    t_rot = Rotation(np.quaternion(1), origin=NominalOriginStandard)
    t_pnt = Translation(5 + np.random.rand(3), origin=NominalOriginStandard)
    target_point = Transform(t_rot, t_pnt)
    target_speed = 10
    wp = planning.Waypoint(target_point, target_speed)
    distance = np.linalg.norm(t_pnt.vector)
    WPP.push(distance, wp)

    assert len(WPP) == 1
    assert not WPP.needs_waypoint()
    wp_1 = WPP.top()[1]
    wp_2 = WPP.pop()[1]
    assert wp == wp_1
    assert wp == wp_2