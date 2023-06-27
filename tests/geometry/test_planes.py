# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-19
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""


import numpy as np

from avstack.geometry import GlobalOrigin3D, planes


# def test_convert_plane():
#     coeffs = [1,2,3,4]
#     P1 = planes.GroundPlane(coeffs, StandardCoordinates)
#     P2 = P1.convert(CameraCoordinates, dz=0)
#     assert np.all(StandardCoordinates.convert(P1.normal, CameraCoordinates) == P2.normal)


def test_plane_angle_self():
    coeffs = [1, 2, 3, 4]
    P1 = planes.GroundPlane(coeffs, GlobalOrigin3D)
    assert P1.angle_between(P1) == 0


def tests_plane_to_transform():
    coeffs = [1, 2, 3, 4]
    P_g = planes.GroundPlane(coeffs, GlobalOrigin3D)
    T_s2g = P_g.as_reference()
    assert np.all(T_s2g.x == np.array([0, 0, coeffs[3]]))
