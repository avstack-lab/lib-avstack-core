# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-11
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""

import sys

import numpy as np
import quaternion

from avstack.environment.objects import VehicleState
from avstack.geometry import (
    NominalOriginStandard,
    NominalTransform,
    Rotation,
    Translation,
    Vector,
    bbox,
)


sys.path.append("tests/")
from utilities import get_ego


def test_vehicle_state_init():
    VS = get_ego(1)
    assert isinstance(VS, VehicleState)


def test_vehicle_state_convert():
    VS_1 = get_ego(1)
    VS_2 = get_ego(2)
    VS_2_in_1 = VS_1.global_to_local(VS_2)
    assert np.allclose(
        VS_2_in_1.position.vector,
        (VS_1.attitude @ (VS_2.position - VS_1.position)).vector,
    )
