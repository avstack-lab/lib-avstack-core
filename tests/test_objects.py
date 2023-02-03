# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-08-22
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""

import sys
from copy import deepcopy

import numpy as np

from avstack.geometry import Origin, Rotation, Transform, Translation
from avstack.geometry import transformations as tforms


sys.path.append("tests/")
from utilities import get_ego


def test_change_reference():
    O_new = Origin(
        np.array([1, 2, 3]), tforms.transform_orientation([1, -1, 0.1], "euler", "dcm")
    )
    obj1 = get_ego(seed=1)
    O_orig = deepcopy(obj1.origin)
    obj1_copy = get_ego(seed=1)
    obj1.change_origin(O_new)
    obj1.change_origin(O_orig)
    assert obj1.box.allclose(obj1_copy.box)
