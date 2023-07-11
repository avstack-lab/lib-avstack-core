# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-08-22
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""

import json
import sys
from copy import deepcopy

import numpy as np

from avstack.environment.objects import ObjectStateDecoder
from avstack.geometry import Attitude, GlobalOrigin3D, ReferenceFrame
from avstack.geometry import transformations as tforms


sys.path.append("tests/")
from utilities import get_ego


def test_encode_decode_object():
    obj_1 = get_ego(seed=1)
    obj_2 = json.loads(obj_1.encode(), cls=ObjectStateDecoder)
    assert obj_1.allclose(obj_2)


def test_change_reference():
    O_new = ReferenceFrame(
        np.array([1, 2, 3]),
        tforms.transform_orientation([1, -1, 0.1], "euler", "quat"),
        GlobalOrigin3D,
    )
    obj1 = get_ego(seed=1)
    O_orig = deepcopy(obj1.reference)
    obj1_copy = get_ego(seed=1)
    obj1.change_reference(O_new, inplace=True)
    obj1.change_reference(O_orig, inplace=True)
    assert obj1.box.allclose(obj1_copy.box)


def test_object_as_reference():
    obj1 = get_ego(seed=1)
    obj_ref = obj1.as_reference()
    assert np.allclose(obj_ref.x, obj1.position.x)
    assert np.allclose(obj_ref.v, obj1.velocity.x)


def test_object_transform_reference():
    obj1 = get_ego(seed=1)
    obj1.attitude = Attitude(np.quaternion(1), obj1.attitude.reference)
    obj2 = get_ego(seed=2)
    obj1.attitude = Attitude(np.quaternion(1), obj1.attitude.reference)
    obj2_in_1 = obj2.change_reference(obj1, inplace=False)
    assert np.allclose(obj2_in_1.position.x, obj2.position.x - obj1.position.x)
    assert np.allclose(obj2_in_1.velocity.x, obj2.velocity.x - obj1.velocity.x)
