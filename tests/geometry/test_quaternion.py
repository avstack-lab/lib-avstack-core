# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-11
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""
My custom tests for the quaternion package
"""

import numpy as np
import quaternion
from avstack.geometry import Rotation, NominalOriginStandard


def test_compose_rotations():
    np.random.seed(1)
    v = np.random.randn(3)
    q_a_to_b = np.quaternion(*np.random.rand(3))
    q_a_to_c = np.quaternion(*np.random.rand(3))
    R_a_to_b = quaternion.as_rotation_matrix(q_a_to_b)
    R_a_to_c = quaternion.as_rotation_matrix(q_a_to_c)

    # test compose rotations
    q_b_to_c = q_a_to_c * q_a_to_b.conjugate()
    R_b_to_c = R_a_to_c @ R_a_to_b.T
    assert np.allclose(quaternion.as_rotation_matrix(q_b_to_c), R_b_to_c)

    # test apply composed rotations
    v1 = quaternion.rotate_vectors(q_b_to_c, v)
    v2 = R_b_to_c @ v
    assert np.allclose(v1, v2)


def test_quat_to_transform():
    q_a_to_b = np.quaternion(*np.random.rand(3))
    R = Rotation(q_a_to_b, NominalOriginStandard)
    assert np.allclose(quaternion.as_rotation_matrix(q_a_to_b), R.R)


def test_rotate_vector():
    v = np.random.randn(3)
    q_a_to_b = np.quaternion(*np.random.rand(3))
    x = np.random.rand(3)

    # import timeit

    # THIS IS WAY SLOW!!!
    x1 = quaternion.rotate_vectors(q_a_to_b, x)

    # THIS IS WAY FASTER!!!!
    x2 = (q_a_to_b * quaternion.from_vector_part(x) * q_a_to_b.inverse()).vec
    assert np.allclose(x1, x2)
