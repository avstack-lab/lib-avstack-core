# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-19
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-23
# @Description:
"""

"""

from copy import copy, deepcopy

import numpy as np
import quaternion

from avstack import transformations as tforms
from avstack.geometry import (
    CameraCoordinates,
    NominalOriginCamera,
    NominalOriginStandard,
    Origin,
    Rotation,
    StandardCoordinates,
    Transform,
    Translation,
    get_transform_from_line,
)


R_stan_to_cam = StandardCoordinates.get_conversion_matrix(CameraCoordinates)
q_stan_to_cam = quaternion.from_rotation_matrix(R_stan_to_cam)
R_cam_to_stan = R_stan_to_cam.T
q_cam_to_stan = q_stan_to_cam.conjugate()

NominalOriginStandard = Origin(np.zeros((3,)), np.quaternion(1))
NominalOriginCamera = Origin(np.zeros((3,)), q_stan_to_cam)

NominalRotation = Rotation(np.quaternion(1), origin=NominalOriginStandard)
NominalTranslation = Translation([0, 0, 0], origin=NominalOriginStandard)
NominalTransform = Transform(NominalRotation, NominalTranslation)


# ===========================================================
# ORIGIN
# ===========================================================


def test_origin():
    origin = Origin(np.array([1, 2, 3]), np.quaternion(1))
    assert np.all(
        origin.matrix
        == np.array([[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 1, -3], [0, 0, 0, 1]])
    )


def test_origin_change_origin_rotation():
    r2 = [1, -1.1, -1]
    R2 = tforms.transform_orientation(r2, "euler", "DCM")
    rot_2 = Rotation(R2, origin=NominalOriginCamera)
    rot_2.change_origin(NominalOriginStandard)
    R2_stan = R2 @ R_stan_to_cam
    assert np.allclose(R2_stan, rot_2.R)


def test_origin_manual_change_coords():
    # the explicit, manual way
    Rs = tforms.transform_orientation([1, 0.2, -0.5], "euler", "dcm")
    vs_1 = np.array([3, -2, 4])
    vs_2 = Rs @ vs_1
    Rs_to_c = StandardCoordinates.get_conversion_matrix(CameraCoordinates)
    vs_1_in_c = Rs_to_c @ vs_1
    vs_2_in_c = Rs_to_c @ vs_2
    Rc = Rs_to_c @ Rs @ Rs_to_c.T
    vc_2 = Rc @ vs_1_in_c
    assert np.allclose(vc_2, vs_2_in_c)


def test_origin_rotate():
    x_s = np.array([1, 2, 3])
    origin = Origin(x_s, np.quaternion(1))
    origin_new = origin.rotate(q_stan_to_cam)
    origin_new_2 = origin.rotate(R_stan_to_cam)
    assert origin_new.allclose(origin_new_2)
    origin_orig = origin_new.rotate(q_stan_to_cam.conjugate())
    assert origin_orig.allclose(origin)


def test_origin_translate():
    x_s = np.array([1, 2, 3])
    x_d = np.array([1, 2, 3])
    origin = Origin(x_s, np.quaternion(1))
    origin_new = origin.translate(x_d)
    origin_orig = origin_new.translate(-x_d)
    assert origin_orig == origin


def test_invert_origin():
    vector = np.random.rand(3)  # in the "origin" frame
    x = np.array([1, 2, 3])
    q = tforms.transform_orientation([1, 0.2, -0.5], "euler", "quat")
    origin = Origin(x, q)
    Tr = Translation(vector, origin)

    vector_global_man = quaternion.rotate_vectors(q.conjugate(), vector) + x
    Tr.change_origin(NominalOriginStandard)
    assert np.allclose(Tr.vector, vector_global_man)
    vector_global = origin.inv() @ vector
    assert np.allclose(vector_global, vector_global_man)


def test_identity_inverse():
    assert NominalOriginStandard == NominalOriginStandard.inv()


def test_change_origin_points():
    x1 = np.array([1, 1, 1])
    q1 = tforms.transform_orientation([1, 0.2, -0.5], "euler", "quat")
    O1 = Origin(x1, q1)
    x2 = np.array([-1, 0, -1])
    q2 = q_stan_to_cam
    O2 = Origin(x2, q2)
    pts = np.random.rand(100, 3)
    pts_new = O1.change_points_origin(pts, O2)
    pts_old = O2.change_points_origin(pts_new, O1)
    assert np.allclose(pts, pts_old)


# ===========================================================
# TRANSLATION
# ===========================================================


def test_change_point_origin():
    v1 = np.array([10, 0, 0])
    O1 = NominalOriginStandard
    Tr1 = Translation(v1, origin=O1)
    x2 = np.array([-1, 0, 0])
    q2 = q_stan_to_cam
    O2 = Origin(x2, q2)
    Tr1.change_origin(O2)
    assert np.allclose(Tr1.vector, np.array([0, 0, 11]))
    vector_manual = O2 @ (O1.inv() @ v1)
    assert np.allclose(Tr1.vector, vector_manual)


def test_change_point_origin_2():
    v1 = np.array([10, 3, 2])
    x1 = np.array([1, 1, 1])
    q1 = tforms.transform_orientation([1, 0.2, -0.5], "euler", "quat")
    O1 = Origin(x1, q1)
    Tr1 = Translation(v1, origin=O1)
    x2 = np.array([-1, 0, -1])
    q2 = q_stan_to_cam
    O2 = Origin(x2, q2)
    Tr1.change_origin(O2)
    vector_manual = O2 @ (O1.inv() @ v1)
    assert np.allclose(Tr1.vector, vector_manual)


def test_translation():
    v1 = np.array([10, -2.2, 1.3])
    v2 = np.array([11, 22.1, 10])
    trans_1 = Translation(v1)
    trans_2 = Translation(v2[0], v2[1], v2[2])
    assert np.all((trans_1 - trans_2).vector == (v1 - v2))
    assert trans_1.distance(trans_2) == np.linalg.norm(v1 - v2)


def test_translation_nonsame():
    v1 = np.array([10, -2.2, 1.3])  # defined in standard coords
    v2 = np.array([11, 22.1, 10])  # defined in camera coords
    trans_1 = Translation(v1, origin=NominalOriginStandard)
    trans_2 = Translation(v2, origin=NominalOriginCamera)
    trans_sub = trans_1 - trans_2
    assert trans_sub.origin == NominalOriginStandard
    assert np.allclose(trans_sub.vector, v1 - R_cam_to_stan @ v2)
    assert np.isclose(
        trans_1.distance(trans_2), np.linalg.norm(v1 - R_cam_to_stan @ v2)
    )


def test_translation_change_origin():
    # Define origins
    # -- 1
    O1 = NominalOriginStandard
    v1 = np.array([10, -2.2, 1.3])
    trans_1 = Translation(v1, origin=O1)
    # -- 2
    q2 = tforms.transform_orientation([0.1, 0.2, -0.1], "euler", "quat")
    x2 = np.array([1, 2, 3])
    O2 = Origin(x2, q2)

    # Change origin and test
    trans_1.change_origin(O2)
    assert np.allclose(
        trans_1.vector, quaternion.rotate_vectors(q2, v1 - x2), atol=1e-4
    )
    assert np.allclose(trans_1.vector, O2.R @ (v1 - x2))


def test_translation_change_origin_nonsame():
    # Define origins
    O1 = NominalOriginStandard
    v1 = np.array([10, -2.2, 1.3])
    trans_1 = Translation(v1, origin=O1)
    q2 = q_stan_to_cam * tforms.transform_orientation([0.1, 0.2, -0.1], "euler", "quat")
    x2 = np.array([1, 2, 3])
    O2 = Origin(x2, q2)
    # Change origin and test
    trans_1.change_origin(O2)
    assert np.allclose(
        trans_1.vector, quaternion.rotate_vectors(q2, v1 - x2), atol=1e-4
    )


# ===========================================================
# ROTATION
# ===========================================================


def test_rotation():
    q1 = tforms.transform_orientation([0, 1.1, -2], "euler", "quat")
    q2 = tforms.transform_orientation([1, -1.1, -1], "euler", "quat")
    rot_1 = Rotation(q1, origin=NominalOriginStandard)
    rot_2 = Rotation(q2, origin=NominalOriginStandard)
    assert np.allclose((rot_1 @ rot_2).q, q1 * q2)


def test_rotation_nonsame():
    # import ipdb; ipdb.set_trace()
    r2 = [1, -1.1, -1]
    q1 = tforms.transform_orientation([0, 1.1, -2], "euler", "quat")
    q2 = tforms.transform_orientation(r2, "euler", "quat")
    rot_1 = Rotation(q1, origin=NominalOriginStandard)
    rot_2 = Rotation(q2, origin=NominalOriginCamera)
    assert quaternion.allclose(
        (rot_1 @ rot_2).q, q1 * q2 * q_stan_to_cam, rtol=1e-5, atol=1e-8
    )


def test_rotation_change_origin_1():
    eulers = np.array([0, 1.1, -2])
    q1 = tforms.transform_orientation(eulers, "euler", "quat")
    rot_1 = Rotation(q1, origin=NominalOriginStandard)
    O_new = Origin(np.array([1, 2, 3]), np.quaternion(1))
    rot_1.change_origin(O_new)
    assert quaternion.allclose(rot_1.q, q1, rtol=1e-5, atol=1e-8)
    d_eulers = np.array([0, 0, 0.4])  # this only works because yaw is applied first
    q_new = tforms.transform_orientation(d_eulers, "euler", "quat")
    O_new = Origin(np.zeros((3,)), q_new)
    rot_1.change_origin(O_new)
    eulers_1 = tforms.transform_orientation(rot_1.R, "DCM", "euler")
    assert np.allclose(eulers_1, eulers - d_eulers, rtol=1e-5, atol=1e-5)


def test_rotation_change_origin_nonsame():
    eulers = np.array([0, 1.1, -2])
    q1 = tforms.transform_orientation(eulers, "euler", "quat")
    rot_1 = Rotation(q1, origin=NominalOriginStandard)
    O_new = Origin(np.array([1, 2, 3]), q_stan_to_cam)
    rot_1.change_origin(O_new)
    assert quaternion.allclose(rot_1.q, q1 * q_cam_to_stan, rtol=1e-5, atol=1e-8)
    d_eulers = np.array([0, 0, 0.4])  # this only works because yaw is applied first
    q_new = tforms.transform_orientation(d_eulers, "euler", "quat")
    O_new = Origin(np.zeros((3,)), q_new)
    rot_1.change_origin(O_new)
    eulers_1 = tforms.transform_orientation(rot_1.q, "quat", "euler")
    assert np.allclose(eulers_1, eulers - d_eulers, rtol=1e-5, atol=1e-5)


def test_rotation_vectors_simple():
    O1 = NominalOriginCamera
    q1 = np.quaternion(1)
    rot_1 = Rotation(q1, origin=O1)
    assert np.allclose(rot_1.forward_vector, R_stan_to_cam @ np.array([1, 0, 0]))
    assert np.allclose(rot_1.left_vector, np.array([-1, 0, 0]))
    assert np.allclose(rot_1.up_vector, np.array([0, -1, 0]))


def test_rotation_vectors_complex_2():
    O1 = NominalOriginCamera
    q1 = tforms.transform_orientation([0, 0, 0.5], "euler", "quat")
    rot_1 = Rotation(q1, origin=O1)
    R = quaternion.as_rotation_matrix(q1)
    assert np.allclose(rot_1.forward_vector, R_stan_to_cam @ np.array([1, 0, 0]))
    assert not np.allclose(rot_1.left_vector, np.array([-1, 0, 0]))
    assert not np.allclose(rot_1.up_vector, np.array([0, -1, 0]))


# ===========================================================
# TRANSFORM
# ===========================================================


def test_transform_1():
    v1 = np.array([10, -2.2, 1.3])
    v2 = np.array([11, 22.1, 10])
    q1 = tforms.transform_orientation([0, 1.1, -2], "euler", "quat")
    q2 = tforms.transform_orientation([1, -1.1, -1], "euler", "quat")
    transf_1 = Transform(
        Rotation(q1, origin=NominalOriginStandard),
        Translation(v1, origin=NominalOriginStandard),
    )
    transf_2 = Transform(
        Rotation(q2, origin=NominalOriginStandard),
        Translation(v2, origin=NominalOriginStandard),
    )

    # apply transforms
    v = np.random.rand(3)
    v_t = transf_1 @ v
    assert np.allclose(v_t.vector, quaternion.rotate_vectors(q1, (v - v1)))


def test_transform_change_origin():
    v1 = np.array([10, -2.2, 1.3])
    q1 = tforms.transform_orientation([0, 1.1, -2], "euler", "quat")
    T_1 = Transform(
        Rotation(q1, origin=NominalOriginStandard),
        Translation(v1, origin=NominalOriginStandard),
    )
    T_1.change_origin(NominalOriginCamera)
    T_1.change_origin(NominalOriginStandard)
    assert np.allclose(T_1.rotation.q, q1)
    assert np.allclose(T_1.translation.vector, v1)


def test_transform_invert_1():
    v1 = np.array([10, -2.2, 1.3])
    q1 = tforms.transform_orientation([0, 1.1, -2], "euler", "quat")
    T_1 = Transform(
        Rotation(q1, origin=NominalOriginStandard),
        Translation(v1, origin=NominalOriginStandard),
    )
    T_null = T_1.T @ T_1
    assert np.allclose(T_null.rotation.R, np.eye(3))
    assert np.allclose(T_null.translation.vector, np.zeros((3,)))
    assert T_null.origin == NominalOriginStandard


def test_transform_invert_mixed():
    v1 = np.array([10, -2.2, 1.3])
    q1 = tforms.transform_orientation([0, 1.1, -2], "euler", "quat")
    T_1 = Transform(
        Rotation(q1, origin=NominalOriginStandard),
        Translation(v1, origin=NominalOriginStandard),
    )
    T_1b = deepcopy(T_1)
    T_1b.change_origin(NominalOriginCamera)
    T_null = T_1.T @ T_1b
    assert np.allclose(T_null.rotation.R, np.eye(3))
    assert np.allclose(T_null.translation.vector, np.zeros((3,)))
    assert T_null.origin == NominalOriginStandard


def test_transform_compose_associative():
    v1 = np.array([10, -2.2, 1.3])
    v2 = np.array([11, 22.1, 10])
    q1 = tforms.transform_orientation([0, 1.1, -2], "euler", "quat")
    q2 = tforms.transform_orientation([1, -1.1, -1], "euler", "quat")
    T_1 = Transform(
        Rotation(q1, origin=NominalOriginStandard),
        Translation(v1, origin=NominalOriginStandard),
    )
    T_2 = Transform(
        Rotation(q2, origin=NominalOriginStandard),
        Translation(v2, origin=NominalOriginStandard),
    )

    # Perform multiplications -- should check out associatively
    T_res = T_1 @ T_1 @ T_2
    T_res_2 = T_1 @ (T_1 @ T_2)
    T_res_man = T_1.matrix @ T_1.matrix @ T_2.matrix

    # check all values are the same after operations
    assert np.allclose(
        T_1.rotation.R, tforms.transform_orientation([0, 1.1, -2], "euler", "DCM")
    )
    assert np.allclose(T_1.translation.vector, np.array([10, -2.2, 1.3]))
    assert np.allclose(
        T_2.rotation.R, tforms.transform_orientation([1, -1.1, -1], "euler", "DCM")
    )
    assert np.allclose(T_2.translation.vector, np.array([11, 22.1, 10]))

    # check associativity test
    assert np.allclose(T_res.matrix, T_res_2.matrix)
    assert np.allclose(T_res.matrix, T_res_man)


def test_transform_invert_composed():
    v1 = np.array([10, -2.2, 1.3])
    v2 = np.array([11, 22.1, 10])
    q1 = tforms.transform_orientation([0, 1.1, -2], "euler", "quat")
    q2 = tforms.transform_orientation([1, -1.1, -1], "euler", "quat")
    T_1 = Transform(
        Rotation(q1, origin=NominalOriginStandard),
        Translation(v1, origin=NominalOriginStandard),
    )
    T_2 = Transform(
        Rotation(q2, origin=NominalOriginStandard),
        Translation(v2, origin=NominalOriginStandard),
    )
    T_null = T_1.T @ T_2.T @ T_2 @ T_1
    T_nullb = (T_2 @ T_1).T @ (T_2 @ T_1)
    T_nullc = (T_2 @ T_1).T @ T_2 @ T_1
    assert np.allclose(T_nullb.matrix, T_nullc.matrix, rtol=1e-5, atol=1e-5)
    assert np.allclose(T_null.matrix, T_nullb.matrix, rtol=1e-5, atol=1e-5)
    assert np.allclose(T_null.rotation.R, np.eye(3), rtol=1e-5, atol=1e-5)
    assert np.allclose(T_null.translation.vector, np.zeros((3,)), rtol=1e-5, atol=1e-5)
    assert T_null.origin == NominalOriginStandard


def get_T_for_compose_mixed():
    v1 = np.array([10, -2.2, 1.3])
    v2 = np.array([11, 22.1, 10])
    q1 = tforms.transform_orientation([0, 1.1, -2], "euler", "quat")  # standard
    q2 = tforms.transform_orientation([1, -1.1, -1], "euler", "quat")  # camera
    T_1 = Transform(
        Rotation(q1, origin=NominalOriginStandard),
        Translation(v1, origin=NominalOriginStandard),
    )
    T_2 = Transform(
        Rotation(q2, origin=NominalOriginCamera),
        Translation(v2, origin=NominalOriginCamera),
    )
    return T_1, T_2


def test_transform_invert_composed_mixed():
    T_1, T_2 = get_T_for_compose_mixed()
    T_null_part_1 = T_1.T @ T_2.T
    T_null_part_2 = (T_1.T @ T_2.T).T
    assert T_null_part_1.origin == NominalOriginStandard
    assert T_null_part_2.origin == NominalOriginStandard
    T_null_parts = T_null_part_1 @ T_null_part_2
    assert np.allclose(T_null_parts.rotation.R, np.eye(3))

    T_1, T_2 = get_T_for_compose_mixed()
    T_null = T_1.T @ T_2.T @ T_2 @ T_1  # why isn't this one working????
    T_1, T_2 = get_T_for_compose_mixed()
    T_nullb = (T_2 @ T_1).T @ (T_2 @ T_1)
    T_1, T_2 = get_T_for_compose_mixed()
    T_nullc = (T_2 @ T_1).T @ T_2 @ T_1

    # Check coordinates
    assert T_null.origin == NominalOriginStandard
    assert T_nullb.origin == NominalOriginCamera
    assert T_nullc.origin == NominalOriginCamera

    # Check tests
    assert np.allclose(T_nullb.matrix, T_nullc.matrix, rtol=1e-5, atol=1e-5)
    assert np.allclose(T_nullb.rotation.R, np.eye(3), rtol=1e-5, atol=1e-5)
    assert np.allclose(T_nullb.translation.vector, np.zeros((3,)), rtol=1e-5, atol=1e-5)


def test_transform_from_string():
    v1 = np.array([10, -2.2, 1.3])
    q1 = tforms.transform_orientation([0, 1.1, -2], "euler", "quat")
    T_1 = Transform(
        Rotation(q1, origin=NominalOriginStandard),
        Translation(v1, origin=NominalOriginStandard),
    )
    T_str = T_1.format_as_string()
    T_1_from_str = get_transform_from_line(T_str)
    assert T_1_from_str == T_1
