# @Author: Spencer Hallyburton <spencer>
# @Date:   2021-02-24
# @Filename: test_bbox_util.py
# @Last modified by:   spencer
# @Last modified time: 2021-08-10

from copy import copy, deepcopy

import numpy as np
import quaternion

import avstack.geometry.bbox as bbox
from avstack import calibration, exceptions
from avstack.geometry import NominalOriginStandard, Origin, q_mult_vec
from avstack.geometry import transformations as tforms


def test_2d_intersection_union():
    # No intersection
    box1 = [100, 200, 200, 300]
    box2 = [201, 301, 300, 400]
    assert bbox.box_intersection(box1, box2) == 0.0

    # Partial intersection
    box1 = [1, 1, 10, 10]
    box2 = [1, 6, 10, 15]
    assert bbox.box_intersection(box1, box2) == bbox.box_union(box1, box2) / 3

    # Full intersection
    box1 = [100, 200, 200, 300]
    assert bbox.box_intersection(box1, box1) == bbox.box_union(box1, box1)


def test_3d_intersection_union():
    q = tforms.transform_orientation([0, 0, 1.2], "euler", "quat")
    box1 = bbox.Box3D([3, 4, 5, [3.4, 5.4, -10.2], q], origin=NominalOriginStandard)
    box1a = box1.translate(np.array([1, 1, 1]))
    assert bbox.box_intersection(box1.corners, box1a.corners) > 0.0
    box2 = bbox.Box3D([3, 4, 5, [10, 5.4, -10.2], q], origin=NominalOriginStandard)
    assert bbox.box_intersection(box1.corners, box2.corners) == 0.0


def test_same_intersection_union():
    q = tforms.transform_orientation([0, 0, 1.2], "euler", "quat")
    box1 = bbox.Box3D([3, 4, 5, [3.4, 5.4, -10.2], q], origin=NominalOriginStandard)
    assert abs(
        bbox.box_intersection(box1.corners, box1.corners)
        - bbox.box_union(box1.corners, box1.corners)
        < 1e-8
    )


def test_2d_box_valid():
    im_w = 100
    im_h = 200
    cam_calib = None

    box = bbox.Box2D([10, 20, 50, 60], cam_calib)
    assert box.squeeze(im_h, im_w) == box
    assert box._x_valid(im_w) and box._y_valid(im_h) and box.check_valid(im_h, im_w)

    box = bbox.Box2D([-10, 20, 50, 60], cam_calib)
    assert box.squeeze(im_h, im_w) != box
    assert (
        not box._x_valid(im_w)
        and box._y_valid(im_h)
        and not box.check_valid(im_h, im_w)
    )
    assert box.squeeze(im_h, im_w).check_valid(im_h, im_w)

    box = bbox.Box2D([10, 20, 5000, 60], cam_calib)
    assert box.squeeze(im_h, im_w) != box
    assert (
        not box._x_valid(im_w)
        and box._y_valid(im_h)
        and not box.check_valid(im_h, im_w)
    )
    assert box.squeeze(im_h, im_w).check_valid(im_h, im_w)

    try:
        box = bbox.Box2D([10, 20, 50, 10], cam_calib)
    except exceptions.BoundingBoxError as e:
        pass


def test_3d_box():
    q = tforms.transform_orientation([0, 0, 1.2], "euler", "quat")
    box = bbox.Box3D([3, 4, 5, [3.4, 5.4, -10.2], q], origin=NominalOriginStandard)


def test_bbox_from_string():
    q = tforms.transform_orientation([0, 0, 1.2], "euler", "quat")
    box = bbox.Box3D([3, 4, 5, [3.4, 5.4, -10.2], q], origin=NominalOriginStandard)
    box_str = box.format_as_string()
    box_from_str = bbox.get_box_from_line(box_str)
    assert box_from_str == box


def test_box_IoU():
    q_1 = tforms.transform_orientation([0, 0, 1.2], "euler", "quat")
    box_1 = bbox.Box3D([3, 4, 5, [3.4, 5.4, -10.2], q_1], origin=NominalOriginStandard)
    q_2 = tforms.transform_orientation([0, 0, 1.4], "euler", "quat")
    box_2 = bbox.Box3D([3, 4, 5, [5, 5, -10.4], q_2], origin=NominalOriginStandard)
    assert box_1.IoU(box_2) > 0


# ===========================================================
# TRANSFORMS ON BOXES
# ===========================================================


def corners_close_perm_invar(c1, c2):
    """Sort matrices to be permutation-invariant"""
    c1_pm = sort_corners_invariant(np.round(c1, 3))
    c2_pm = sort_corners_invariant(np.round(c2, 3))
    return np.allclose(c1_pm, c2_pm)


def sort_corners_invariant(mat):
    """Sort matrices to be permutation-invariant

    Sort according to the proper bounding box corner ordering
    Need to flip 2 <--> 3 and  6 <--> 7 to preserve order

    Negatives to allow sorting descending
    """
    # get ordering over the columns manually...
    mat *= -1
    mat = -np.sort(mat.view("f8,f8,f8"), order=["f2", "f1", "f0"], axis=0).view(
        np.float64
    )
    mat[[2, 3], :] = mat[[3, 2], :]
    mat[[6, 7], :] = mat[[6, 7], :]
    return mat


def get_origins():
    x1 = np.array([1, 1, 1])
    q1 = tforms.transform_orientation([1, 0.2, -0.5], "euler", "quat")
    O1 = Origin(x1, q1)
    x2 = np.array([-1, 0, -1])
    q2 = tforms.transform_orientation([-0.5, 0.7, -0.1], "euler", "quat")
    O2 = Origin(x2, q2)
    return O1, O2


def test_change_origin_simple_corners():
    x1 = np.array([1, -1, 1])
    q1 = tforms.transform_orientation([0, 0, np.pi - np.pi / 3], "euler", "quat")
    O1 = Origin(x1, q1)
    q_obj = tforms.transform_orientation([0, 0, 0], "euler", "quat")
    box_1 = bbox.Box3D([2, 8, 15, [1, 2, 3], q_obj], origin=NominalOriginStandard)
    corners_pre_1 = box_1.corners
    corners_pre_2 = box_1.corners_global
    box_1.change_origin(O1)
    # corners_new_full_man = q_mult_vec(q1 , corners_pre_1 - x1)
    corners_new = box_1.corners
    corners_new_global = box_1.corners_global
    corners_new_man = O1 @ corners_pre_1
    corners_new_invert = O1.inv() @ box_1.corners

    assert corners_close_perm_invar(corners_pre_1, corners_pre_2)
    assert corners_close_perm_invar(corners_new_man, corners_new)
    assert corners_close_perm_invar(corners_new_invert, corners_pre_1)
    assert corners_close_perm_invar(corners_pre_1, corners_new_global)


def test_change_origin_full_corners():
    x1 = np.array([1, -1, 1])
    q1 = tforms.transform_orientation([1, 0.2, -0.5], "euler", "quat")
    O1 = Origin(x1, q1)
    # q_obj = tforms.transform_orientation([1.0,0.7,-0.3], 'euler', 'quat')
    q_obj = tforms.transform_orientation([1.0, 0.7, -0.3], "euler", "quat")
    box_1 = bbox.Box3D([2, 8, 15, [1, 2, 3], q_obj], origin=NominalOriginStandard)
    corners_pre_1 = box_1.corners
    corners_pre_2 = box_1.corners_global
    box_1.change_origin(O1)
    corners_new = box_1.corners
    corners_new_global = box_1.corners_global
    corners_new_man = O1 @ corners_pre_1
    corners_new_invert = O1.inv() @ box_1.corners

    assert corners_close_perm_invar(corners_pre_1, corners_pre_2)
    assert corners_close_perm_invar(corners_new_man, box_1.corners)
    assert corners_close_perm_invar(corners_new_invert, corners_pre_1)
    assert corners_close_perm_invar(corners_pre_1, corners_new_global)


def test_change_origin_iou():
    x1 = np.array([1, -1, 1])
    q1 = tforms.transform_orientation([1, 0.2, -0.5], "euler", "quat")
    O1 = Origin(x1, q1)
    x2 = np.array([-1, 0, -1])
    q2 = tforms.transform_orientation([-0.5, 0.7, -0.1], "euler", "quat")
    O2 = Origin(x2, q2)

    q_obj = tforms.transform_orientation([0, 0, -0.3], "euler", "quat")
    box_1 = bbox.Box3D([2, 8, 16, [1, 2, 3], q_obj], origin=NominalOriginStandard)
    box_2 = deepcopy(box_1)
    box_1.change_origin(O1)
    box_2.change_origin(O2)
    assert np.allclose(box_1.t.vector_global, box_2.t.vector_global)
    assert corners_close_perm_invar(box_1.corners_global, box_2.corners_global)
    assert np.allclose(box_1.IoU(box_2), 1.0)


# ===========================================================
# I/O
# ===========================================================


def test_io_2d_bbox():
    P = np.random.rand(3, 4)
    img_shape = (256, 1024)
    cam_calib = calibration.CameraCalibration(
        NominalOriginStandard, P, img_shape=img_shape
    )
    box1 = bbox.Box2D([-10, 20, 50, 60], cam_calib)
    box2 = bbox.get_box_from_line(box1.format_as_string())
    assert box1 == box2


def test_io_3d_bbox():
    box1 = bbox.Box3D([1, 1, 1, [1, 2, 3], np.quaternion(1)], NominalOriginStandard)
    box2 = bbox.get_box_from_line(box1.format_as_string())
    assert box1 == box2


def test_io_2d_segmask():
    P = np.random.rand(3, 4)
    img_shape = (256, 1024)
    cam_calib = calibration.CameraCalibration(
        NominalOriginStandard, P, img_shape=img_shape
    )
    segmask1 = np.random.rand(*img_shape)
    segmask1 = bbox.SegMask2D(segmask1 > 0.5, cam_calib)
    segmask2 = bbox.get_segmask_from_line(segmask1.format_as_string())
    assert segmask1 == segmask2
