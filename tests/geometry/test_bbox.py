import json
import sys
from copy import deepcopy

import numpy as np

import avstack.geometry.bbox as bbox
from avstack import calibration, exceptions
from avstack.geometry import (
    Attitude,
    GlobalOrigin3D,
    Position,
    ReferenceFrame,
    q_stan_to_cam,
)
from avstack.geometry import transformations as tforms


sys.path.append("tests/")
from utilities import get_test_sensor_data


(
    obj,
    box_calib,
    lidar_calib,
    pc,
    camera_calib,
    img,
    radar_calib,
    rad,
    box_2d,
    box_3d,
) = get_test_sensor_data()


def get_box():
    x = np.array([3.4, 5.4, -10.2])
    q = tforms.transform_orientation([0, 0, 1.2], "euler", "quat")
    pos = Position(x, GlobalOrigin3D)
    rot = Attitude(q, GlobalOrigin3D)
    return bbox.Box3D(pos, rot, [3, 4, 5])


def test_box_translate():
    box = get_box()
    L = Position(np.random.rand(3), GlobalOrigin3D)
    box2 = box.translate(L, inplace=False)
    assert not box2.position.allclose(box.position)
    assert box2.attitude.allclose(box.attitude)


def test_box_rotation():
    box = get_box()
    R = Attitude(q_stan_to_cam, GlobalOrigin3D)
    box2 = box.rotate(R, inplace=False)
    assert not box2.position.allclose(box.position)
    assert not box2.attitude.allclose(box.attitude)


def test_box_rotation_attitude():
    box = get_box()
    R = Attitude(q_stan_to_cam, GlobalOrigin3D)
    box2 = box.rotate_attitude(R, inplace=False)
    assert box2.position.allclose(box.position)
    assert not box2.attitude.allclose(box.attitude)


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
    box1 = get_box()
    box1a = box1.translate(Position(np.array([1, 1, 1]), GlobalOrigin3D), inplace=False)
    assert bbox.box_intersection(box1.corners, box1a.corners) > 0.0
    pos2 = Position(np.array([10, 5.4, -10.2]), GlobalOrigin3D)
    box2 = bbox.Box3D(pos2, box1.attitude, [3, 4, 5])
    assert bbox.box_intersection(box1.corners, box2.corners) == 0.0


def test_same_intersection_union():
    box1 = get_box()
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
    assert box.squeeze(im_h, im_w, inplace=False) == box
    assert box._x_valid(im_w) and box._y_valid(im_h) and box.check_valid(im_h, im_w)

    box = bbox.Box2D([-10, 20, 50, 60], cam_calib)
    assert box.squeeze(im_h, im_w, inplace=False) != box
    assert (
        not box._x_valid(im_w)
        and box._y_valid(im_h)
        and not box.check_valid(im_h, im_w)
    )
    assert box.squeeze(im_h, im_w, inplace=False).check_valid(im_h, im_w)

    box = bbox.Box2D([10, 20, 5000, 60], cam_calib)
    assert box.squeeze(im_h, im_w, inplace=False) != box
    assert (
        not box._x_valid(im_w)
        and box._y_valid(im_h)
        and not box.check_valid(im_h, im_w)
    )
    assert box.squeeze(im_h, im_w, inplace=False).check_valid(im_h, im_w)

    try:
        box = bbox.Box2D([10, 20, 50, 10], cam_calib)
    except exceptions.BoundingBoxError as e:
        pass


def test_box_IoU():
    x_1 = np.array([3.4, 5.4, -10.2])
    q_1 = tforms.transform_orientation([0, 0, 1.2], "euler", "quat")
    box_1 = bbox.Box3D(
        Position(x_1, GlobalOrigin3D), Attitude(q_1, GlobalOrigin3D), [3, 4, 5]
    )

    x_2 = np.array([5, 5, -10.4])
    q_2 = tforms.transform_orientation([0, 0, 1.4], "euler", "quat")
    box_2 = bbox.Box3D(
        Position(x_2, GlobalOrigin3D), Attitude(q_2, GlobalOrigin3D), [3, 4, 5]
    )
    assert box_1.IoU(box_2) > 0


def test_2d_3d_iou():
    box_1_3d = box_3d
    box_1_2d = box_3d.project_to_2d_bbox(camera_calib)
    assert np.isclose(box_1_2d.IoU(box_1_3d), 1.0)
    assert np.isclose(box_1_3d.IoU(box_1_2d), 1.0)


def test_project_to_2d_box():
    box_1_2d = box_3d.project_to_2d_bbox(camera_calib)
    assert box_1_2d.xmin < box_1_2d.xmax < camera_calib.img_shape[1]
    assert box_1_2d.ymin < box_1_2d.ymax < camera_calib.img_shape[0]


def test_segmask_from_2d_box():
    segmask = box_2d.as_seg_mask()
    assert len(segmask) > 0


def test_upscale_2d_to_3d_box():
    # reconstruct a 3d box
    box_3d_cam = box_3d.change_reference(camera_calib.reference, inplace=False)

    # fix attitude in the original frame of reference
    box_3d_cam.attitude.q = q_stan_to_cam.conjugate()

    # perform projection
    box_1_2d = box_3d.project_to_2d_bbox(camera_calib)
    box_1_3d = box_1_2d.upscale_to_3d(z_to_box=box_3d_cam.position[2])

    # check the consistency of the boxes
    assert np.linalg.norm(box_1_3d.position.x - box_3d_cam.position.x) < 3
    assert box_1_3d.IoU(box_3d_cam) > 0


# ===========================================================
# TRANSFORMS ON BOXES
# ===========================================================


def corners_close_perm_invar(c1, c2):
    """Sort matrices to be permutation-invariant"""
    c1_pm = sort_corners_invariant(np.round(c1.x, 3))
    c2_pm = sort_corners_invariant(np.round(c2.x, 3))
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
    O1 = ReferenceFrame(x1, q1, GlobalOrigin3D)
    x2 = np.array([-1, 0, -1])
    q2 = tforms.transform_orientation([-0.5, 0.7, -0.1], "euler", "quat")
    O2 = ReferenceFrame(x2, q2, GlobalOrigin3D)
    return O1, O2


def test_change_reference_simple_corners():
    # make box
    x_obj = np.array([1, 2, 3])
    q_obj = tforms.transform_orientation([0, 0, 0], "euler", "quat")
    box_1 = bbox.Box3D(
        Position(x_obj, GlobalOrigin3D), Attitude(q_obj, GlobalOrigin3D), [2, 8, 15]
    )

    # corners in global
    corners_pre_1 = box_1.corners
    corners_pre_2 = box_1.corners_global
    # corners in new reference
    x1 = np.array([1, -1, 1])
    q1 = tforms.transform_orientation([0, 0, np.pi - np.pi / 3], "euler", "quat")
    O1 = ReferenceFrame(x1, q1, GlobalOrigin3D)
    box_1.change_reference(O1, inplace=True)
    corners_new = box_1.corners
    # check corners
    corners_new_global = box_1.corners_global
    corners_new_man = corners_pre_1.change_reference(O1, inplace=False)
    corners_new_invert = corners_new_man.change_reference(
        corners_pre_1.reference, inplace=False
    )

    assert corners_close_perm_invar(corners_pre_1, corners_pre_2)
    assert corners_close_perm_invar(corners_new_man, corners_new)
    assert corners_close_perm_invar(corners_new_invert, corners_pre_1)
    assert corners_close_perm_invar(corners_pre_1, corners_new_global)


def test_change_reference_full_corners():

    box_1 = get_box()
    corners_pre_1 = box_1.corners
    corners_pre_2 = box_1.corners_global
    x1 = np.array([1, -1, 1])
    q1 = tforms.transform_orientation([0, 0, np.pi - np.pi / 3], "euler", "quat")
    O1 = ReferenceFrame(x1, q1, GlobalOrigin3D)
    box_1.change_reference(O1, inplace=True)
    corners_new = box_1.corners
    corners_new_global = box_1.corners_global
    corners_new_man = corners_pre_1.change_reference(O1, inplace=False)
    corners_new_invert = corners_new_man.change_reference(
        corners_pre_1.reference, inplace=False
    )

    assert corners_close_perm_invar(corners_pre_1, corners_pre_2)
    assert corners_close_perm_invar(corners_new_man, box_1.corners)
    assert corners_close_perm_invar(corners_new_invert, corners_pre_1)
    assert corners_close_perm_invar(corners_pre_1, corners_new_global)


def test_change_reference_iou():
    x1 = np.array([1, -1, 1])
    q1 = tforms.transform_orientation([1, 0.2, -0.5], "euler", "quat")
    O1 = ReferenceFrame(x1, q1, GlobalOrigin3D)
    x2 = np.array([-1, 0, -1])
    q2 = tforms.transform_orientation([-0.5, 0.7, -0.1], "euler", "quat")
    O2 = ReferenceFrame(x2, q2, GlobalOrigin3D)

    q_obj = tforms.transform_orientation([0, 0, -0.3], "euler", "quat")
    box_1 = get_box()
    box_2 = deepcopy(box_1)
    box_1.change_reference(O1, inplace=True)
    box_2.change_reference(O2, inplace=True)
    assert box_1.center_global.allclose(box_2.center_global)
    assert corners_close_perm_invar(box_1.corners_global, box_2.corners_global)
    assert np.allclose(box_1.IoU(box_2), 1.0)


# ===========================================================
# I/O
# ===========================================================


def test_io_2d_bbox():
    box_2d_2 = json.loads(box_2d.encode(), cls=bbox.BoxDecoder)
    assert box_2d_2.allclose(box_2d)


def test_io_3d_bbox():
    box_3d_2 = json.loads(box_3d.encode(), cls=bbox.BoxDecoder)
    assert box_3d_2.allclose(box_3d)


def test_io_2d_segmask():
    P = np.random.rand(3, 4)
    img_shape = (256, 1024)
    cam_calib = calibration.CameraCalibration(GlobalOrigin3D, P, img_shape=img_shape)
    segmask1 = np.random.rand(*img_shape)
    segmask1 = bbox.SegMask2D(segmask1 > 0.5, cam_calib)
    segmask2 = json.loads(segmask1.encode(), cls=bbox.BoxDecoder)
    assert segmask1.allclose(segmask2)
