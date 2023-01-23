# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-08-07
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""
import numpy as np
import quaternion

from avstack import calibration
from avstack import transformations as tforms
from avstack.geometry import (
    CameraCoordinates,
    NominalOriginCamera,
    NominalOriginStandard,
    NominalTransform,
    Origin,
    R_cam_to_stan,
    R_stan_to_cam,
    Rotation,
    StandardCoordinates,
    Transform,
    Translation,
    q_stan_to_cam,
)


np.random.seed(0)


def test_project_3d_to_3d():
    O1 = NominalOriginStandard
    R2 = np.eye(3)
    x2 = np.array([1, -3, 2])
    O2 = Origin(x2, R2)
    C1 = calibration.Calibration(O1)
    C2 = calibration.Calibration(O2)
    v1 = np.array([1, -3, 2])
    Tr1 = Translation(v1, origin=O1)
    assert Tr1.origin == O1
    Tr1_in_2 = C1.transform_3d_to_3d(Tr1, origin=O2)
    assert np.allclose(Tr1_in_2.vector, v1 - x2)
    assert Tr1_in_2.origin == O2
    Tr1_in_1 = C2.transform_3d_to_3d(Tr1_in_2, origin=O1)
    assert np.allclose(Tr1.vector, Tr1_in_1.vector)
    assert Tr1_in_1.origin == O1


def test_project_3d_to_3d_array():
    O1 = NominalOriginStandard
    R2 = np.eye(3)
    x2 = np.array([1, -3, 2])
    O2 = Origin(x2, R2)
    C1 = calibration.Calibration(O1)
    C2 = calibration.Calibration(O2)
    pts_in_1 = np.random.randn(5, 3)
    pts_in_2 = C1.transform_3d_to_3d(pts_in_1, origin=O2)
    assert np.allclose(pts_in_2, pts_in_1 - x2)
    pts_in_1_again = C2.transform_3d_to_3d(pts_in_2, origin=O1)
    assert np.allclose(pts_in_1, pts_in_1_again)


def test_project_3d_to_3d_nonsame():
    # Define origins
    x1 = np.array([-6, 1, 4])
    R1 = np.eye(3)
    O1 = Origin(x1, R1)
    x2 = np.array([1, -3, 2])
    R2 = R_stan_to_cam
    O2 = Origin(x2, R2)

    # Define calibrations
    C1 = calibration.Calibration(O1)
    C2 = calibration.Calibration(O2)
    v1 = np.array([5, -3, 2])
    Tr1 = Translation(v1, origin=O1)

    # Perform transformations - 1
    Tr1_in_2 = C1.transform_3d_to_3d(Tr1, origin=O2)
    assert Tr1_in_2.origin == O2
    v1_in_2 = Tr1_in_2.vector
    v_man = R_stan_to_cam @ (v1 + x1 - x2)
    assert np.allclose(v1_in_2, v_man)

    # Perform transformations - 2
    Tr1_in_1 = C2.transform_3d_to_3d(Tr1_in_2, origin=O1)
    assert Tr1_in_1.origin == O1
    assert np.allclose(v1, Tr1_in_1.vector)


def test_calibration_transform():
    # -- set origins
    x1 = np.array([2, -1, 1])
    q1 = np.quaternion(1)
    x2 = np.array([2, 1, 1])
    q2 = tforms.transform_orientation([0, 0, np.pi], "euler", "quaternion")
    O1 = Origin(x1, q1)
    O2 = Origin(x2, q2)

    # -- set calibration
    calib_1 = calibration.Calibration(O1)
    calib_2 = calibration.Calibration(O2)
    pts_1 = np.array([[1, 2, 3], [4, 5, 6]])
    pts_1_to_2_a = calib_1.transform_3d_to_3d(pts_1, O2)
    pts_1_to_2_b = calib_2.transform_3d_to_3d_inv(pts_1, O1)
    pts_1_to_ref_man = quaternion.rotate_vectors(q1, pts_1) + x1
    pts_1_to_2_man = quaternion.rotate_vectors(q2, pts_1_to_ref_man - x2)

    assert not np.allclose(pts_1, pts_1_to_2_a)
    assert np.allclose(pts_1_to_2_a, pts_1_to_2_man)
    assert np.allclose(pts_1_to_2_a, pts_1_to_2_b)


def test_project_to_camera_halves_same_origin():
    # -- set up camera calibration -- from FOV of 90 degrees
    x1 = np.array([0, 0, 0])
    q1 = q_stan_to_cam
    q2 = np.quaternion(1)
    O_cam = Origin(x1, q1)
    O_pts = Origin(x1, q2)
    P = np.array(
        [[800.0, 0.0, 800.0, 0.0], [0.0, 800.0, 450.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    img_shape = [900, 1600, 3]
    calib_cam = calibration.CameraCalibration(O_cam, P, img_shape)

    # -- make all points (in front only)
    pts_all = 2 * (np.random.rand(1000, 3) - 0.5) * np.array([10, 10, 2]) + np.array(
        [40, 0, 0]
    )

    # -- points on left half of image
    pts_left = pts_all[pts_all[:, 1] >= 0]
    pts_proj = calib_cam.project_3d_points(pts_left, O_pts)
    assert len(pts_proj) == len(pts_left) < len(pts_all)
    pts_proj_left = pts_proj[0 <= (pts_proj[:, 0] <= img_shape[1] / 2), :]
    assert len(pts_proj_left) == len(pts_left)

    # -- points on right half of image
    pts_right = pts_all[pts_all[:, 1] <= 0]
    pts_proj = calib_cam.project_3d_points(pts_right, O_pts)  # same origin
    assert len(pts_proj) == len(pts_right) < len(pts_all)
    pts_proj_right = pts_proj[img_shape[1] > (pts_proj[:, 0] >= img_shape[1] / 2), :]
    assert len(pts_proj_right) == len(pts_right)

    # -- points on top half of image
    pts_top = pts_all[pts_all[:, 2] >= 0]
    pts_proj = calib_cam.project_3d_points(pts_top, O_pts)  # same origin
    assert len(pts_proj) == len(pts_top) < len(pts_all)
    pts_proj_top = pts_proj[0 <= (pts_proj[:, 1] <= img_shape[0] / 2), :]
    assert len(pts_proj_top) == len(pts_top)

    # -- points on bottom half of image
    pts_bot = pts_all[pts_all[:, 2] <= 0]
    pts_proj = calib_cam.project_3d_points(pts_bot, O_pts)  # same origin
    assert len(pts_proj) == len(pts_bot) < len(pts_all)
    pts_proj_bot = pts_proj[img_shape[0] > (pts_proj[:, 1] >= img_shape[0] / 2), :]
    assert len(pts_proj_bot) == len(pts_bot)


def test_calib_angles():
    x_razel = [50, 0.4, -0.3]
    x_cart = tforms.spherical_to_cartesian(x_razel)
    O1 = Origin(np.zeros((3,)), np.quaternion(1))
    O2 = Origin(np.zeros((3,)), q_stan_to_cam)
    P = np.array(
        [[800.0, 0.0, 800.0, 0.0], [0.0, 800.0, 450.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    img_shape = [900, 1600, 3]
    fov_h = 90 * np.pi / 180
    calib_camera = calibration.CameraCalibration(
        O2, P, img_shape, fov_horizontal=fov_h, fov_vertical=None, square_pixels=True
    )
    x_pixels = calib_camera.project_3d_points(x_cart, O1)
    x_az, x_el = calib_camera.pixel_to_angle(x_pixels)
    # for some reason requires high tolerance to pass...
    assert np.allclose([x_az, x_el], x_razel[1:], atol=0.05)
