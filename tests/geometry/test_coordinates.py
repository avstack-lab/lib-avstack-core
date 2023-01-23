# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-19
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-08-07
# @Description:
"""

"""


import numpy as np

from avstack.geometry import coordinates


R_lidar_2_carla = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
R_lidar_2_camera = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
R_camera_2_carla = R_lidar_2_carla @ R_lidar_2_camera.T


def test_coord_equality():
    assert coordinates.LidarCoordinates == coordinates.StandardCoordinates
    assert coordinates.LidarCoordinates != coordinates.CameraCoordinates


def test_self_transform():
    C_LID = coordinates.LidarCoordinates
    C_CAM = coordinates.CameraCoordinates
    C_CAR = coordinates.CarlaCoordinates
    pts = np.random.randn(20, 3)
    assert np.allclose(C_LID.convert(pts, C_LID), pts)
    assert np.allclose(C_CAM.convert(pts, C_CAM), pts)
    assert np.allclose(C_CAR.convert(pts, C_CAR), pts)


def test_lidar_to_camera():
    C_LID = coordinates.LidarCoordinates
    C_CAM = coordinates.CameraCoordinates
    pts = np.random.randn(20, 3)
    assert np.allclose(C_LID.convert(pts, C_CAM), pts @ R_lidar_2_camera.T)
    assert np.allclose(C_CAM.convert(pts, C_LID), pts @ R_lidar_2_camera)


def test_lidar_carla():
    C_CAR = coordinates.CarlaCoordinates
    C_LID = coordinates.LidarCoordinates
    pts = np.random.randn(20, 3)
    assert np.allclose(C_LID.convert(pts, C_CAR), pts @ R_lidar_2_carla.T)
    assert np.allclose(C_CAR.convert(pts, C_LID), pts @ R_lidar_2_carla)


def test_camera_carla():
    C_CAM = coordinates.CameraCoordinates
    C_CAR = coordinates.CarlaCoordinates
    pts = np.random.randn(20, 3)
    assert np.allclose(C_CAM.convert(pts, C_CAR), pts @ R_camera_2_carla.T)
    assert np.allclose(C_CAR.convert(pts, C_CAM), pts @ R_camera_2_carla)


def test_conversion_matrix():
    M_CAM = coordinates.CameraCoordinates.matrix
    assert np.allclose(M_CAM, R_lidar_2_camera)
    M_LID = coordinates.LidarCoordinates.matrix
    assert np.allclose(M_LID, np.eye(3))


def test_convert_coords():
    C_CAM = coordinates.CameraCoordinates
    C_LID = coordinates.LidarCoordinates
    R_c2l = C_CAM.get_conversion_matrix(C_LID)
    assert np.allclose(R_c2l, R_lidar_2_camera.T)
