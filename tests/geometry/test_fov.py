import math
import sys

import numpy as np

from avstack.calibration import LidarCalibration
from avstack.geometry import GlobalOrigin3D
from avstack.geometry.fov import (
    Circle,
    Polygon,
    Vesica,
    Wedge,
    box_in_fov,
    points_in_fov,
)


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


def test_lidar_concave_hull():
    hull = pc.concave_hull_bev()
    assert hull.check_point(np.array([0, 0], dtype=float))
    assert not hull.check_point(np.array([200, 0], dtype=float))


def test_make_fov_from_pc_hull():
    fov = pc.concave_hull_bev()
    assert isinstance(fov, Polygon)


def test_point_in_fov():
    fov = pc.concave_hull_bev()
    in_fov = points_in_fov(np.array([0, 0, 0], dtype=float), fov)
    assert in_fov


def test_box_in_polygon_fov():
    fov = pc.concave_hull_bev()
    in_fov = box_in_fov(box_3d, fov)
    assert in_fov


def test_polygon_change_reference():
    # add some translation to the sensor for testing...
    pc.calibration.reference.x += np.random.rand(3)
    calib_global = LidarCalibration(GlobalOrigin3D)
    pc_global = pc.project(calib_global)
    fov = pc.concave_hull_bev()
    fov2 = fov.change_reference(GlobalOrigin3D, inplace=False)
    fov3 = pc_global.concave_hull_bev()
    assert fov.reference != fov2.reference
    assert fov2.reference == fov3.reference
    assert not np.allclose(fov.boundary, fov2.boundary)
    assert np.allclose(fov2.boundary, fov3.boundary)


def test_wedge():
    fov = Wedge(radius=2, angle_start=-math.pi / 4, angle_stop=math.pi / 4)
    assert fov.check_point(np.array([2, 0]))
    assert fov.check_point(np.array([0.25, 0.2]))
    assert fov.check_point(np.array([0.25, -0.2]))
    assert not fov.check_point(np.array([3, 0]))
    assert not fov.check_point(np.array([0, -0.1]))


def test_circle_circle_overlap_as_circle():
    c1 = Circle(radius=10, center=np.array([1, 1]))
    c2 = Circle(radius=1, center=np.array([4, 5]))
    overlap = c1.intersection(c2)
    assert isinstance(overlap, Circle)
    assert overlap.radius == c2.radius


def test_circle_circle_overlap_as_vesica():
    c1 = Circle(radius=3, center=np.array([1, 1]))
    c2 = Circle(radius=1, center=np.array([4, 1]))
    overlap = c1.intersection(c2)
    assert isinstance(overlap, Vesica)
    assert len(overlap.circles) == 2


def test_circle_circle_no_overlap():
    c1 = Circle(radius=1, center=np.array([1, 1]))
    c2 = Circle(radius=1, center=np.array([4, 5]))
    assert c1.intersection(c2) is None


def test_vesica_circle_overlap():
    c1 = Circle(radius=1, center=np.array([1, 1]))
    c2 = Circle(radius=1, center=np.array([1, 1.5]))
    c3 = Circle(radius=1, center=np.array([1, 1.75]))
    ves = c1.intersection(c2)
    ves2 = ves.intersection(c3)
    assert isinstance(ves2, Vesica)
    assert len(ves2.circles) == 3


def test_vesica_circle_overlap():
    c1 = Circle(radius=1, center=np.array([1, 1]))
    c2 = Circle(radius=1, center=np.array([1, 1.5]))
    c3 = Circle(radius=1, center=np.array([1, 5]))
    ves = c1.intersection(c2)
    ves2 = ves.intersection(c3)
    assert ves2 is None
