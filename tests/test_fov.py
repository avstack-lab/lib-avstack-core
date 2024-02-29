import math

import numpy as np

from avstack.geometry.fov import Circle, Vesica, Wedge


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


# def test_get_disjoint_sets_two_circles_with_overlap():
#     c1 = Circle(radius=1, center=np.array([1, 1]))
#     c2 = Circle(radius=1, center=np.array([1, 1.5]))
#     fovs = {1: c1, 2: c2}
#     disjoint_sets = get_disjoint_fov_subsets(fovs)
#     assert len(disjoint_sets) == 3
