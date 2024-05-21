import numpy as np

from avstack.geometry.utils import in_convex_hull, in_polygon, parallel_in_polygon


def test_in_convex_hull():
    hull = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert in_convex_hull(np.array([0.5, 0.5]), hull)
    assert not in_convex_hull(np.array([2, 0]), hull)


def test_concave_hull_failure():
    hull = np.array([[0, 0], [0.5, 0.5], [1, 0], [1, 1], [0, 1]])
    assert in_convex_hull(np.array([0.1, 0.2]), hull)
    assert in_convex_hull(np.array([0.5, 0.25]), hull)  # concave hull doesn't work here
    assert not in_convex_hull(np.array([2, 0]), hull)


def test_in_polygon():
    hull = np.array([[0, 0], [0.5, 0.5], [1, 0], [1, 1], [0, 1]])
    assert in_polygon(np.array([0.1, 0.2]), hull)
    assert not in_polygon(np.array([0.5, 0.25]), hull)
    assert not in_polygon(np.array([2, 0]), hull)


def test_parallel_polygon():
    hull = np.array([[0, 0], [0.5, 0.5], [1, 0], [1, 1], [0, 1]])
    assert np.all(parallel_in_polygon(np.array([[0.1, 0.2], [0.5, 0.5]]), hull))
