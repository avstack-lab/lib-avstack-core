import numpy as np
from numba import jit
from numba.types import float64, int64


def q_mult_vec(q, v):
    """
    v' = v + 2 * r x (s * v + r x v) / m
    where x represents the cross product, s and r are the scalar and vector parts
    of the quaternion, respectively, and m is the sum of the squares of the
    components of the quaternion.
    """
    try:
        q = q.q  # if input is a Rotation
    except AttributeError:
        pass
    s = q.w
    r = q.vec
    m = q.w**2 + q.x**2 + q.y**2 + q.z**2
    try:
        v2x = _q_mult_vec(s, r, m, v.x)
        return v.factory()(v2x, v.reference)
    except AttributeError:
        return _q_mult_vec(s, r, m, v)


@jit(nopython=True, fastmath=False)
def _q_mult_vec(s, r, m, v):
    return v + 2 * np.cross(r, (s * v + np.cross(r, v))) / m


@jit(float64[:](float64[:], int64, float64[:]), nopython=True)
def fastround(arr, ndec, out):
    for i in range(len(arr)):
        if np.abs(arr[i]) < 1e-12:
            arr[i] = 0.0
    return np.round_(arr, ndec, out)
