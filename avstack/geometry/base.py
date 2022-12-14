import numpy as np
from numba import jit


def q_mult_vec(q, v):
    """
    v' = v + 2 * r x (s * v + r x v) / m
    where x represents the cross product, s and r are the scalar and vector parts
    of the quaternion, respectively, and m is the sum of the squares of the
    components of the quaternion.
    """
    s = q.w
    r = q.vec
    m = q.w**2 + q.x**2 + q.y**2 + q.z**2
    return _q_mult_vec(s, r, m, v)


@jit(nopython=True, fastmath=False)
def _q_mult_vec(s, r, m, v):
    return v + 2*np.cross(r, (s*v + np.cross(r,v)))/m
