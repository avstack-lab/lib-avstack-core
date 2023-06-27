# @Author: Spencer Hallyburton <spencer>
# @Date:   2021-02-24
# @Filename: test_tforms.py
# @Last modified by:   spencer
# @Last modified time: 2021-02-24


import numpy as np

from avstack.geometry import transformations as tforms


def test_rotation_conversion():
    D = tforms.transform_orientation([0, 0, 0], "euler", "DCM")
    assert np.all(D == np.eye(3))


def test_xyzvel_to_razelrrt():
    xyzvel = np.array([1, 0, 0, -3, -4, 0])
    razelrrt_exp = np.array([1, 0, 0, -3])  # only in the range direction
    razelrrt_cal = tforms.xyzvel_to_razelrrt(xyzvel)
    assert np.allclose(razelrrt_cal, razelrrt_exp)


def test_razelrrt_to_xyzvel():
    xyzvel = np.random.randn(6)
    razelrrt = tforms.xyzvel_to_razelrrt(xyzvel)
    xyzvel_recon = tforms.razelrrt_to_xyzvel(razelrrt)
    assert np.allclose(xyzvel[:3], xyzvel_recon[:3])
    assert np.isclose(abs(razelrrt[3]), np.linalg.norm(xyzvel_recon[3:6]))
