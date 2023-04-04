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


def test_razelrrt_to_xyzvel():
    raise NotImplementedError


def test_xyzvel_to_razelrrt():
    raise NotImplementedError