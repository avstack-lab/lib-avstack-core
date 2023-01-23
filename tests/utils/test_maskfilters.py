# @Author: Spencer Hallyburton <spencer>
# @Date:   2021-02-24
# @Filename: test_maskfilters.py
# @Last modified by:   spencer
# @Last modified time: 2021-08-03


import os
import sys
from copy import copy, deepcopy

import numpy as np

import avstack.utils.maskfilters as maskfilters


sys.path.append("tests/")
from utilities import get_test_sensor_data


(
    obj,
    box_calib,
    lidar_calib,
    pc,
    camera_calib,
    img,
    box_2d,
    box_3d,
) = get_test_sensor_data()


def test_filter_frustum():
    frustum_filter = maskfilters.filter_points_in_image_frustum(
        pc, box_2d, camera_calib
    )
    assert sum(frustum_filter) > 0
    assert max(np.where(frustum_filter)[0]) < pc.shape[0]


def test_filter_bbox():
    bbox_filter = maskfilters.filter_points_in_object_bbox(pc, box_3d)
    assert sum(bbox_filter) > 0
    assert max(np.where(bbox_filter)[0]) < pc.shape[0]


def test_single_in_frustum_1():
    # Test underlying function
    box_3d_2 = deepcopy(box_3d)
    box_3d_2.t += 1
    assert maskfilters._item_in_frustum(box_3d, box_3d_2, camera_calib)


def test_single_in_frustum_2():
    # Test interface function
    box_3d_2 = deepcopy(box_3d)
    box_3d_2.t += 1
    assert maskfilters.filter_objects_in_frustum(box_3d, box_3d_2, camera_calib)[0, 0]
    # Test interface function
    box_3d_2 = deepcopy(box_3d)
    box_3d_2.t += 30
    assert not maskfilters.filter_objects_in_frustum(box_3d, box_3d_2, camera_calib)[
        0, 0
    ]


def test_multiple_in_frustum():
    # Test interface function
    box_3d_2 = deepcopy(box_3d)
    box_3d_2.t += 1
    box_3d_3 = deepcopy(box_3d)
    box_3d_3.t += 30
    res = maskfilters.filter_objects_in_frustum(
        box_3d, [box_3d_2, box_3d_3], camera_calib
    )
    assert res[0, 0]
    assert not res[0, 1]


def test_range_filter():
    assert sum(maskfilters.filter_points_range(pc, 0, np.inf)) == pc.shape[0]
    assert sum(maskfilters.filter_points_range(pc, 0, -1)) == 0


# def test_shadow_filter():
#     shadow_filter = maskfilters.filter_points_in_shadow(pc, box_2d, box_3d, box_calib, camera_calib)
#     assert sum(shadow_filter) > 0

#     box_3d_2 = deepcopy(box_3d)
#     box_3d_2.t[0] += 3
#     box_2d_2 = box_3d_2.project_to_2d_bbox(calib=camera_calib)
#     shadow_filter2 = maskfilters.filter_points_in_shadow(pc, box_2d_2, box_3d_2, box_calib, camera_calib)
#     assert sum(shadow_filter2) > 0
#     assert sum(shadow_filter2) > sum(shadow_filter)


def test_cone_filter():
    uv = np.array([1, 0, 0])
    cone_filter_1 = maskfilters.filter_points_in_cone(pc, uv, 15 * np.pi / 180)
    cone_filter_2 = maskfilters.filter_points_in_cone(pc, uv, 30 * np.pi / 180)
    cone_filter_3 = maskfilters.filter_points_in_cone(pc, uv, np.pi)
    assert 0 < sum(cone_filter_1) < sum(cone_filter_2) < sum(cone_filter_3)


def test_object_in_fov():
    assert maskfilters.box_in_fov(box_3d, camera_calib)
    box_3d_2 = deepcopy(box_3d)
    box_3d_2.t[2] -= 30
    assert not maskfilters.box_in_fov(box_3d_2, camera_calib)
