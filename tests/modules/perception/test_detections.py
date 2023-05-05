# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-28
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""
import numpy as np

from avstack.datastructs import DataContainer
from avstack.geometry import Box3D, NominalOriginStandard, Translation
from avstack.modules.perception import detections


alg_ID = 0
alg_name = "detector-1"


def make_data_container(n_datas):
    frame = 0
    timestamp = 0
    alg_name = "detector-1"
    dets = [
        detections.CentroidDetection(alg_name, np.random.randn(3), obj_type="Car")
        for _ in range(n_datas)
    ]
    dc = DataContainer(frame, timestamp, dets, source_identifier=alg_name)
    return dc


def test_detection_container():
    n_datas = 4
    dc = make_data_container(n_datas=n_datas)
    assert len(dc) == n_datas


def test_detection_container_as_string():
    n_datas = 4
    dc = make_data_container(n_datas=n_datas)
    dc_string = detections.format_data_container_as_string(dc)
    dc_reconstruct = detections.get_data_container_from_line(dc_string)
    assert len(dc_reconstruct) == len(dc)


def test_centroid_detection():
    centroid = np.array([10, -20, 2])
    d = detections.CentroidDetection(
        source_identifier=alg_name, centroid=centroid, obj_type="Car"
    )
    assert np.all(d.centroid == centroid)


def test_razel_detection():
    raz = np.array([100, 1.0])
    d = detections.RazDetection(
        source_identifier=alg_name, raz=raz, obj_type="Car"
    )
    assert np.all(d.raz == raz)


def test_razel_detection():
    razel = np.array([100, 1.0, -0.3])
    d = detections.RazelDetection(
        source_identifier=alg_name, razel=razel, obj_type="Car"
    )
    assert np.all(d.razel == razel)


def test_razelrrt_detection():
    razelrrt = np.array([100, 1.0, -0.3, 1.23])
    d = detections.RazelRrtDetection(
        source_identifier=alg_name, razelrrt=razelrrt, obj_type="Car"
    )
    assert np.all(d.razelrrt == razelrrt)


def test_box_detection():
    h = 1
    w = 2
    l = 6
    t = [10, -4, 1]
    box = Box3D([h, w, l, t, np.quaternion(1)], NominalOriginStandard)
    d = detections.BoxDetection(source_identifier=alg_name, box=box, obj_type="Car")
    assert d.box == box


def test_lane_line_in_pixels():
    imsize = [200, 200]
    nrows = 100
    lane_left = detections.LaneLineInPixels([(r, 1) for r in range(nrows)], imsize)
    assert np.all(lane_left.coordinate_by_index(10) == np.array([10, 1]))
    assert np.all(lane_left.y == 1)


def test_lane_line_in_space():
    pt_pairs = [(i, i + 4) for i in range(20)]
    pts = [Translation([x, y, 0], origin=NominalOriginStandard) for x, y in pt_pairs]
    lane_left = detections.LaneLineInSpace(pts)
    assert np.all(lane_left.x == np.array([p[0] for p in pt_pairs]))
    assert np.all(lane_left.y == np.array([p[1] for p in pt_pairs]))


def test_lane_line_in_space_detection():
    pt_pairs_left = [(i, 4) for i in range(20)]
    pt_pairs_right = [(i + 1, -3) for i in range(20)]
    pts_left = [
        Translation([x, y, 0], origin=NominalOriginStandard) for x, y in pt_pairs_left
    ]
    lane_left = detections.LaneLineInSpace(pts_left)
    pts_right = [
        Translation([x, y, 0], origin=NominalOriginStandard) for x, y in pt_pairs_right
    ]
    lane_right = detections.LaneLineInSpace(pts_right)
    lane_center, lane_width = lane_left.compute_center_lane(lane_right)
    assert lane_center[0].x == 1
    assert lane_center[0].y == 1 / 2
    assert lane_center[-1].x == 19
    assert lane_center[-1].y == 1 / 2
    lc2, lateral_offset, yaw_offset = lane_left.compute_center_lane_and_offset(
        lane_right
    )
    assert np.all(lc2.x == lane_center.x)
    assert np.isclose(lateral_offset, 1 / 2)


def test_lane_line_in_pixels_detection():
    imsize = [200, 200]
    lane_width = 3.7
    lane_left = detections.LaneLineInPixels([(r, 1) for r in range(100)], imsize)
    lane_right = detections.LaneLineInPixels([(r, 5) for r in range(80)], imsize)
    lane_center, scaling_center = lane_left.compute_center_lane(lane_right, lane_width)
    c_true = np.array([(r, 3) for r in range(80)])
    assert np.all(lane_center._points == c_true)
    lc2, lateral_offset, yaw_offset = lane_left.compute_center_lane_and_offset(
        lane_right
    )
    assert np.isclose(lateral_offset, lane_width / (5 - 1) * (3 - 200 / 2))
    assert np.all(lc2._points == lane_center._points)


def test_distance_closest():
    pt_pairs_left = [(i, 4) for i in range(20)]
    pts_left = [
        Translation([x, y, 0], origin=NominalOriginStandard) for x, y in pt_pairs_left
    ]
    lane_left = detections.LaneLineInSpace(pts_left)
    obj_1 = Translation([10, 1, 0], origin=NominalOriginStandard)
    assert lane_left.distance_closest(obj_1) == 3
    obj_2 = Translation([10, 6, 0], origin=NominalOriginStandard)
    assert lane_left.distance_closest(obj_2) == 2


def test_object_in_lane():
    pt_pairs_left = [(i, 4) for i in range(20)]
    pt_pairs_right = [(i + 1, -3) for i in range(20)]
    pts_left = [
        Translation([x, y, 0], origin=NominalOriginStandard) for x, y in pt_pairs_left
    ]
    lane_left = detections.LaneLineInSpace(pts_left)
    pts_right = [
        Translation([x, y, 0], origin=NominalOriginStandard) for x, y in pt_pairs_right
    ]
    lane_right = detections.LaneLineInSpace(pts_right)
    obj_1 = Translation([10, 1, 0], origin=NominalOriginStandard)
    assert lane_left.object_between_lanes(lane_right, obj_1)
    obj_2 = Translation([10, 6, 0], origin=NominalOriginStandard)
    assert not lane_left.object_between_lanes(lane_right, obj_2)
    obj_3 = Translation([10, -3.2, 0], origin=NominalOriginStandard)
    assert not lane_right.object_between_lanes(lane_left, obj_3)
