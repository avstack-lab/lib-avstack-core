# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-23
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-28
# @Description:
"""

"""
import sys

import numpy as np

from avstack.datastructs import DataContainer
from avstack.modules.tracking import tracks


sys.path.append("tests/")
from utilities import camera_calib, get_object_global


def test_boxtrack3d_as_string():
    random_object = get_object_global(1)
    t0 = 1.25
    box3d = random_object.box
    obj_type = random_object.obj_type
    random_track = tracks.BasicBoxTrack3D(t0, box3d, obj_type)
    trk_string = random_track.format_as_string()
    random_track_reconstruct = tracks.get_track_from_line(trk_string)
    assert np.allclose(random_track.kf.x, random_track_reconstruct.kf.x)
    assert np.allclose(random_track.kf.P, random_track_reconstruct.kf.P)


def test_boxtrack2d_as_string():
    random_object = get_object_global(1)
    t0 = 1.25
    box2d = random_object.box.project_to_2d_bbox(camera_calib)
    obj_type = random_object.obj_type
    random_track = tracks.BasicBoxTrack2D(t0, box2d, obj_type)
    trk_string = random_track.format_as_string()
    random_track_reconstruct = tracks.get_track_from_line(trk_string)
    assert np.allclose(random_track.kf.x, random_track_reconstruct.kf.x)
    assert np.allclose(random_track.kf.P, random_track_reconstruct.kf.P)


def test_trackcontainer_as_string():
    frame = 0
    timestamp = 0.0
    n_tracks = 14
    trks = [
        tracks.BasicBoxTrack3D(timestamp, get_object_global(i).box, "car")
        for i in range(n_tracks)
    ]
    dc = DataContainer(frame, timestamp, trks, source_identifier="tracks-1")
    dc_string = tracks.format_data_container_as_string(dc)
    dc_reconstruct = tracks.get_data_container_from_line(dc_string)
    assert len(dc_reconstruct) == len(dc)
