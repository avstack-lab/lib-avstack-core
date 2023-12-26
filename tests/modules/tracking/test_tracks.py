# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-23
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-28
# @Description:
"""

"""
import json
import sys

import numpy as np

from avstack.datastructs import DataContainer
from avstack.geometry import GlobalOrigin3D
from avstack.geometry.transformations import cartesian_to_spherical, xyzvel_to_razelrrt
from avstack.modules.tracking import tracks


sys.path.append("tests/")
from utilities import camera_calib, get_object_global


def test_raz_track():
    random_object = get_object_global(1)
    t0 = 1.25
    box3d = random_object.box
    obj_type = random_object.obj_type
    razel = cartesian_to_spherical(box3d.t.x)
    random_track_1 = tracks.XyFromRazTrack(t0, razel[:2], GlobalOrigin3D, obj_type)
    random_track_2 = json.loads(random_track_1.encode(), cls=tracks.TrackDecoder)
    assert np.allclose(random_track_1.x, random_track_2.x)
    assert np.allclose(random_track_1.P, random_track_2.P)


def test_razel_track():
    random_object = get_object_global(1)
    t0 = 1.25
    box3d = random_object.box
    obj_type = random_object.obj_type
    razel = cartesian_to_spherical(box3d.t.x)
    random_track_1 = tracks.XyzFromRazelTrack(t0, razel, GlobalOrigin3D, obj_type)
    random_track_2 = json.loads(random_track_1.encode(), cls=tracks.TrackDecoder)
    assert np.allclose(random_track_1.x, random_track_2.x)
    assert np.allclose(random_track_1.P, random_track_2.P)


def test_razelrrt_track():
    random_object = get_object_global(1)
    t0 = 1.25
    obj_type = random_object.obj_type
    razelrrt = xyzvel_to_razelrrt(
        np.array([*random_object.position.x, *random_object.velocity.x])
    )
    random_track_1 = tracks.XyzFromRazelRrtTrack(t0, razelrrt, GlobalOrigin3D, obj_type)
    assert np.isclose(random_track_1.rrt, razelrrt[3])
    random_track_2 = json.loads(random_track_1.encode(), cls=tracks.TrackDecoder)
    assert np.allclose(random_track_1.x, random_track_2.x)
    assert np.allclose(random_track_1.P, random_track_2.P)


def random_razel_track(seed):
    t0 = 1.25
    random_object = get_object_global(seed)
    box3d = random_object.box
    obj_type = random_object.obj_type
    razel = cartesian_to_spherical(box3d.t.x)
    return tracks.XyzFromRazelTrack(t0, razel, GlobalOrigin3D, obj_type)


def test_grouptrack_track():
    state = random_razel_track(1)
    members = [random_razel_track(i) for i in range(3)]
    gt = tracks.GroupTrack(state=state, members=members)


def test_boxtrack3d_encode_decode():
    random_object = get_object_global(1)
    t0 = 1.25
    box3d = random_object.box
    obj_type = random_object.obj_type
    random_track_1 = tracks.BasicBoxTrack3D(t0, box3d, GlobalOrigin3D, obj_type)
    random_track_2 = json.loads(random_track_1.encode(), cls=tracks.TrackDecoder)
    assert np.allclose(random_track_1.x, random_track_2.x)
    assert np.allclose(random_track_1.P, random_track_2.P)


def test_boxtrack2d_encode_decode():
    random_object = get_object_global(1)
    t0 = 1.25
    box2d = random_object.box.project_to_2d_bbox(camera_calib)
    obj_type = random_object.obj_type
    random_track_1 = tracks.BasicBoxTrack2D(t0, box2d, GlobalOrigin3D, obj_type)
    random_track_2 = json.loads(random_track_1.encode(), cls=tracks.TrackDecoder)
    assert np.allclose(random_track_1.x, random_track_2.x)
    assert np.allclose(random_track_1.P, random_track_2.P)


def test_grouptrack_encode_decode():
    state = random_razel_track(1)
    members = [random_razel_track(i) for i in range(3)]
    gt_1 = tracks.GroupTrack(state=state, members=members)
    gt_2 = json.loads(gt_1.encode(), cls=tracks.TrackDecoder)
    assert np.allclose(gt_1.state.x, gt_2.state.x)
    assert len(gt_1.members) == len(gt_2.members)


def test_trackcontainer_encode_decode():
    frame = 0
    timestamp = 0.0
    n_tracks = 14
    trks = [
        tracks.BasicBoxTrack3D(
            timestamp, get_object_global(i).box, GlobalOrigin3D, "car"
        )
        for i in range(n_tracks)
    ]
    dc_1 = DataContainer(frame, timestamp, trks, source_identifier="tracks-1")
    dc_2 = json.loads(dc_1.encode(), cls=tracks.TrackContainerDecoder)
    assert len(dc_1) == len(dc_2)
