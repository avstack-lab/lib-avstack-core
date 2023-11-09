# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-27
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-27
# @Description:
"""

"""
from copy import deepcopy

import numpy as np

from avstack.geometry import Attitude, GlobalOrigin3D, Position, bbox
from avstack.modules import clustering, fusion, tracking


def get_two_tracks():
    t0 = 0.0
    obj_type = "car"
    pos = Position(np.array([-1, -2, -3]), GlobalOrigin3D)
    rot = Attitude(np.quaternion(1), GlobalOrigin3D)
    box1 = bbox.Box3D(pos, rot, [2, 2, 4])
    box2 = deepcopy(box1)
    track_1 = tracking.tracker3d.BasicBoxTrack3D(t0, box1, GlobalOrigin3D, obj_type)
    track_2 = tracking.tracker3d.BasicBoxTrack3D(t0, box2, GlobalOrigin3D, obj_type)
    return track_1, track_2


def test_track_to_track_CI():
    fuser = fusion.BoxTrackToBoxTrackFusion3D()
    frame = 0
    track_1, track_2 = get_two_tracks()
    track_fused = fuser([track_1], [track_2], frame=frame)[0]
    assert fuser.ID_registry == {track_1.ID: {track_2.ID: track_fused.ID}}
    assert track_fused.box3d.allclose(track_1.box)


def test_no_fusion():
    fuser = fusion.track_to_track.NoFusion()
    clusterer = clustering.NoClustering()
    track_1, track_2 = get_two_tracks()
    clusters = clusterer({0: [track_1, track_2]}, frame=0, timestamp=0)
    tracks_fused = [fuser(cluster) for cluster in clusters]
    assert tracks_fused == [track_1, track_2]
