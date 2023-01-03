# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-27
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-27
# @Description:
"""

"""
from copy import copy, deepcopy
import numpy as np
import quaternion
from avstack.geometry import bbox, NominalOriginStandard
from avstack.modules import tracking
from avstack.modules import fusion


def test_track_to_track_CI():
    fuser = fusion.BoxTrackToBoxTrackFusion3D()
    t0 = 0.0
    frame = 0
    obj_type = 'car'
    framerate = 10
    box1 = bbox.Box3D([2,2,4,[-1,-2,-3],np.quaternion(1)], NominalOriginStandard)
    box2 = deepcopy(box1)
    track_1 = tracking.tracker3d.BasicBoxTrack3D(t0, box1, obj_type, framerate)
    track_2 = tracking.tracker3d.BasicBoxTrack3D(t0, box2, obj_type, framerate)
    track_fused = fuser(frame, [track_1], [track_2])[0]
    assert fuser.ID_registry == {track_1.ID:{track_2.ID:track_fused.ID}}
    assert track_fused.box3d == box1
