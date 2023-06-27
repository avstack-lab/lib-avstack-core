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

from avstack.geometry import Attitude, GlobalOrigin3D, Position, bbox
from avstack.modules import fusion, tracking


def test_track_to_track_CI():
    fuser = fusion.BoxTrackToBoxTrackFusion3D()
    t0 = 0.0
    frame = 0
    obj_type = "car"
    pos = Position(np.array([-1, -2, -3]), GlobalOrigin3D)
    rot = Attitude(np.quaternion(1), GlobalOrigin3D)
    box1 = bbox.Box3D(pos, rot, [2, 2, 4])
    box2 = deepcopy(box1)
    track_1 = tracking.tracker3d.BasicBoxTrack3D(t0, box1, GlobalOrigin3D, obj_type)
    track_2 = tracking.tracker3d.BasicBoxTrack3D(t0, box2, GlobalOrigin3D, obj_type)
    track_fused = fuser([track_1], [track_2], frame=frame)[0]
    assert fuser.ID_registry == {track_1.ID: {track_2.ID: track_fused.ID}}
    assert track_fused.box3d.allclose(box1)
