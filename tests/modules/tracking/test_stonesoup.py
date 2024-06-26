import sys
from datetime import datetime

import numpy as np

from avstack.geometry import GlobalOrigin3D
from avstack.modules.tracking.stonesoup import StoneSoupKalmanTracker2DBox


sys.path.append("tests/")
sys.path.append("tests/modules/tracking")
from track_utils import make_2d_tracking_data


def test_stone_soup_2d_box_tracker():
    tracker = StoneSoupKalmanTracker2DBox(t0=datetime.now())

    platform = GlobalOrigin3D
    n_targs = 4
    dt = 0.05
    n_frames = 100
    dets_2d_all, cam_calib = make_2d_tracking_data(
        dt=dt, n_frames=n_frames, n_targs=n_targs
    )
    for frame, dets_2d in enumerate(dets_2d_all):
        _ = tracker(
            t=frame * dt,
            frame=frame,
            detections=dets_2d,
            platform=platform,
            calibration=cam_calib,
            identifier="tracker-1",
        )
    assert len(tracker.tracks_active) == len(dets_2d_all[-1])
    for trk in tracker.tracks_active:
        trk_center = np.array(
            [
                trk.state_vector[0] + trk.state_vector[4] / 2,
                trk.state_vector[2] + trk.state_vector[5] / 2,
            ]
        )
        for det in dets_2d_all[-1]:
            if (
                np.linalg.norm(trk_center - det.box.center) < 25
            ):  # NOTE: this is a little high
                break
        else:
            raise
