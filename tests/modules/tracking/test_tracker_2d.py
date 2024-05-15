import sys

import numpy as np

from avstack.geometry import GlobalOrigin3D
from avstack.modules.tracking import tracker2d


sys.path.append("tests/")
sys.path.append("tests/modules/tracking")
from track_utils import make_2d_tracking_data, run_tracker


def test_xy_tracker():
    tracker = tracker2d.BasicXyTracker(threshold_coast=5)
    run_tracker(tracker=tracker, det_type="xy")


def test_raz_tracker():
    tracker = tracker2d.BasicRazTracker(threshold_coast=5)
    run_tracker(tracker=tracker, det_type="raz")


def test_basic_box_tracker_2d():
    platform = GlobalOrigin3D
    dt = 0.1
    n_targs = 4
    n_frames = 30
    dets_2d_all, cam_calib = make_2d_tracking_data(
        dt=dt, n_frames=n_frames, n_targs=n_targs
    )
    tracker = tracker2d.BasicBoxTracker2D()

    for frame, dets_2d in enumerate(dets_2d_all):
        _ = tracker(
            t=frame * dt,
            frame=frame,
            detections=dets_2d,
            platform=platform,
            identifier="tracker-1",
        )

    assert len(tracker.tracks_active) == len(dets_2d_all[-1])
    for trk in tracker.tracks_active:
        for det in dets_2d_all[-1]:
            if (
                np.linalg.norm(trk.box2d.center - det.box.center) < 25
            ):  # NOTE: this is a little high
                break
        else:
            raise
