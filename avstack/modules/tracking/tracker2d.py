# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-27
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-27
# @Description:
"""

"""

import numpy as np

from avstack.geometry import Box2D

from ._sort import Sort
from .base import _TrackingAlgorithm
from .tracks import BasicBoxTrack2D


# ==============================================================
# BASIC BOX TRACKER
# ==============================================================


class PassthroughTracker2D(_TrackingAlgorithm):
    def __init__(self, **kwargs):
        super().__init__("PassthroughTracker")

    def track(self, t, frame, detections, platform, **kwargs):
        tracks = []
        t = detections.timestamp
        # frame = detections_2d.frame
        for det in detections:
            trk = BasicBoxTrack2D(
                t0=t,
                box2d=det.box2d,
                obj_type=det.obj_type,
                ID_force=None,
                v=None,
                P=np.eye(6),  # fake this
                t=t,
                coast=0,
                n_updates=1,
                age=1,
            )
            tracks.append(trk)
        return tracks


class BasicBoxTracker2D(_TrackingAlgorithm):
    def __init__(
        self,
        threshold_confirmed=2,
        threshold_coast=4,
        v_max=200,  # pixels per second
        cost_threshold=-0.10,
        **kwargs,
    ):
        super().__init__(
            assign_metric="IoU",
            assign_radius=None,
            threshold_confirmed=threshold_confirmed,
            threshold_coast=threshold_coast,
            cost_threshold=cost_threshold,
            v_max=v_max,
            **kwargs,
        )

    def spawn_track_from_detection(self, detection):
        return BasicBoxTrack2D(
            t0=self.t,
            box2d=detection.box2d,
            reference=detection.reference,
            obj_type=detection.obj_type,
        )


class SortTracker2D(_TrackingAlgorithm):
    def __init__(
        self,
        assign_metric="IoU",
        assign_radius=4,
        save_output=False,
        save_folder="",
        **kwargs,
    ):
        super().__init__(
            assign_metric, assign_radius, save_output, save_folder, **kwargs
        )
        self.sort_algorithm = Sort()

    def track(self, t, frame, detections, platform, **kwargs):
        """Just a wrapping to the sort algorithm

        sort inputs: [xmin, ymin, xmax, ymax, score]

        sort state vector: [u, v, s, r, udot, vdot, sdot]
            - u, v are (x, y) of target center
            - s, r are scale (area) and aspect ratio of box

        sort outputs:
            track_bbs_ids: [xmin, ymin, xmax, ymax, object_ID]
            also "trackers"
        """
        if detections is None:
            raise NotImplementedError("Need to implement this for prediction")

        # inputs wrap to SORT format
        dets_sort = [det.box2d.box2d + [float(det.score)] for det in detections]
        calibs = [det.box2d.calibration for det in detections]
        obj_types = [det.obj_type for det in detections]
        ts = [detections.timestamp] * len(detections)
        _, trackers = self.sort_algorithm.update(dets_sort, calibs, obj_types, ts)
        # outputs wrap to AVstack format
        tracks_avstack = []
        for tracker in trackers:
            box2d = Box2D(tracker.get_state()[0, :], tracker.calibration)
            trk = BasicBoxTrack2D(
                t0=tracker.t0,
                box2d=box2d,
                obj_type=tracker.obj_type,
                ID_force=tracker.id,
                v=tracker.kf.x[4:6, 0],
                P=np.eye(6),  # fake this
                t=tracker.t,
                coast=tracker.time_since_update,
                n_updates=tracker.hits,
                age=tracker.age,
            )
            tracks_avstack.append(trk)
        return tracks_avstack
