# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-28
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-08-11
# @Description:
"""

"""

import os
import shutil

import numpy as np

from avstack.datastructs import DataContainer
from avstack.environment.objects import VehicleState
from avstack.modules.perception.detections import BoxDetection

from ..assignment import gnn_single_frame_assign


class _TrackingAlgorithm:
    def __init__(
        self,
        assign_metric="IoU",
        assign_radius=4,
        save_output=False,
        save_folder="",
        **kwargs,
    ):
        self.iframe = -1
        self.frame = 0
        self.t = 0
        self.assign_metric = assign_metric
        self.assign_radius = assign_radius
        self.save = save_output
        self.save_folder = save_folder
        self.save = save_output
        self.save_folder = os.path.join(save_folder, "tracking")
        if save_output:
            if os.path.exists(self.save_folder):
                shutil.rmtree(self.save_folder)
            os.makedirs(self.save_folder)

    @property
    def confirmed_tracks(self):
        return self.tracks_confirmed

    def assign(self, dets, tracks):
        A = np.zeros((len(dets), len(tracks)))
        for i, b1 in enumerate(dets):
            boxa = b1.box if isinstance(b1, VehicleState) else b1
            box1 = boxa.box if isinstance(b1, BoxDetection) else boxa
            for j, b2 in enumerate(tracks):
                try:
                    box2 = b2.as_object().box
                except AttributeError:
                    box2 = b2.box2d

                # -- either way, change origin and use radius to filter coarsely
                try:
                    if box1.origin != box2.origin:
                        box1.change_origin(box2.origin)
                except AttributeError as e:
                    pass
                if self.assign_radius is not None:
                    dist = box1.t.distance(box2.t)
                    if dist > self.assign_radius:
                        continue
                # -- use the metric of choice
                if self.assign_metric == "IoU":
                    cost = -box1.IoU(box2)  # lower is better
                elif self.assign_metric == "center_dist":
                    cost = dist - self.assign_radius  # lower is better
                else:
                    raise NotImplementedError(self.assign_metric)
                # -- store result
                A[i, j] = cost
        return A

    def __call__(self, *args, **kwargs):
        self.t = kwargs.get("t")
        if self.t is None:
            raise RuntimeError("t must be passed in for this to work")
        self.frame = kwargs.get("frame")
        self.iframe += 1
        tracks = self.track(*args, **kwargs)
        if self.save:
            trk_str = "\n".join([trk.format_as("avstack") for trk in tracks])
            fname = os.path.join(self.save_folder, "%06d.txt" % self.frame)
            with open(fname, "w") as f:
                f.write(trk_str)
        track_data = DataContainer(self.frame, self.t, tracks, "tracker")
        return track_data


class _BasicBoxTracker(_TrackingAlgorithm):
    def __init__(
        self,
        threshold_confirmed=3,
        threshold_coast=3,
        v_max=60,
        assign_metric="center_dist",
        assign_radius=4,
        **kwargs,
    ):
        self.tracks = []
        self.threshold_confirmed = threshold_confirmed
        self.threshold_coast = threshold_coast
        self.v_max = v_max
        super().__init__(assign_metric, assign_radius, **kwargs)

    @property
    def tracks_confirmed(self):
        return [trk for trk in self.tracks if trk.n_updates >= self.threshold_confirmed]

    @property
    def tracks_active(self):
        return [trk for trk in self.tracks if trk.active]

    def spawn_track_from_detection(self, detection):
        raise NotImplementedError

    def track(self, t, detections_nd, *args, **kwargs):
        """
        :detections_nd

        Use IoU for association

        HACK: if detections_nd is a dictionary, then it's from multiple sensors...
        so we need to run over assignment and update for each one
        """
        # -- propagation
        for trk in self.tracks:
            trk.predict(t)
            if np.linalg.norm(trk.velocity) > self.v_max:
                trk.active = False

        # -- loop over each sensor providing detections
        if not isinstance(detections_nd, dict):
            detections_nd = {"sensor_1": detections_nd}
        for sensor, detections in detections_nd.items():
            # -- assignment with active tracks
            trks_active = self.tracks_active
            A = self.assign(detections, trks_active)
            assign_sol = gnn_single_frame_assign(A, cost_threshold=-0.10)

            # -- update tracks with associations
            for i_det, j_trk in assign_sol.assignment_tuples:
                trks_active[j_trk].update(detections[i_det].box3d)

            # -- unassigned dets for new tracks
            for i_det in assign_sol.unassigned_rows:
                self.tracks.append(self.spawn_track_from_detection(detections[i_det]))

        # -- prune dead tracks
        self.tracks = [
            trk
            for trk in self.tracks
            if (trk.coast < self.threshold_coast) and trk.active
        ]

        return self.tracks_confirmed
