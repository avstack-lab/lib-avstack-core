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
from avstack.geometry import Box2D, Box3D, ReferenceFrame
from avstack.modules.perception.detections import (
    BoxDetection,
    CentroidDetection,
    RazDetection,
    RazelDetection,
    RazelRrtDetection,
)

from ..assignment import gnn_single_frame_assign


class _TrackingAlgorithm:
    def __init__(
        self,
        assign_metric="IoU",
        assign_radius=4,
        threshold_confirmed=3,
        threshold_coast=3,
        cost_threshold=-0.10,
        v_max=None,
        save_output=False,
        save_folder="",
        check_reference=True,
        ID=None,
        **kwargs,
    ):
        """Base class for tracking algorithm

        Cost threshold means any cost higher than this value is rejected
        """
        self.ID = ID
        self.tracks = []
        self.iframe = -1
        self.frame = 0
        self.t = 0
        self.assign_metric = assign_metric
        self.assign_radius = assign_radius
        if assign_metric == "center_dist":
            assert (
                cost_threshold == 0
            ), "Cost threshold should be 0 to let dist threshold work"
        elif assign_metric == "IoU":
            assert cost_threshold < 0, "Cost threshold must be negative in IoU mode"
        else:
            raise NotImplementedError(assign_metric)
        self.cost_threshold = cost_threshold
        self.last_assignment = None
        self.threshold_confirmed = threshold_confirmed
        self.threshold_coast = threshold_coast
        self.check_reference = check_reference
        self.v_max = v_max
        self.save = save_output
        self.save_folder = save_folder
        self.save = save_output
        self.save_folder = os.path.join(save_folder, "tracking")
        if save_output:
            if os.path.exists(self.save_folder):
                shutil.rmtree(self.save_folder)
            os.makedirs(self.save_folder)

    @property
    def tracks(self):
        return self._tracks

    @tracks.setter
    def tracks(self, tracks):
        self._tracks = tracks

    @property
    def confirmed_tracks(self):
        return self.tracks_confirmed

    @property
    def tracks_confirmed(self):
        return [trk for trk in self.tracks if trk.confirmed]

    @property
    def tracks_active(self):
        return [trk for trk in self.tracks if trk.active]

    def __call__(
        self, t: float, frame: int, detections, platform: ReferenceFrame, **kwargs
    ):
        self.t = float(t)
        self.frame = int(frame)
        self.iframe += 1
        tracks = self.track(self.t, self.frame, detections, platform, **kwargs)
        track_data = DataContainer(self.frame, self.t, tracks, "tracker")
        if self.save:
            fname = os.path.join(self.save_folder, "%06i.txt" % frame)
            with open(fname, "w") as f:
                f.write(track_data.encode())
        return track_data

    def get_assignment_matrix(self, dets, tracks):
        A = np.zeros((len(dets), len(tracks)))
        for i, det_ in enumerate(dets):
            # -- pull off detection state
            if isinstance(det_, (VehicleState, BoxDetection)):
                det = det_.box
            elif isinstance(det_, (Box3D, Box2D)):
                det = det_
            elif isinstance(det_, RazelRrtDetection):
                det = det_.xyzrrt  # use the cartesian coordinates for gating
            elif isinstance(det_, RazelDetection):
                det = det_.xyz  # use the cartesian coordinates for gating
            elif isinstance(det_, RazDetection):
                det = det_.xy  # use the cartesian coordinates for gating
            elif isinstance(det_, CentroidDetection):
                if self.dimensions == 3:
                    det = det_.xyz
                elif self.dimensions == 2:
                    det = det_.xy
                else:
                    raise NotImplementedError(self.dimensions)
            else:
                raise NotImplementedError(type(det_))

            for j, trk in enumerate(tracks):
                # -- pull off track state
                if isinstance(det_, (VehicleState, BoxDetection, Box3D, Box2D)):
                    try:
                        trk = trk.as_object().box
                    except AttributeError:
                        trk = trk.box2d
                elif isinstance(det_, RazelRrtDetection):
                    trk = np.array([*trk.x[:3], trk.rrt])
                elif isinstance(det_, RazelDetection):
                    trk = trk.x[:3]
                elif isinstance(det_, RazDetection):
                    trk = trk.x[:2]
                elif isinstance(det_, CentroidDetection):
                    if self.dimensions == 3:
                        trk = trk.x[:3]
                    elif self.dimensions == 2:
                        trk = trk.x[:2]
                    else:
                        raise NotImplementedError(self.dimensions)
                else:
                    raise NotImplementedError(type(det_))

                # -- either way, change origin and use radius to filter coarsely
                try:
                    if self.check_reference:
                        if det.reference != trk.reference:
                            raise RuntimeError(
                                "Should have performed reference transformations earlier..."
                            )
                            # det = det.change_reference(trk.reference, inplace=False)
                except AttributeError as e:
                    pass

                # -- gating
                if self.assign_radius is not None:
                    if isinstance(det_, (VehicleState, BoxDetection, Box3D)):
                        dist = det.t.distance(
                            trk.t, check_reference=self.check_reference
                        )
                    else:
                        dist = np.linalg.norm(trk - det)

                # -- use the metric of choice
                if self.assign_metric == "IoU":
                    cost = -det.IoU(
                        trk, check_reference=self.check_reference
                    )  # lower is better
                elif self.assign_metric == "center_dist":
                    cost = dist - self.assign_radius  # lower is better
                else:
                    raise NotImplementedError(self.assign_metric)

                # -- store result
                A[i, j] = cost
        return A

    def spawn_track_from_detection(self, detection):
        raise NotImplementedError

    def track(
        self,
        t: float,
        frame: int,
        detections,
        platform: ReferenceFrame,
        change_in_place=False,
        *args,
        **kwargs,
    ):
        """ "Basic tracking implementation

        Note: detections being None means only do a prediction but don't penalize misses
        """
        # -- propagation
        for trk in self.tracks_active:
            trk.change_reference(platform, inplace=True)
            trk.predict(t)
            if self.v_max is not None:
                if trk.velocity.norm() > self.v_max:
                    trk.active = False

        # -- loop over each sensor providing detections
        if detections is not None:
            if not isinstance(detections, dict):
                detections = {"sensor_1": detections}
            for sensor, dets in detections.items():
                # -- change to platform reference
                if not change_in_place:
                    dets = [
                        det.change_reference(platform, inplace=False) for det in dets
                    ]
                else:
                    for det in dets:
                        det.change_reference(platform, inplace=True)

                # -- assignment with active tracks
                trks_active = self.tracks_active
                A = self.get_assignment_matrix(dets, trks_active)
                assign_sol = gnn_single_frame_assign(
                    A, cost_threshold=self.cost_threshold
                )
                self.last_assignment = assign_sol

                # -- update tracks with associations
                for i_det, j_trk in assign_sol.assignment_tuples:
                    trks_active[j_trk].update(dets[i_det].z)

                # -- unassigned dets for new tracks
                for i_det in assign_sol.unassigned_rows:
                    self.tracks.append(self.spawn_track_from_detection(dets[i_det]))

            # -- prune dead tracks -- only in a non-predict_only state
            self.tracks = [
                trk
                for trk in self.tracks
                if (trk.coast < self.threshold_coast) and trk.active
            ]

        return self.tracks_confirmed
