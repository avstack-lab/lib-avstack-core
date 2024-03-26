from typing import TYPE_CHECKING, List


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.geometry import ReferenceFrame, TransformManager

import numpy as np

from avstack.exceptions import FrameEquivalenceError
from avstack.geometry import BoundingBox2D, BoundingBox3D
from avstack.modules.perception.detections import (
    BoxDetection,
    CartesianDetection,
    RazDetection,
    RazelDetection,
    RazelRrtDetection,
)
from avstack.time import Stamp
from avstack.utils.decorators import apply_hooks

from ..assignment import gnn_single_frame_assign
from ..base import BaseModule


class _TrackingAlgorithm(BaseModule):
    def __init__(
        self,
        assign_metric="IoU",
        assign_radius=4,
        threshold_confirmed=3,
        threshold_coast=3,
        cost_threshold=-0.10,
        v_max=None,
        ID=None,
        name="tracking",
        **kwargs,
    ):
        """Base class for tracking algorithm

        Cost threshold means any cost higher than this value is rejected
        """
        super().__init__(name=name, **kwargs)

        self.ID = ID
        self.tracks = []
        self.stamp = Stamp(0, 0)
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
        self.v_max = v_max

    @property
    def tracks(self):
        return self._tracks

    @tracks.setter
    def tracks(self, tracks):
        self._tracks = tracks

    @property
    def tracks_confirmed(self):
        return DataContainer(
            [trk for trk in self.tracks if trk.confirmed], self.identifier, self.stamp
        )

    @property
    def tracks_active(self):
        return DataContainer(
            [trk for trk in self.tracks if trk.active], self.identifier, self.stamp
        )

    @apply_hooks
    def __call__(self, detections: "DataContainer", frame: "ReferenceFrame", **kwargs):
        self.stamp = detections.stamp
        tracks = self.track(detections, frame, **kwargs)
        return tracks

    def get_assignment_matrix(self, dets, tracks):
        A = np.zeros((len(dets), len(tracks)))
        for i, det_ in enumerate(dets):
            # -- pull off detection state
            if isinstance(det_, (BoxDetection)):
                det = det_.box
            elif isinstance(det_, (BoundingBox2D, BoundingBox3D)):
                det = det_
            elif isinstance(det_, (RazelRrtDetection)):
                det = det_.xyzrrt  # use the cartesian coordinates for gating
            elif isinstance(det_, (RazelDetection)):
                det = det_.xyz  # use the cartesian coordinates for gating
            elif isinstance(det_, (RazDetection)):
                det = det_.xy  # use the cartesian coordinates for gating
            elif isinstance(det_, (CartesianDetection)):
                if self.dimensions == 3:
                    det = det_.xyz
                elif self.dimensions == 2:
                    det = det_.xy
                else:
                    raise NotImplementedError(self.dimensions)
            else:
                raise NotImplementedError(type(det_))

            for j, trk_ in enumerate(tracks):
                # -- pull off track state
                if isinstance(det_, (BoxDetection, BoundingBox2D, BoundingBox3D)):
                    trk = trk_.box
                elif isinstance(det_, RazelRrtDetection):
                    trk = np.array([*trk_.x[:3], trk_.rrt])
                elif isinstance(det_, RazelDetection):
                    trk = trk_.x[:3]
                elif isinstance(det_, RazDetection):
                    trk = trk_.x[:2]
                elif isinstance(det_, CartesianDetection):
                    if self.dimensions == 3:
                        trk = trk_.x[:3]
                    elif self.dimensions == 2:
                        trk = trk_.x[:2]
                    else:
                        raise NotImplementedError(self.dimensions)
                else:
                    raise NotImplementedError(type(det_))

                # -- check reference
                if det.frame != trk.frame:
                    raise FrameEquivalenceError(det.frame, trk.frame)

                # -- gating
                if self.assign_radius is not None:
                    try:
                        dist = np.linalg.norm(trk - det)
                    except AttributeError:
                        dist = det.distance(trk)

                # -- use the metric of choice
                if self.assign_metric == "IoU":
                    cost = -det.IoU(trk)  # lower is better
                elif self.assign_metric == "center_dist":
                    cost = dist - self.assign_radius  # lower is better
                else:
                    raise NotImplementedError(self.assign_metric)

                # -- store result
                A[i, j] = cost
        return A

    def spawn_track_from_detection(self, detection):
        raise NotImplementedError("Implement in subclass")

    def track(
        self,
        detections: "DataContainer",
        frame: "ReferenceFrame",
        trks_observable: List = None,
        tm: "TransformManager" = None,
        *args,
        **kwargs,
    ):
        """ "Basic tracking implementation

        Note: detections being None means only do a prediction but don't penalize misses
        """
        t = detections.stamp
        if not trks_observable:
            trks_observable = self.tracks_active

        # -- propagation
        for trk in self.tracks_active:
            if trk.reference != frame:
                trk.change_reference(frame, tm=tm)
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
                dets = [
                    det.copy().change_reference(frame, tm=tm)
                    if det.frame != frame
                    else det
                    for det in dets
                ]

                # -- assignment with active and observable tracks
                A = self.get_assignment_matrix(dets, trks_observable)
                assign_sol = gnn_single_frame_assign(
                    A, cost_threshold=self.cost_threshold
                )
                self.last_assignment = assign_sol

                # -- update tracks with associations
                for i_det, j_trk in assign_sol.assignment_tuples:
                    trks_observable[j_trk].update(dets[i_det].z)

                # -- unassigned dets for new tracks
                for i_det in assign_sol.unassigned_rows:
                    self.tracks.append(self.spawn_track_from_detection(dets[i_det]))

                # -- tell unassigned tracks we missed them UNLESS they're whitelisted
                for j_trk in assign_sol.unassigned_cols:
                    trks_observable[j_trk].missed()

            # -- prune dead tracks -- only in a non-predict_only state
            self.tracks = [
                trk
                for trk in self.tracks
                if (trk.dt_coast < self.threshold_coast) and trk.active
            ]

        return self.tracks_confirmed
