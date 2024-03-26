import numpy as np

from avstack.config import MODELS

from ..assignment import build_A_from_iou, gnn_single_frame_assign, greedy_assignment
from .base import _TrackingAlgorithm
from .tracks import (
    BasicBoxTrack3D,
    BasicJointBoxTrack,
    XyzFromRazelRrtTrack,
    XyzFromRazelTrack,
    XyzFromXyzTrack,
)


@MODELS.register_module()
class GroundTruthTracker(_TrackingAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_ground_truth = True

    def __call__(self, t, frame, detections, ground_truth, **kwargs):
        return ground_truth.objects


@MODELS.register_module()
class BasicBoxTracker3D(_TrackingAlgorithm):
    dimensions = 3

    def __init__(
        self,
        threshold_confirmed=3,
        threshold_coast=3,
        v_max=60,  # meters per second
        assign_metric="center_dist",
        assign_radius=4,
        cost_threshold=0,
        **kwargs,
    ):
        super().__init__(
            assign_metric=assign_metric,
            assign_radius=assign_radius,
            threshold_confirmed=threshold_confirmed,
            threshold_coast=threshold_coast,
            cost_threshold=cost_threshold,
            v_max=v_max,
            **kwargs,
        )

    def spawn_track_from_detection(self, detection):
        return BasicBoxTrack3D(
            self.stamp,
            box3d=detection.box3d,
            reference=detection.box3d.reference,
            obj_class=detection.obj_class,
        )


@MODELS.register_module()
class BasicBoxTrackerFusion3Stage(_TrackingAlgorithm):
    dimensions = 3

    def __init__(
        self,
        threshold_confirmed_3d=3,
        threshold_confirmed_2d=3,
        threshold_coast_3d=3,
        threshold_coast_2d=3,
        threshold_assoc_2=3,
        threshold_assoc_3=-0.2,
        **kwargs,
    ):
        self.tracks = []
        self.threshold_confirmed_3d = threshold_confirmed_3d
        self.threshold_confirmed_2d = threshold_confirmed_2d
        self.threshold_coast_3d = threshold_coast_3d
        self.threshold_coast_2d = threshold_coast_2d
        self.threshold_assoc_2 = threshold_assoc_2
        self.threshold_assoc_3 = threshold_assoc_3

        self.calib_cam = None
        super().__init__(**kwargs)

    @property
    def tracks_confirmed(self):
        return [
            trk
            for trk in self.tracks
            if (trk.n_updates_3d >= self.threshold_confirmed_3d)
            and (trk.n_updates_2d >= self.threshold_confirmed_2d)
        ]

    def track(self, detections, platform, **kwargs):
        """
        :detections_2d
        :detections_3d

        Use 3-stage association from EagerMOT

        ASSUMPTION: only 1 camera used for now
        """
        t = detections.stamp
        for trk in self.tracks:
            trk.change_reference(platform, inplace=True)
            trk.predict(t)

        if detections is not None:
            # -- change reference frame
            detections_2d = [
                det.change_reference(platform, inplace=False)
                for det in detections["2d"]
            ]
            detections_3d = [
                det.change_reference(platform, inplace=False)
                for det in detections["3d"]
            ]

            # -- STAGE 1: assignment between detections
            boxes_2d = [det.box2d for det in detections_2d]
            obj_classs_2d = [det.obj_class for det in detections_2d]
            boxes_3d = [det.box3d for det in detections_3d]
            obj_classs_3d = [det.obj_class for det in detections_3d]

            if len(detections_2d) > 0:
                if self.calib_cam is None:
                    self.calib_cam = detections_2d[0].box.calibration
                boxes_3d_in_2d = [
                    box.project_to_2d_bbox(calib=self.calib_cam) for box in boxes_3d
                ]
                A = build_A_from_iou(boxes_2d, boxes_3d_in_2d)
                assign_sol_1 = gnn_single_frame_assign(A, cost_threshold=-0.10)
                lone_2d = [boxes_2d[i_2d] for i_2d in assign_sol_1.unassigned_rows]
                lone_2d_to_det_map = {
                    i: k for i, k in enumerate(assign_sol_1.unassigned_rows)
                }
                lone_3d = [boxes_3d[i_3d] for i_3d in assign_sol_1.unassigned_cols]
                lone_3d_to_det_map = {
                    i: k for i, k in enumerate(assign_sol_1.unassigned_cols)
                }
                fused_detections = [
                    (boxes_2d[i], boxes_3d[j])
                    for i, j in assign_sol_1.assignment_tuples
                ]
                fused_to_det_map = {
                    i: (k1, k2)
                    for i, (k1, k2) in enumerate(assign_sol_1.assignment_tuples)
                }
            else:
                lone_2d = []
                lone_3d = boxes_3d
                lone_2d_to_det_map = {}
                lone_3d_to_det_map = {i: i for i in range(len(boxes_3d))}
                fused_detections = []
                fused_to_det_map = {}

            # -- STAGE 2: assignment between fused and lone 3d to tracks
            i = 0
            A = np.inf * np.ones(
                (len(fused_detections) + len(lone_3d), len(self.tracks))
            )
            for ds in (fused_detections, lone_3d):
                for d_ in ds:
                    if i < len(fused_detections):
                        d_ = d_[1]  # index into tuple for 3D box
                    for j, t in enumerate(self.tracks):
                        if t.box3d is None:
                            continue
                        Bi = np.array(
                            [
                                t.box3d.t[0],
                                t.box3d.t[1],
                                t.box3d.t[2],
                                t.box3d.h,
                                t.box3d.w,
                                t.box3d.l,
                            ]
                        )
                        Bj = np.array([d_.t[0], d_.t[1], d_.t[2], d_.h, d_.w, d_.l])
                        alpha = 2 - np.cos((t.yaw - d_.yaw) % (np.pi / 2))
                        A[i, j] = np.linalg.norm(Bi - Bj) * alpha
                    i += 1
            assign_sol_2 = greedy_assignment(A, threshold=self.threshold_assoc_2)
            lone_fused = [
                fused_detections[j]
                for j in assign_sol_2.unassigned_rows
                if j < len(fused_detections)
            ]
            lone_fused_to_det_map = {
                i: fused_to_det_map[k]
                for i, k in enumerate(assign_sol_2.unassigned_rows)
                if k < len(fused_detections)
            }
            lone_tracks = [self.tracks[k] for k in assign_sol_2.unassigned_cols]
            lone_track_to_track_map = {
                i: k for i, k in enumerate(assign_sol_2.unassigned_cols)
            }

            # -- STAGE 3: assignment between lone fused and lone 2d to tracks via 2D
            i = 0
            A = np.zeros((len(lone_fused) + len(lone_2d), len(lone_tracks)))
            for ds in (lone_fused, lone_2d):
                for d_ in ds:
                    if i < len(lone_fused):
                        d_ = d_[0]  # index into tuple for 2D box
                    for j, t in enumerate(lone_tracks):
                        if t.box2d is None:
                            t_box = t.box3d.project_to_2d_bbox(self.calib_cam)
                        else:
                            t_box = t.box2d
                        A[i, j] = -d_.IoU(t_box)
                    i += 1
            assign_sol_3 = greedy_assignment(A, threshold=self.threshold_assoc_3)

            # -- update tracks
            # ----- from assignment 2
            for i_det, j_trk in assign_sol_2.assignment_tuples:
                if i_det < len(fused_detections):
                    d_2 = fused_detections[i_det]
                    o2d = obj_classs_2d[fused_to_det_map[i_det][0]]
                    o3d = obj_classs_3d[fused_to_det_map[i_det][1]]
                else:
                    d_2 = lone_3d[i_det - len(fused_detections)]
                    o3d = obj_classs_3d[
                        lone_3d_to_det_map[i_det - len(fused_detections)]
                    ]
                self.tracks[j_trk].update(d_2, o3d, platform)
            # ----- from assignment 3
            for i_det, j_trk in assign_sol_3.assignment_tuples:
                if i_det < len(lone_fused):
                    d_3 = lone_fused[i_det]
                    o2d = obj_classs_2d[lone_fused_to_det_map[i_det][0]]
                    o3d = obj_classs_3d[lone_fused_to_det_map[i_det][1]]
                else:
                    d_3 = lone_2d[i_det - len(lone_fused)]
                    o2d = obj_classs_2d[lone_2d_to_det_map[i_det - len(lone_fused)]]
                self.tracks[lone_track_to_track_map[j_trk]].update(d_3, o2d, platform)

            # -- unassigned dets for new tracks
            # ----- unassigned from the 3D to 3D step
            for i_det in assign_sol_2.unassigned_rows:
                if i_det < len(fused_detections):
                    continue
                else:
                    d3d = lone_3d[i_det - len(fused_detections)]
                    o3d = obj_classs_3d[
                        lone_3d_to_det_map[i_det - len(fused_detections)]
                    ]
                    self.tracks.append(
                        BasicJointBoxTrack(self.stamp, None, d3d, platform, o3d)
                    )
            # ----- unassigned from the 2D to 2D step
            for i_det in assign_sol_3.unassigned_rows:
                if i_det < len(lone_fused):
                    d2d, d3d = lone_fused[i_det]
                    o2d, o3d = (
                        obj_classs_2d[lone_fused_to_det_map[i_det][0]],
                        obj_classs_3d[lone_fused_to_det_map[i_det][1]],
                    )
                    self.tracks.append(
                        BasicJointBoxTrack(self.stamp, d2d, d3d, platform, o2d)
                    )
                else:
                    d2d = lone_2d[i_det - len(lone_fused)]
                    o2d = obj_classs_2d[lone_2d_to_det_map[i_det - len(lone_fused)]]
                    self.tracks.append(
                        BasicJointBoxTrack(self.stamp, d2d, None, platform, o2d)
                    )

            # -- prune dead tracks
            self.tracks = [
                trk
                for trk in self.tracks
                if (trk.coast_2d < self.threshold_coast_2d)
                and (trk.coast_3d < self.threshold_coast_3d)
            ]

        return self.tracks_confirmed


class _BaseCenterTracker(_TrackingAlgorithm):
    def __init__(
        self,
        threshold_confirmed=3,
        threshold_coast=3,
        v_max=60,  # meters per second
        assign_metric="center_dist",
        assign_radius=8,
        **kwargs,
    ):
        super().__init__(
            assign_metric=assign_metric,
            assign_radius=assign_radius,
            threshold_confirmed=threshold_confirmed,
            threshold_coast=threshold_coast,
            cost_threshold=0,  # bc we are subtracting off assign radius
            v_max=v_max,
            **kwargs,
        )

    def spawn_track_from_detection(self, detection):
        raise NotImplementedError


@MODELS.register_module()
class BasicXyzTracker(_BaseCenterTracker):
    dimensions = 3

    def spawn_track_from_detection(self, detection):
        return XyzFromXyzTrack(
            t0=self.stamp,
            xyz=detection.xyz,
            reference=detection.reference,
            obj_class=detection.obj_class,
        )


@MODELS.register_module()
class BasicRazelTracker(_BaseCenterTracker):
    dimensions = 3

    def spawn_track_from_detection(self, detection):
        return XyzFromRazelTrack(
            t0=self.stamp,
            razel=detection.razel,
            reference=detection.reference,
            obj_class=detection.obj_class,
        )


@MODELS.register_module()
class BasicRazelRrtTracker(_BaseCenterTracker):
    dimensions = 3

    def spawn_track_from_detection(self, detection):
        return XyzFromRazelRrtTrack(
            t0=self.stamp,
            razelrrt=detection.razelrrt,
            reference=detection.reference,
            obj_class=detection.obj_class,
        )
