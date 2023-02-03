# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-27
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-24
# @Description:
"""

"""

from copy import deepcopy

import numpy as np
from filterpy.kalman import KalmanFilter

from avstack.environment.objects import VehicleState
from avstack.geometry import Box2D, Box3D
from avstack.geometry import transformations as tforms
from avstack.modules.perception.detections import BoxDetection

from ..assignment import build_A_from_iou, gnn_single_frame_assign, greedy_assignment
from . import libraries
from .base import _TrackingAlgorithm


class GroundTruthTracker(_TrackingAlgorithm):
    def __init__(self, **kwargs):
        self.is_ground_truth = True
        super().__init__(**kwargs)

    def __call__(self, ground_truth, *args, **kwargs):
        return ground_truth.objects


# ==============================================================
# BASIC BOX TRACKER
# ==============================================================

class BasicBoxTracker(_TrackingAlgorithm):
    def __init__(
        self,
        framerate,
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
        self.framerate = framerate
        self.dt = 1.0 / self.framerate
        self.v_max = v_max
        super().__init__(assign_metric, assign_radius, **kwargs)

    @property
    def tracks_confirmed(self):
        return [trk for trk in self.tracks if trk.n_updates >= self.threshold_confirmed]

    @property
    def tracks_active(self):
        return [trk for trk in self.tracks if trk.active]

    def track(self, detections_3d, *args, metric="center_dist", **kwargs):
        """
        :detections_3d

        Use IoU for association

        HACK: if detections_3d is a dictionary, then it's from multiple sensors...
        so we need to run over assignment and update for each one
        """
        # -- propagation
        self.t = self.frame * self.dt
        for trk in self.tracks:
            trk.predict()
            if np.linalg.norm(trk.velocity) > self.v_max:
                trk.active = False

        # -- loop over each sensor providing detections
        if not isinstance(detections_3d, dict):
            detections_3d = {"sensor_1": detections_3d}
        for sensor, detections in detections_3d.items():
            # -- assignment with active tracks
            trks_active = self.tracks_active
            A = self.assign(detections, trks_active)
            assign_sol = gnn_single_frame_assign(A, cost_threshold=-0.10)

            # -- put detections into track vector format
            det_vectors = []

            # -- update tracks with associations
            for i_det, j_trk in assign_sol.assignment_tuples:
                trks_active[j_trk].update(detections[i_det].box3d)

            # -- unassigned dets for new tracks
            for i_det in assign_sol.unassigned_rows:
                self.tracks.append(
                    BasicBoxTrack3D(
                        self.t,
                        detections[i_det].box3d,
                        detections[i_det].obj_type,
                        self.framerate,
                    )
                )

        # -- prune dead tracks
        self.tracks = [
            trk
            for trk in self.tracks
            if (trk.coast < self.threshold_coast) and trk.active
        ]

        return self.tracks_confirmed


class BasicBoxTrackerFusion3Stage(_TrackingAlgorithm):
    def __init__(
        self,
        framerate,
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
        self.framerate = framerate
        self.dt = 1.0 / self.framerate

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

    def track(self, detections_2d, detections_3d, *args, **kwargs):
        """
        :detections_2d
        :detections_3d

        Use 3-stage association from EagerMOT

        ASSUMPTION: only 1 camera used for now
        """
        self.t = self.iframe * self.dt

        for trk in self.tracks:
            trk.predict()

        # -- STAGE 1: assignment between detections
        boxes_2d = [det.box2d for det in detections_2d]
        obj_types_2d = [det.obj_type for det in detections_2d]
        boxes_3d = [det.box3d for det in detections_3d]
        obj_types_3d = [det.obj_type for det in detections_3d]

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
                (boxes_2d[i], boxes_3d[j]) for i, j in assign_sol_1.assignment_tuples
            ]
            fused_to_det_map = {
                i: (k1, k2) for i, (k1, k2) in enumerate(assign_sol_1.assignment_tuples)
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
        A = np.inf * np.ones((len(fused_detections) + len(lone_3d), len(self.tracks)))
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
                o2d = obj_types_2d[fused_to_det_map[i_det][0]]
                o3d = obj_types_3d[fused_to_det_map[i_det][1]]
            else:
                d_2 = lone_3d[i_det - len(fused_detections)]
                o3d = obj_types_3d[lone_3d_to_det_map[i_det - len(fused_detections)]]
            self.tracks[j_trk].update(d_2, o3d)
        # ----- from assignment 3
        for i_det, j_trk in assign_sol_3.assignment_tuples:
            if i_det < len(lone_fused):
                d_3 = lone_fused[i_det]
                o2d = obj_types_2d[lone_fused_to_det_map[i_det][0]]
                o3d = obj_types_3d[lone_fused_to_det_map[i_det][1]]
            else:
                d_3 = lone_2d[i_det - len(lone_fused)]
                o2d = obj_types_2d[lone_2d_to_det_map[i_det - len(lone_fused)]]
            self.tracks[lone_track_to_track_map[j_trk]].update(d_3, o2d)

        # -- unassigned dets for new tracks
        # ----- unassigned from the 3D to 3D step
        for i_det in assign_sol_2.unassigned_rows:
            if i_det < len(fused_detections):
                continue
            else:
                d3d = lone_3d[i_det - len(fused_detections)]
                o3d = obj_types_3d[lone_3d_to_det_map[i_det - len(fused_detections)]]
                self.tracks.append(
                    BasicJointBoxTrack(self.t, None, d3d, o3d, self.framerate)
                )
        # ----- unassigned from the 2D to 2D step
        for i_det in assign_sol_3.unassigned_rows:
            if i_det < len(lone_fused):
                d2d, d3d = lone_fused[i_det]
                o2d, o3d = (
                    obj_types_2d[lone_fused_to_det_map[i_det][0]],
                    obj_types_3d[lone_fused_to_det_map[i_det][1]],
                )
                self.tracks.append(
                    BasicJointBoxTrack(self.t, d2d, d3d, o2d, self.framerate)
                )
            else:
                d2d = lone_2d[i_det - len(lone_fused)]
                o2d = obj_types_2d[lone_2d_to_det_map[i_det - len(lone_fused)]]
                self.tracks.append(
                    BasicJointBoxTrack(self.t, d2d, None, o2d, self.framerate)
                )

        # -- prune dead tracks
        self.tracks = [
            trk
            for trk in self.tracks
            if (trk.coast_2d < self.threshold_coast_2d)
            and (trk.coast_3d < self.threshold_coast_3d)
        ]

        return self.tracks_confirmed


class BasicBoxTrack3D:
    ID_counter = 0

    def __init__(self, t0, box3d, obj_type, framerate, ID_force=None, v=None, P=None):
        """Box state is: [x, y, z, h, w, l, vx, vy, vz] w/ yaw as attribute"""
        dt = 1.0 / framerate
        self.dt = dt
        self.framerate = framerate
        if ID_force is None:
            self.ID = BasicBoxTrack3D.ID_counter
            BasicBoxTrack3D.ID_counter += 1
        else:
            self.ID = ID_force
        self.obj_type = obj_type
        self.origin = box3d.origin
        self.where_is_t = box3d.where_is_t

        # -- initialize filter
        self.kf = KalmanFilter(dim_x=9, dim_z=6)
        if P is None:
            P = np.diag([5, 5, 5, 2, 2, 2, 10, 10, 10]) ** 2
        self.kf.P = P
        self.kf.F = np.eye(9)
        self.kf.F[:3, 6:9] = dt * np.eye(3)
        self.kf.H = np.zeros((6, 9))
        self.kf.H[:6, :6] = np.eye(6)
        self.kf.R = np.diag([1, 1, 1, 0.5, 0.5, 0.5]) ** 2
        self.kf.Q = (np.diag([2, 2, 2, 0.5, 0.5, 0.5, 3, 3, 3]) * dt) ** 2
        if v is None:
            v = np.array([0, 0, 0])
        self.kf.x = np.array(
            [
                box3d.t[0],
                box3d.t[1],
                box3d.t[2],
                box3d.h,
                box3d.w,
                box3d.l,
                v[0],
                v[1],
                v[2],
            ]
        )
        self.q = box3d.q

        # -- initialize parameters
        self.coast = 0
        self.active = True
        self.n_updates = 1
        self.age = 0
        self.t0 = t0
        self.t = t0

    @property
    def x(self):
        return self.kf.x

    @property
    def P(self):
        return self.kf.P

    @property
    def position(self):
        return self.kf.x[:3]

    @property
    def velocity(self):
        return self.kf.x[6:9]

    @property
    def box3d(self):
        as_l = [el for el in self.kf.x[3:6]]
        as_l.extend([el for el in self.kf.x[:3]])
        as_l.append(self.q)
        return Box3D(as_l, self.origin, where_is_t=self.where_is_t)

    @property
    def yaw(self):
        return self.box3d.yaw

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"BasicBoxTrack w/ {self.n_updates} updates: {self.as_object()}"

    def update(self, box3d):
        if box3d.origin != self.origin:
            box3d.change_origin(self.origin)
        if self.where_is_t != box3d.where_is_t:
            raise NotImplementedError(
                "Differing t locations not implemented: {}, {}".format(
                    self.where_is_t, box3d.where_is_t
                )
            )
        self.coast = 0
        self.n_updates += 1
        det = np.array([box3d.t[0], box3d.t[1], box3d.t[2], box3d.h, box3d.w, box3d.l])
        self.kf.update(det)
        self.q = box3d.q

    def predict(self):
        self.kf.predict()
        self.t += self.dt
        self.age += 1
        self.coast += 1

    def as_object(self):
        vs = VehicleState(obj_type=self.obj_type, ID=self.ID)
        vs.set(
            t=self.t,
            position=self.position,
            box=self.box3d,
            velocity=self.velocity,
            acceleration=None,
            attitude=self.q,
            angular_velocity=None,
            origin=self.origin,
        )
        return vs

    def format_as(self, format="avstack"):
        return self.as_object().format_as(format)


class BasicBoxTrack2D:
    ID_counter = 0

    def __init__(self, t0, box2d, obj_type, framerate):
        dt = 1.0 / framerate
        self.dt = dt
        self.t0 = t0
        self.t = t0
        self.ID = BasicBoxTrack2D.ID_counter
        BasicBoxTrack2D.ID_counter += 1
        self.obj_type = obj_type
        self.box2d = box2d
        self.coast = 0
        self.n_updates = 0

    @property
    def calibration(self):
        return self.box2d.calibration

    def update(self, box2d):
        """For now, the update is trivial"""
        self.coast = 0
        self.n_updates += 1
        self.box2d = box2d

    def predict(self):
        self.coast += 1
        self.t += self.dt


class BasicJointBoxTrack:
    def __init__(self, t0, box2d, box3d, obj_type, framerate):
        self.framerate = framerate
        self.track_2d = (
            BasicBoxTrack2D(t0, box2d, obj_type, framerate)
            if box2d is not None
            else None
        )
        self.track_3d = (
            BasicBoxTrack3D(t0, box3d, obj_type, framerate)
            if box3d is not None
            else None
        )

    @property
    def ID(self):
        return self.track_3d.ID

    @property
    def box(self):
        return self.box3d

    @property
    def box2d(self):
        return self.track_2d.box2d if self.track_2d is not None else None

    @property
    def box3d(self):
        return self.track_3d.box3d if self.track_3d is not None else None

    @property
    def yaw(self):
        return self.box3d.yaw

    @property
    def n_updates_2d(self):
        return self.track_2d.n_updates if self.track_2d is not None else 0

    @property
    def n_updates_3d(self):
        return self.track_3d.n_updates if self.track_3d is not None else 0

    @property
    def coast_2d(self):
        return self.track_2d.coast if self.track_2d is not None else 0

    @property
    def coast_3d(self):
        return self.track_3d.coast if self.track_3d is not None else 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"BasicJointBoxTrack w/ {self.track_2d.n_updates} 2D n_updates: {self.box2d}; {self.track_3d.n_updates} 3D n_updates: {self.box3d}"

    def update(self, box, obj_type):
        if isinstance(box, tuple):
            b2 = box[0]
            b3 = box[1]
        elif isinstance(box, Box2D):
            b2 = box
            b3 = None
        elif isinstance(box, Box3D):
            b2 = None
            b3 = box
        else:
            raise NotImplementedError(type(box))

        # -- update 2d
        if self.track_2d is None and b2 is not None:
            self.track_2d = BasicBoxTrack2D(
                self.track_3d.t, b2, obj_type, self.framerate
            )
        elif b2 is not None:
            self.track_2d.update(b2)

        # -- update 3d
        if self.track_3d is None and b3 is not None:
            self.track_3d = BasicBoxTrack3D(
                self.track_2d.t, b3, obj_type, self.framerate
            )
        elif b3 is not None:
            self.track_3d.update(b3)

    def predict(self):
        if self.track_2d is not None:
            self.track_2d.predict()
        if self.track_3d is not None:
            self.track_3d.predict()

    def as_object(self):
        if self.track_3d is not None:
            return self.track_3d.as_object()
        else:
            raise RuntimeError("No 3d track to convert to object")

    def format_as(self, format_):
        return self.as_object().format_as(format_)


# ==============================================================
# EXTERNALS
# ==============================================================


class Ab3dmotTracker(_TrackingAlgorithm):
    def __init__(self, framerate, **kwargs):
        self.iframe = 0
        self.framerate = framerate
        self.tracker = libraries.AB3DMOT.model.AB3DMOT(framerate)
        self.n_tracks_total = 0
        self.n_tracks_last = 0
        self._ID_set = set()
        self.origin = None
        super().__init__(**kwargs)

    @property
    def n_tracks(self):
        return self.n_tracks_total

    @property
    def n_tracks_active(self):
        return self.n_tracks_last

    @property
    def tracks(self):
        return self.tracker.trackers

    def track(self, detections_3d, *args, **kwargs):
        """
        :detections - list of class Detection

        AB3DMOT expects detections in camera coordinates, so do conversion
        """
        if (len(detections_3d) > 0) and (self.origin is None):
            self.origin = detections_3d[0].box.origin
            self.z_up = np.all(
                np.round(self.origin.rotation.up_vector) == np.array([0, 0, 1])
            )
        ts = self.iframe * 1.0 / self.framerate

        # -- get information on the object
        ori_array = np.asarray([d.box.yaw for d in detections_3d]).reshape((-1, 1))
        score = 1
        other_array = np.asarray(
            [[d.obj_type, None, None, None, None, None] for d in detections_3d]
        )
        if len(other_array.shape) == 1:
            other_array = other_array[:, None]
        additional_info = np.concatenate((ori_array, other_array), axis=1)

        # --- make detections format
        dets = np.asarray(
            [
                [
                    d.box.h,
                    d.box.w,
                    d.box.l,
                    d.box.t[0],
                    d.box.t[1],
                    d.box.t[2],
                    d.box.yaw,
                ]
                for d in detections_3d
            ]
        )
        if len(dets.shape) == 1:
            dets = dets[:, None]
        dets_all = {"dets": dets, "info": additional_info}

        # --- update
        tracks = self.tracker.update(dets_all, self.z_up)
        return self._format_tracks(tracks, ts)

    def _format_tracks(self, tracks, ts):
        # --- put in common format
        tracks_format = []
        for d in tracks:
            ID = d[7]
            pos = d[3:6]
            vel = d[15:18]
            acc = np.nan * np.ones((3,))
            h, w, l, yaw = d[0], d[1], d[2], d[6]
            q = tforms.transform_orientation([0, 0, yaw], "euler", "quat")
            vs = VehicleState(obj_type=d[9], ID=ID)
            vs.set(
                t=ts,
                position=pos,
                box=Box3D([h, w, l, pos, q], self.origin),
                velocity=vel,
                acceleration=acc,
                attitude=q,
                angular_velocity=None,
                origin=self.origin,
            )
            tracks_format.append(vs)
            # n_updates=d[18],
            # score=(d[14] if d[14] is not None else 1.0))
        self._ID_set = self._ID_set.union(set([t.ID for t in tracks_format]))
        self.n_tracks_total = len(self._ID_set)
        self.n_tracks_last = len(tracks_format)
        return tracks_format


class EagermotTracker(_TrackingAlgorithm):
    def __init__(
        self, framerate, plus=False, n_box_confirmed=0, n_joint_coast=np.inf, **kwargs
    ):
        self.tracker = libraries.EagerMOT.model.EagerMOT(
            framerate, plus, n_box_confirmed, n_joint_coast
        )
        super().__init__(**kwargs)

    def track(self, detections_2d, detections_3d, *args, **kwargs):
        tracks = self.tracker(detections_2d, detections_3d)
        return self._format_tracks(tracks, detections_2d, detections_3d)

    @property
    def n_tracks(self):
        return len(self.tracker.tracks)

    @property
    def n_tracks_confirmed(self):
        return len(self.tracker.tracks_confirmed)

    def _format_tracks(self, tracks, detections_2d, detections_3d):
        tracks_format = []
        if isinstance(detections_2d, dict):
            frame = detections_2d[list(detections_2d.keys())[0]].frame
            timestamp = detections_2d[list(detections_2d.keys())[0]].timestamp
        else:
            frame = detections_2d.frame
            timestamp = detections_2d.frame
        for ID, trk in tracks.items():
            x, y, z, yaw, h, w, l, vx, vy, vz = trk.kf.x[:, 0]
            origin = trk.box3d.origin
            vs = VehicleState(obj_type=None, ID=ID)
            pos = np.array([x, y, z])
            q = tforms.transform_orientation([0, 0, yaw], "euler", "quat")
            vs.set(
                t=timestamp,
                position=pos,
                box=Box3D([h, w, l, pos, q], origin),
                velocity=np.array([vx, vy, vz]),
                acceleration=None,
                attitude=q,
                angular_velocity=None,
                origin=origin,
            )
            tracks_format.append(vs)
        return tracks_format


class Chaser3DBoxTracker(_TrackingAlgorithm):
    def __init__(self, **kwargs):
        self.tracker = chaser.get_tracker(
            association="gnn", dim=3, model="cv", maneuver="low", use_box=True
        )
        super().__init__(**kwargs)

    def track(self, detections, *args, **kwargs):
        detections_3d = detections["object_3d"]
        msmts = self._convert_dets_to_msmts(detections_3d)
        self.tracker.process_msmts(msmts)
        return self._format_tracks(self.tracker.confirmed_tracks, detections_3d)

    @property
    def tracks(self):
        return {trk.ID: trk for trk in self.tracker.confirmed_tracks}

    def _convert_dets_to_msmts(self, dets):
        msmts = []
        for d in dets:
            if isinstance(d, BoxDetection):
                if isinstance(d.box, Box3D):
                    r = np.array([1, 1, 1, 0.5, 0.5, 0.5, 0.1])
                    m = estimators.measurements.BoxMeasurement_3D_XYZHWLYaw(
                        source_ID=d.source_ID,
                        t=dets.timestamp,
                        r=r,
                        x=d.box.t[0],
                        y=d.box.t[1],
                        z=d.box.t[2],
                        h=d.box.h,
                        w=d.box.w,
                        l=d.box.l,
                        yaw=d.box.yaw,
                    )
                    msmts.append(m)
                elif isinstance(d.box, Box2D):
                    raise NotImplementedError
                    r = np.array([5, 5, 5, 5])
                    m = estimators.measurements.BoxMeasurement_2D_XYXY(
                        source_ID=d.source_ID,
                        t=dets.timestamp,
                        r=r,
                        x_min=d.box.xmin,
                        y_min=d.box.ymin,
                        x_max=d.box.xmax,
                        y_max=d.box.ymax,
                    )
                    msmts.append(m)
                else:
                    raise NotImplementedError(type(d))
            else:
                raise NotImplementedError(type(d))
        return msmts

    def _format_tracks(self, tracks, detections_3d):
        tracks_format = []
        if isinstance(detections_3d, dict):
            frame = detections_3d[list(detections_3d.keys())[0]].frame
            timestamp = detections_3d[list(detections_3d.keys())[0]].timestamp
        else:
            frame = detections_3d.frame
            timestamp = detections_3d.frame
        for trk in tracks:
            x, y, z, vx, vy, vz, h, w, l, yaw = trk.filter.x_vector
            q = None
            raise
            vs = VehicleState(obj_type=None, ID=ID)
            vs.set(
                t=timestamp,
                position=np.array([x, y, z]),
                box=Box3D([h, w, l, np.zeros((3,)), q], StandardCoordinates),
                velocity=np.array([vx, vy, vz]),
                acceleration=None,
                angular_velocity=None,
                origin=NominalOriginStandard,
            )
            tracks_format.append(vs)
        return tracks_format
