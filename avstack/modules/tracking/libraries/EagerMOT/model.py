# -*- coding: utf-8 -*-
# @Author: spencer@primus
# @Date:   2022-05-31
# @Last Modified by:   Spencer H
# @Last Modified time: 2022-06-29

import numpy as np
import avstack

from avstack.datastructs import OneEdgeBipartiteGraph, DataContainer
from avstack.geometry import Box2D, Box3D
from avstack.modules.perception.detections import JointBoxDetection, JointBoxDetectionAndOther, BoxDetection
from avstack.modules import assignment
from avstack.objects import IOU_2d, IOU_3d

from .kalman import EagerMOTTrack


class EagerMOT():
    def __init__(self, framerate, plus=False, n_box_confirmed=0, n_joint_coast=np.inf):
        self.n_frames = 0
        self.fusion = EagerMOTFusion()
        self.tracker = EagerMOTTracking(framerate, plus=plus,
            n_box_confirmed=n_box_confirmed, n_joint_coast=n_joint_coast)

    @property
    def tracks(self):
        return self.tracker.tracks

    @property
    def tracks_confirmed(self):
        return self.tracker.tracks_confirmed

    def __call__(self, detections_2d, detections_3d):
        """
        detections_2d: dictionary for {sensor_ID: [detections]}
        detections_3d: dictionary for {sensor_ID: [detections]}
        """
        self.n_frames += 1

        # -- fuse objects
        lone_2d, lone_3d, fused_detections = self.fusion(detections_2d, detections_3d)

        # -- track
        tracks = self.tracker(lone_2d, lone_3d, fused_detections)

        return tracks


class EagerMOTTracking():
    def __init__(self, framerate, threshold_1=3, threshold_2=0.33,
                 max_age=5, min_hits=3, plus=False, n_box_confirmed=0, n_joint_coast=np.inf):
        """
        threshold 2 is the greedy assignment IoU threshold
        """
        self.n_frames = 0
        self.tracks = []
        self.framerate = framerate
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.max_age = max_age
        self.min_hits = min_hits
        self.plus = plus
        if self.plus and n_box_confirmed <= 0:
            n_box_confirmed = 1
        self.n_box_confirmed = n_box_confirmed
        self.n_joint_coast = n_joint_coast

    def __call__(self, lone_2d, lone_3d, fused_detections):
        """Main call for EagerMOT tracking"""
        # import ipdb; ipdb.set_trace()
        self.n_frames += 1
        for trk in self.tracks:
            trk.predict()
        assign_1, lone_fused_1, lone_3d_1, lone_track_1 = self._first_data_association(lone_3d, fused_detections)
        assign_2, lone_fused_2, lone_2d_2, lone_track_2 = self._second_data_association(lone_2d, lone_fused_1, lone_track_1)
        self._update(assign_1, assign_2)
        self._maintenance(lone_3d_1, lone_2d_2, lone_fused_2)
        return self.tracks_confirmed

    @property
    def _tracks_confirmed_all(self):
        return {t.id:t for t in self.tracks if t.hits >= self.min_hits}

    @property
    def tracks_confirmed(self):
        if self.plus:
            return {t_id:t for t_id, t in self._tracks_confirmed_all.items()
                    if t.box3d_initialized and t.box2d_confirmed and
                    (t.box2d_n_confirmed >= self.n_box_confirmed) and
                    (t.box3d_n_confirmed >= self.n_box_confirmed) and
                    (t.box3d_coast <= self.n_joint_coast) and
                    (t.box2d_coast <= self.n_joint_coast)}
        else:
            return {t_id:t for t_id, t in self._tracks_confirmed_all.items() if t.box3d_initialized}

    def _maintenance(self, lone_3d_1, lone_2d_2, lone_fused_1):
        """Initialize tracks based on 2d detection"""
        # -- delete old track
        # import ipdb; ipdb.set_trace()

        i = len(self.tracks)
        for trk in reversed(self.tracks):
            i -= 1
            if trk.time_since_update >= self.max_age:
                self.tracks.pop(i)

        # -- initialize new tracks
        for d_3d in lone_3d_1:
            self.tracks.append(EagerMOTTrack(None, d_3d.box, self.framerate))
        for d_2d in lone_2d_2:
            self.tracks.append(EagerMOTTrack(d_2d.box, None, self.framerate))
        for d_2d, d_3d in lone_fused_1:
            self.tracks.append(EagerMOTTrack(d_2d, d_3d, self.framerate))

    def _update(self, assign_1, assign_2):
        """Update tracks with assignments"""

        # -- coast each track always
        for track in self.tracks:
            track.box2d_coast += 1
            track.box3d_coast += 1

        # -- update 2d assignments first
        for det_2d, track in assign_2:
            if isinstance(det_2d, JointBoxDetection):
                track.update(det_2d.box2d, det_2d.box3d)
            elif isinstance(det_2d, BoxDetection):
                track.update(det_2d.data, None)
            elif isinstance(det_2d, Box2D):
                track.update(det_2d, None)
            elif isinstance(det_2d, tuple) and len(det_2d) == 2:
                track.update(det_2d[0], det_2d[1])
            else:
                raise NotImplementedError(f'Update using {type(det_2d)} not implemented -- {det_2d}')

        # -- update kalman filter of 3d track
        for det_3d, track in assign_1:
            if isinstance(det_3d, tuple):
                track.update(det_3d[0], det_3d[1])
            elif isinstance(det_3d, BoxDetection):
                track.update(None, det_3d.box)
            elif isinstance(det_3d, Box3D):
                track.update(None, det_3d)
            else:
                raise NotImplementedError

    def _first_data_association(self, lone_3d, fused_detections):
        """First association of fused + 3d detections to existing tracks"""
        # Get assignment scores
        i = 0
        A = np.inf * np.ones((len(fused_detections)+len(lone_3d), len(self.tracks)))
        for ds in (fused_detections, lone_3d):
            for d_ in ds:
                for j, t in enumerate(self.tracks):
                    if t.box3d is None:
                        continue
                    if i < len(fused_detections):
                        d = d_.box3d
                    else:
                        d = d_.data
                    Bi = np.array([t.box3d.t[0], t.box3d.t[1], t.box3d.t[2], t.box3d.h, t.box3d.w, t.box3d.l])
                    Bj = np.array([d.t[0], d.t[1], d.t[2], d.h, d.w, d.l])
                    alpha = 2 - np.cos((t.yaw - d.yaw) % (np.pi/2))
                    A[i,j] = np.linalg.norm(Bi-Bj) * alpha
                i += 1

        # Perform greedy assignment
        rows = fused_detections + lone_3d
        cols = self.tracks
        lone_3d, lone_tracks, assigns = assign_and_create_results(A, rows, cols, 1, 1, self.threshold_1)
        lone_fused = [l for l in lone_3d if isinstance(l, (JointBoxDetection,))]
        lone_3d = [l for l in lone_3d if not isinstance(l, (JointBoxDetection,))]
        return assigns, lone_fused, lone_3d, lone_tracks

    def _second_data_association(self, lone_2d, lone_fused_1, lone_tracks):
        """Second association of 2d detections to remaining tracks"""
        A = np.zeros((len(lone_2d)+len(lone_fused_1), len(lone_tracks)))
        # Get assignment scores
        i = 0
        for ds in (lone_2d, lone_fused_1):
            for d in ds:
                for j, t in enumerate(lone_tracks):
                    try:
                        t_box = t.box2d if t.box2d is not None else t.box3d.project_to_2d_bbox(d.data.calibration)
                        iou = IOU_2d(d.data.box2d, t_box.box2d)
                    except AttributeError as e:
                        t_box = t.box2d if t.box2d is not None else t.box3d.project_to_2d_bbox(d.box2d.calibration)
                        iou = IOU_2d(d.box2d.box2d, t_box.box2d)
                    A[i,j] = iou
                i += 1

        # Perform greedy assignment
        rows = lone_2d + lone_fused_1
        cols = lone_tracks
        lone_2d, lone_tracks, assigns = assign_and_create_results(-A, rows, cols, 1, 1, -self.threshold_2)
        lone_fused = [l for l in lone_2d if isinstance(l, (JointBoxDetection,))]
        lone_2d = [l for l in lone_2d if not isinstance(l, (JointBoxDetection,))]
        return assigns, lone_fused, lone_2d, lone_tracks


class EagerMOTFusion():
    def __init__(self, threshold=0.33):
        """
        threshold is the IoU threshold for greedy assignment
        """
        self.threshold = threshold

    def __call__(self, detections_2d, detections_3d):
        """
        detections_2d: dictionary for {sensor_ID: [detections]}
        detections_3d: dictionary for {sensor_ID: [detections]}
        """
        assert isinstance(detections_2d, (DataContainer,dict)), f'{type(detections_2d)}'
        assert isinstance(detections_3d, (DataContainer,dict)), f'{type(detections_3d)}'
        if isinstance(detections_2d, dict):
            assert len(detections_2d) == 1, f'{detections_2d} For now can only do 1 sensor'
            ID_2d = list(detections_2d.keys())[0]
            detections_2d = detections_2d[ID_2d]
        else:
            ID_2d = 'image'
        if isinstance(detections_3d, dict):
            assert len(detections_3d) == 1, f'{detections_3d} For now can only do 1 sensor'
            ID_3d = list(detections_3d.keys())[0]
            detections_3d = detections_3d[ID_3d]
        else:
            ID_3d = 'lidar'

        # Get the 2D IoU's
        n_2d = len(detections_2d)
        n_3d = len(detections_3d)
        frame = detections_3d.frame
        timestamp = detections_3d.timestamp
        IoU_pairs = []
        A = np.zeros((n_2d, n_3d))
        last_calib = None
        for j, det_3d in enumerate(detections_3d):
            for i, det_2d in enumerate(detections_2d):
                if det_2d.data.calibration != last_calib:
                    det_3d_in_2d = det_3d.data.project_to_2d_bbox(det_2d.data.calibration)
                else:
                    assert last_calib is not None
                A[i,j] = IOU_2d(det_2d.data.box2d, det_3d_in_2d.box2d)

        # Do assignment
        rows = detections_2d
        cols = detections_3d
        lone_2d, lone_3d, fused_detections = assign_and_create_results(-A, rows, cols, ID_2d, ID_3d, -self.threshold)

        return lone_2d, lone_3d, fused_detections


def assign_and_create_results(A, rows, cols, sensor_rows, sensor_cols, threshold):
        # Perform greedy association
        assign = assignment.greedy_assignment(A, threshold=threshold)
        assign_map = assign.iterate_over('rows')
        assigns_map_rev = assign.iterate_over('cols')

        # Create fused results based on assignments
        joint_instances = []
        for i, jc in assign.iterate_over('rows').items():
            j = list(jc.keys())[0]
            obj_type = rows[i].obj_type
            try:
                joint_i = JointBoxDetection([sensor_rows, sensor_cols], rows[i].data, cols[j].data, obj_type)
            except AttributeError as e:
                joint_i = JointBoxDetectionAndOther([sensor_rows, sensor_cols], rows[i].data, cols[j], obj_type)
            joint_instances.append(joint_i)

        # Package
        lone_rows = [d for i, d in enumerate(rows) if i not in assign_map]
        lone_cols = [d for i, d in enumerate(cols) if i not in assigns_map_rev]

        return lone_rows, lone_cols, joint_instances
