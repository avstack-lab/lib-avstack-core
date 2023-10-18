# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-27
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-24
# @Description:
"""

"""
from typing import Any
import numpy as np

from avstack.datastructs import DataContainer
from avstack.geometry import Position, bbox
from avstack.modules.assignment import build_A_from_iou, gnn_single_frame_assign
from avstack.modules.tracking.tracker3d import BasicBoxTrack3D
from avstack.modules.tracking.tracks import XyFromRazTrack, _TrackBase

from .base import _FusionAlgorithm


def ci_fusion(x1, P1, x2, P2, w=0.5):
    """Covariance intersection fusion between filter states
    useful if the cross-correlation between the two data elements is not known

    Cross-correlation can be due to various factors such as common platform
    motion, common occlusion scenarios, etc.

    NOTE: only allows for full state fusion right now
    """
    assert len(x1) == len(x2) == P1.shape[0] == P2.shape[0]
    P1_inv = np.linalg.inv(P1)
    P2_inv = np.linalg.inv(P2)
    P_f = np.linalg.inv(w * P1_inv + (1 - w) * P2_inv)
    x_f = P_f @ (w * P1_inv @ x1 + (1 - w) * P2_inv @ x2)
    return x_f, P_f


class NoFusion:
    """Only returns the first set of tracks"""
    def __call__(self, *args: Any, **kwds: Any) -> list:
        tracks_out = [] if len(args) == 0 else args[0]
        if isinstance(tracks_out, (DataContainer, list)):
            pass
        elif isinstance(tracks_out, dict):
            tracks_out = list(tracks_out.values())[0] if len(tracks_out) == 1 else  list(tracks_out.values())
        elif isinstance(tracks_out, _TrackBase):
            tracks_out = [tracks_out]
        else:
            raise NotImplementedError(type(tracks_out))
        return tracks_out


class AggregatorFusion:
    """Simply appends all tracks together not worrying about duplicates"""

    def __call__(self, *args: Any, **kwds: Any) -> list:
        tracks_out = []
        for arg in args:
            if isinstance(arg, list):
                tracks_out += arg
            elif isinstance(arg, dict):
                for v in arg.values():
                    tracks_out += v
            elif isinstance(arg, _TrackBase):
                tracks_out += [arg]
            else:
                raise NotImplementedError(type(arg))
        return tracks_out


class CovarianceIntersectionFusion:
    """Runs assignment algorithm to determine if there are duplicates

    For duplicates, run covariance intersection for fusion
    """

    def __init__(self, clustering):
        self.clustering = clustering

    def __call__(self, *tracks):
        # -- run clustering
        if len(tracks) == 0:
            tracks_out = []
        else:
            objects = []
            for trks in tracks:
                if isinstance(trks, list):
                    objects.extend([trk.data for trk in trks])
                elif isinstance(trks, dict):
                    objects.extend([trk.data for trk in trks.values()])
                elif isinstance(trks, _TrackBase):
                    objects.append(trks.data)
                else:
                    raise NotImplementedError(type(trks))
            clusters = self.clustering(objects)

            # -- run fusion
            tracks_out = []
            for i, cluster in enumerate(clusters):
                if len(cluster) > 0:
                    # perform fusion on the array
                    x_fuse, P_fuse = cluster[0].x, cluster[0].P
                    for track in cluster[1:]:
                        x_fuse, P_fuse = ci_fusion(x_fuse, P_fuse, track.x, track.P, w=0.5)
                    # rebuild the track
                    track = XyFromRazTrack(
                        t0=cluster[0].t0,
                        raz=None,
                        reference=cluster[0].reference,
                        obj_type=cluster[0].obj_type,
                        ID_force=i,
                        x=x_fuse,
                        P=P_fuse,
                        t=cluster[0].t,
                        coast=-1,
                        n_updates=-1,
                        age=-1,
                    )
                    tracks_out.append(track)

        return tracks_out


class BoxTrackToBoxTrackFusion3D(_FusionAlgorithm):
    def __init__(self, association="IoU", assignment="gnn", algorithm="CI", **kwargs):
        """

        NOTE: assumes state vector is [x, y, z, h, w, l, vx, vy, vz]

        Track-to-track fusion can suffer from the track-ID ambiguity problem.
        We attempt to solve that by keeping a double dictionary of track keys.
        The dictionary is as follows: {'track1_ID':{'track2_ID': 'fuse_ID'}}.
        Thus any common combination of track1_ID and track2_ID always yields
        the same fused track ID.
        """
        super().__init__(**kwargs)
        self.association = association
        self.assignment = assignment
        self.algorithm = algorithm

        self.ID_registry = {}

    def fuse(
        self, tracks3d_1, tracks3d_2, keep_lones=False, IoU_thresh=0.2, *args, **kwargs
    ):
        """
        Get association metrics...assign...fuse

        NOTE: tracks should be put into the same coordinate frame before this!!
        """
        if (len(tracks3d_1) == 0) or (len(tracks3d_2) == 0):
            return []

        # -- step 1: association metrics
        if self.association == "IoU":
            # NOTE: this step is ok to have difference origins
            A = build_A_from_iou(
                [trk1.box3d for trk1 in tracks3d_1], [trk2.box3d for trk2 in tracks3d_2]
            )
        else:
            raise NotImplementedError(self.association)
        # import ipdb; ipdb.set_trace()

        # -- step 2: assignment solution
        if self.assignment == "gnn":
            assign_sol = gnn_single_frame_assign(A, cost_threshold=-IoU_thresh)
        else:
            raise NotImplementedError(self.assignment)

        # -- step 3: fusion
        tracks3d_fused = []
        if self.algorithm == "CI":
            for row, col in assign_sol.assignment_tuples:
                t1 = tracks3d_1[row]
                t2 = tracks3d_2[col]
                if not t1.reference == t2.reference:
                    t2.change_reference(t1.reference, inplace=True)
                x_f, P_f = ci_fusion(t1.x, t1.P, t2.x, t2.P)
                x, y, z, h, w, l, vx, vy, vz = x_f
                v_f = [vx, vy, vz]
                t = t2.t0
                obj_type = t2.obj_type
                reference = t2.reference
                pos = Position(np.array([x, y, z]), reference)
                rot = t2.q
                box_f = bbox.Box3D(pos, rot, [h, w, l], where_is_t=t2.box3d.where_is_t)
                ID = None
                if t1.ID in self.ID_registry:
                    ID = self.ID_registry[t1.ID].get(t2.ID, None)
                else:
                    self.ID_registry[t1.ID] = {}
                fused = BasicBoxTrack3D(
                    t, box_f, box_f.reference, obj_type, ID_force=ID, v=v_f, P=P_f
                )
                self.ID_registry[t1.ID][t2.ID] = fused.ID
                tracks3d_fused.append(fused)

            # keep loners
            if keep_lones:
                for idx_row in assign_sol.unassigned_rows:
                    tracks3d_fused.append(tracks3d_1[idx_row])
                for idx_col in assign_sol.unassigned_cols:
                    tracks3d_fused.append(tracks3d_2[idx_col])
        else:
            raise NotImplementedError(self.algorithm)

        return tracks3d_fused
