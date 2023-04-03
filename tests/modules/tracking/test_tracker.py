# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-04
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-23
# @Description:
"""

"""

import sys
from copy import deepcopy

import numpy as np

import avstack
from avstack import GroundTruthInformation
from avstack.datastructs import DataContainer
from avstack.geometry import bbox
from avstack.geometry.transformations import cartesian_to_spherical
from avstack.modules import tracking
from avstack.modules.perception.detections import BoxDetection, CentroidDetection, RazelRrtDetection


sys.path.append("tests/")
from utilities import get_ego, get_object_local, get_test_sensor_data


(
    obj,
    box_calib,
    lidar_calib,
    pc,
    camera_calib,
    img,
    box_2d,
    box_3d,
) = get_test_sensor_data()
T_lidar = lidar_calib.origin
T_camera = camera_calib.origin

name_3d = "lidar"
name_2d = "image"


def test_groundtruth_tracking():
    ego = get_ego(1)
    obj1 = get_object_local(ego, 10)
    frame = timestamp = 0
    ground_truth = GroundTruthInformation(
        frame, timestamp, ego_state=ego, objects=[obj1]
    )
    tracker = tracking.tracker3d.GroundTruthTracker()
    tracks = tracker(ground_truth, frame=frame, identifier="tracker-1")
    assert np.all(tracks[0].position == obj1.position)
    assert np.all(tracks[0].velocity == obj1.velocity)


def make_kitti_tracking_data(dt=0.1, n_frames=10, n_targs=4, det_type='box'):
    ego = get_ego(1)
    objects = [get_object_local(ego, i + 10) for i in range(n_targs)]
    t = 0
    dets_3d_all = []
    for i in range(n_frames):
        dets = deepcopy(objects)
        dets_class = []
        for det in dets:
            det.box3d.t[2] += (
                t * det.velocity.vector[2]
            )  # camera coordinates with z forward
            det.box3d.change_origin(lidar_calib.origin)
            if det_type == 'box':
                det = BoxDetection(name_3d, det.box3d, det.obj_type)
            elif det_type == 'centroid':
                det = CentroidDetection(name_3d, det.box3d.t.vector, det.obj_type)
            elif det_type == 'razelrrt':
                (rng, az, el), rrt = cartesian_to_spherical(det.box3d.t), det.velocity.vector[0]
                razelrrt = np.array([rng, az, el, rrt])
                det = RazelRrtDetection(name_3d, razelrrt, det.obj_type)
            else:
                raise NotImplementedError
            dets_class.append(det)
        detections = DataContainer(i, t, dets_class, source_identifier=name_3d)
        dets_3d_all.append(detections)
        t += dt
    return dets_3d_all


def make_kitti_2d_3d_tracking_data(dt=0.1, n_frames=10, n_targs=4):
    dets_3d_all = make_kitti_tracking_data(dt, n_frames, n_targs=n_targs)
    dets_2d_all = []
    for i, dets_3d in enumerate(dets_3d_all):
        d_2d = [
            BoxDetection(
                "detector-2d", d.box.project_to_2d_bbox(camera_calib), d.obj_type
            )
            for d in dets_3d
        ]
        dets_2d_all.append(
            DataContainer(i, dets_3d.timestamp, d_2d, source_identifier=name_2d)
        )
        assert isinstance(dets_3d, DataContainer)
    return dets_2d_all, dets_3d_all


def test_make_3d_tracking_data():
    dets_3d_all = make_kitti_tracking_data(n_frames=10)
    dets_3d = dets_3d_all[0]
    assert isinstance(dets_3d, DataContainer)
    assert isinstance(dets_3d[0], avstack.modules.perception.detections.BoxDetection)
    assert isinstance(dets_3d[0].data, avstack.geometry.Box3D)


def test_make_2d3d_tracking_data():
    dets_2d_all, dets_3d_all = make_kitti_2d_3d_tracking_data(n_frames=10)
    dets_2d = dets_2d_all[0]
    dets_3d = dets_3d_all[0]
    assert isinstance(dets_2d, DataContainer)
    assert isinstance(dets_2d[0], BoxDetection)
    assert isinstance(dets_3d, DataContainer)
    assert isinstance(dets_3d[0], BoxDetection)
    assert len(dets_2d) == len(dets_3d)


def test_razelrrt_tracker_3d():
    dets_3d_all = make_kitti_tracking_data(dt=0.1, n_frames=10, n_targs=4, det_type='razelrrt')
    tracker = tracking.tracker3d.BasicRazelRrtTracker()
    for frame, dets_3d in enumerate(dets_3d_all):
        tracks = tracker(
            t=frame * 0.10, detections_nd=dets_3d, frame=frame, identifier="tracker-1"
        )
    assert len(tracks) == len(dets_3d_all[-1])
    for i, trk in enumerate(tracks):
        for det in dets_3d_all[-1]:
            if np.linalg.norm(trk.position - det.xyz) < 2:
                break
        else:
            raise


def test_basic_box_tracker_3d():
    n_frames = 10
    dets_3d_all = make_kitti_tracking_data(n_frames=n_frames)
    tracker = tracking.tracker3d.BasicBoxTracker3D()
    for frame, dets_3d in enumerate(dets_3d_all):
        tracks = tracker(
            t=frame * 0.10, detections_nd=dets_3d, frame=frame, identifier="tracker-1"
        )
    assert len(tracks) == len(dets_3d_all[-1])
    for i, trk in enumerate(tracks):
        for det in dets_3d_all[-1]:
            if np.linalg.norm(trk.box3d.t - det.box.t) < 2:
                break
        else:
            raise


def test_basic_box_tracker_2d():
    n_targs = 4
    dets_2d_all, dets_3d_all = make_kitti_2d_3d_tracking_data(
        n_frames=10, n_targs=n_targs
    )
    tracker = tracking.tracker2d.BasicBoxTracker2D()
    for frame, dets_2d in enumerate(dets_2d_all):
        tracks = tracker(
            t=frame * 0.10, detections_nd=dets_2d, frame=frame, identifier="tracker-1"
        )
    assert len(tracks) == len(dets_3d_all[-1])
    for i, trk in enumerate(tracks):
        for det in dets_2d_all[-1]:
            if np.linalg.norm(trk.box2d.center - det.box.center) < 10:
                break
        else:
            raise


def test_basic_joint_box_tracker():
    n_targs = 4
    dets_2d_all, dets_3d_all = make_kitti_2d_3d_tracking_data(
        n_frames=10, n_targs=n_targs
    )
    tracker = tracking.tracker3d.BasicBoxTrackerFusion3Stage()
    for frame, (dets_2d, dets_3d) in enumerate(zip(dets_2d_all, dets_3d_all)):
        tracks = tracker(
            t=frame * 0.10,
            detections_2d=dets_2d,
            detections_3d=dets_3d,
        frame=frame,
            identifier="tracker-1",
        )
    assert len(tracks) == n_targs


def test_ab3dmot_kitti():
    n_frames = 10
    dets_3d_all = make_kitti_tracking_data(n_frames=n_frames)
    tracker = tracking.tracker3d.Ab3dmotTracker()
    for frame, dets_3d in enumerate(dets_3d_all):
        tracks = tracker(
            t=frame * 0.10, detections_3d=dets_3d, frame=frame, identifier="tracker-1"
        )


def test_eagermot_fusion_kitti():
    dets_2d_all, dets_3d_all = make_kitti_2d_3d_tracking_data(n_frames=10)
    tracker_base = (
        avstack.modules.tracking.tracker3d.libraries.EagerMOT.model.EagerMOT()
    )
    for dets_2d, dets_3d in zip(dets_2d_all, dets_3d_all):
        # add false positives
        b2d = BoxDetection(name_2d, bbox.Box2D([10, 20, 20, 30], camera_calib))
        b3d = BoxDetection(
            name_3d,
            bbox.Box3D([1, 1, 1, [10, -20, 2], np.quaternion(1)], lidar_calib.origin),
        )
        dets_2d.append(b2d)
        dets_3d.append(b3d)
        lone_2d, lone_3d, fused_instances = tracker_base.fusion(dets_2d, dets_3d)
        assert len(lone_2d) == 1
        assert len(lone_3d) == 1
        assert len(fused_instances) == len(dets_3d) - 1


def test_eagermot_associations():
    dets_2d_all, dets_3d_all = make_kitti_2d_3d_tracking_data(n_frames=10)
    tracker = tracking.tracker3d.EagermotTracker()
    trk_base = tracker.tracker
    i = 0
    for frame, (dets_2d, dets_3d) in enumerate(zip(dets_2d_all, dets_3d_all)):
        # add false positives
        if i == 0:
            b2d = BoxDetection(name_3d, bbox.Box2D([10, 20, 20, 30], camera_calib))
            b3d = BoxDetection(
                name_2d,
                bbox.Box3D(
                    [1, 1, 1, [19, -20, 2], np.quaternion(1)], lidar_calib.origin
                ),
            )
            dets_2d.append(b2d)
            dets_3d.append(b3d)

        # test association
        lone_2d, lone_3d, fused_instances = trk_base.fusion(dets_2d, dets_3d)
        (
            assign_1,
            lone_fused_1,
            lone_3d_1,
            lone_track_1,
        ) = trk_base.tracker._first_data_association(lone_3d, fused_instances)
        (
            assign_2,
            lone_fused_2,
            lone_2d_2,
            lone_track_2,
        ) = trk_base.tracker._second_data_association(
            lone_2d, lone_fused_1, lone_track_1
        )
        # raise
        if i == 0:
            assert (
                len(lone_track_1)
                == len(lone_track_2)
                == len(assign_1)
                == len(assign_2)
                == 0
            )
        else:
            assert len(assign_1) == len(dets_2d) == len(dets_3d)

        # run for real
        tracks = tracker(
            t=frame * 0.10,
            detections_2d=dets_2d,
            detections_3d=dets_3d,
            frame=frame,
            identifier="tracker-1",
        )
        if i == 1:
            assert (
                len(tracks) == tracker.n_tracks_confirmed == 0
            )  # due to confirmation time
        elif i > 3:
            assert tracker.n_tracks > 0
            assert tracker.n_tracks_confirmed == len(dets_3d)
        i += 1


def test_eagermot_performance():
    n_targs = 4
    dets_2d_all, dets_3d_all = make_kitti_2d_3d_tracking_data(
        n_frames=10, n_targs=n_targs
    )
    tracker = tracking.tracker3d.EagermotTracker()
    tracks = tracker.tracker
    for frame, (dets_2d, dets_3d) in enumerate(zip(dets_2d_all, dets_3d_all)):
        tracks = tracker(
            t=frame * 0.10,
            detections_2d=dets_2d,
            detections_3d=dets_3d,
            frame=frame,
            tracker="tracker-1",
        )
    assert len(tracks) == n_targs
    for i, trk in enumerate(tracks):
        for det in dets_3d:
            if np.linalg.norm(trk.position.vector - det.box.t) < 2:
                break
        else:
            raise


def test_radar_tracker():
    pass