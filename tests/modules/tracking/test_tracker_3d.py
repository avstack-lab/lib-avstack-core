# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-04
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-23
# @Description:
"""

"""

import sys

import numpy as np

import avstack
from avstack import GroundTruthInformation
from avstack.datastructs import DataContainer
from avstack.geometry import Attitude, GlobalOrigin3D, Position, ReferenceFrame, bbox
from avstack.geometry.transformations import xyzvel_to_razelrrt
from avstack.modules.perception.detections import BoxDetection, RazelRrtDetection
from avstack.modules.tracking import tracker3d


sys.path.append("tests/")
sys.path.append("tests/modules/tracking")
from track_utils import (
    make_kitti_2d_3d_tracking_data,
    make_kitti_tracking_data,
    run_tracker,
)
from utilities import get_ego, get_object_local, get_test_sensor_data


(
    obj,
    box_calib,
    lidar_calib,
    pc,
    camera_calib,
    img,
    radar_calib,
    rad,
    box_2d,
    box_3d,
) = get_test_sensor_data()
T_lidar = lidar_calib.reference
T_camera = camera_calib.reference

name_3d = "lidar"
name_2d = "image"


def test_groundtruth_tracking():
    ego = get_ego(1)
    obj1 = get_object_local(ego, 10)
    frame = timestamp = 0
    ground_truth = GroundTruthInformation(
        frame, timestamp, ego_state=ego, objects=[obj1]
    )
    tracker = tracker3d.GroundTruthTracker(name="tracker-1")
    platform = ego.as_reference()
    tracks = tracker(
        t=timestamp,
        frame=frame,
        detections=None,
        platform=platform,
        ground_truth=ground_truth,
    )
    assert np.all(tracks[0].position == obj1.position)
    assert np.all(tracks[0].velocity == obj1.velocity)


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


def test_xyz_tracker_3d():
    tracker = tracker3d.BasicXyzTracker(threshold_coast=5)
    run_tracker(tracker=tracker, det_type="xyz")


def test_razel_tracker_3d():
    tracker = tracker3d.BasicRazelTracker(threshold_coast=5)
    run_tracker(tracker=tracker, det_type="razel")


def test_razelrrt_tracker_3d():
    tracker = tracker3d.BasicRazelRrtTracker(threshold_coast=5)
    run_tracker(tracker=tracker, det_type="razelrrt", dt=0.10)


def test_basic_box_tracker_3d():
    tracker = tracker3d.BasicBoxTracker3D()
    run_tracker(tracker=tracker, det_type="box", dt=0.10)


def test_ab3dmot_kitti():
    tracker = tracker3d.Ab3dmotTracker()
    run_tracker(tracker=tracker, det_type="box", dt=0.10)


def test_basic_joint_box_tracker():
    platform = GlobalOrigin3D
    n_targs = 4
    dt = 0.10
    dets_2d_all, dets_3d_all = make_kitti_2d_3d_tracking_data(
        dt=dt, n_frames=4, n_targs=n_targs
    )
    tracker = tracker3d.BasicBoxTrackerFusion3Stage()
    for frame, (dets_2d, dets_3d) in enumerate(zip(dets_2d_all, dets_3d_all)):
        tracks = tracker(
            t=frame * dt,
            frame=frame,
            detections={"2d": dets_2d, "3d": dets_3d},
            platform=platform,
        )
    assert len(tracks) == n_targs


def test_inline_razelrrt_tracker_3d():
    """test an object traveling in line with sensor"""
    platform = GlobalOrigin3D
    tracker = tracker3d.BasicRazelRrtTracker(threshold_coast=5)
    target_xyzvel = np.array([20, 0, 0, 30, 0, 0])
    dt = 0.1
    nframes = 20
    for frame in range(nframes):
        target_xyzvel[0] += target_xyzvel[3] * dt
        target_xyzvel[1] += target_xyzvel[4] * dt
        target_xyzvel[2] += target_xyzvel[5] * dt
        dets = [
            RazelRrtDetection(
                razelrrt=xyzvel_to_razelrrt(target_xyzvel),
                source_identifier="radar",
                reference=platform,
                obj_type=None,
            )
        ]
        tracks = tracker(t=frame * dt, frame=frame, detections=dets, platform=platform)
    assert len(tracks) == 1
    assert (tracks[0].position - target_xyzvel[:3]).norm() < 3
    assert np.linalg.norm(tracks[0].rrt - dets[0].razelrrt[3]) < 5


def tests_transverse_razelrrt_tracker_3d():
    """test an object traveling transverse to the sensor"""
    platform = GlobalOrigin3D
    tracker = tracker3d.BasicRazelRrtTracker(threshold_coast=5)
    target_xyzvel = np.array([20, 0, 0, 0, 10, 0])
    dt = 0.1
    nframes = 20
    for frame in range(nframes):
        target_xyzvel[0] += target_xyzvel[3] * dt
        target_xyzvel[1] += target_xyzvel[4] * dt
        target_xyzvel[2] += target_xyzvel[5] * dt
        dets = [
            RazelRrtDetection(
                razelrrt=xyzvel_to_razelrrt(target_xyzvel),
                source_identifier="radar",
                reference=platform,
                obj_type=None,
            )
        ]
        tracks = tracker(t=frame * dt, frame=frame, detections=dets, platform=platform)
    assert len(tracks) == 1
    assert (tracks[0].position - target_xyzvel[:3]).norm() < 3
    assert np.linalg.norm(tracks[0].rrt - dets[0].razelrrt[3]) < 5


def tests_full_razelrrt_tracker_3d():
    """test an object traveling transverse to the sensor"""
    platform = GlobalOrigin3D
    tracker = tracker3d.BasicRazelRrtTracker(threshold_coast=5)
    target_xyzvel = np.array([20, 10, -5, 2, 10, 2.4])
    dt = 0.1
    nframes = 20
    for frame in range(nframes):
        target_xyzvel[0] += target_xyzvel[3] * dt
        target_xyzvel[1] += target_xyzvel[4] * dt
        target_xyzvel[2] += target_xyzvel[5] * dt
        det = [
            RazelRrtDetection(
                razelrrt=xyzvel_to_razelrrt(target_xyzvel),
                source_identifier="radar",
                reference=platform,
                obj_type=None,
            )
        ]
        tracks = tracker(t=frame * dt, frame=frame, detections=det, platform=platform)
    assert len(tracks) == 1
    assert (tracks[0].position - target_xyzvel[:3]).norm() < 3


def test_eagermot_fusion_kitti():
    dets_2d_all, dets_3d_all = make_kitti_2d_3d_tracking_data(n_frames=10)
    tracker_base = tracker3d.libraries.EagerMOT.model.EagerMOT()
    for dets_2d, dets_3d in zip(dets_2d_all, dets_3d_all):
        platform = dets_3d[0].reference
        # add false positives
        b2d = BoxDetection(
            name_2d, bbox.Box2D([10, 20, 20, 30], camera_calib), camera_calib.reference
        )
        pos = Position(np.array([10, -20, 2]), lidar_calib.reference)
        rot = Attitude(np.quaternion(1), lidar_calib.reference)
        b3d = BoxDetection(
            name_3d, bbox.Box3D(pos, rot, [1, 1, 1]), lidar_calib.reference
        )
        dets_2d.append(b2d)
        dets_3d.append(b3d)
        lone_2d, lone_3d, fused_instances = tracker_base.fusion(
            dets_2d, dets_3d, platform
        )
        assert len(lone_2d) == 1
        assert len(lone_3d) == 1
        assert len(fused_instances) == len(dets_3d) - 1


def test_eagermot_associations():
    dt = 0.10
    platform = GlobalOrigin3D
    dets_2d_all, dets_3d_all = make_kitti_2d_3d_tracking_data(dt=dt, n_frames=4)
    tracker = tracker3d.EagermotTracker()
    trk_base = tracker.tracker
    i = 0
    for frame, (dets_2d, dets_3d) in enumerate(zip(dets_2d_all, dets_3d_all)):
        # add false positives
        if i == 0:
            b2d = BoxDetection(
                name_3d,
                bbox.Box2D([10, 20, 20, 30], camera_calib),
                camera_calib.reference,
            )
            pos = Position(np.array([19, -20, 2]), lidar_calib.reference)
            rot = Attitude(np.quaternion(1), lidar_calib.reference)
            b3d = BoxDetection(
                name_2d,
                bbox.Box3D(pos, rot, [1, 1, 1]),
                platform,
            )
            dets_2d.append(b2d)
            dets_3d.append(b3d)

        # test association
        lone_2d, lone_3d, fused_instances = trk_base.fusion(dets_2d, dets_3d, platform)
        (
            assign_1,
            lone_fused_1,
            lone_3d_1,
            lone_track_1,
        ) = trk_base.tracker._first_data_association(lone_3d, fused_instances, platform)
        (
            assign_2,
            lone_fused_2,
            lone_2d_2,
            lone_track_2,
        ) = trk_base.tracker._second_data_association(
            lone_2d, lone_fused_1, lone_track_1, platform
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
            # assert len(assign_1) == len(dets_2d) == len(dets_3d)
            pass

        # run for real
        tracks = tracker(
            t=frame * dt,
            frame=frame,
            detections={"2d": dets_2d, "3d": dets_3d},
            platform=platform,
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
    platform = GlobalOrigin3D
    n_targs = 4
    dt = 0.10
    dets_2d_all, dets_3d_all = make_kitti_2d_3d_tracking_data(
        dt=dt, n_frames=4, n_targs=n_targs
    )
    tracker = tracker3d.EagermotTracker()
    tracks = tracker.tracker
    for frame, (dets_2d, dets_3d) in enumerate(zip(dets_2d_all, dets_3d_all)):
        tracks = tracker(
            t=frame * dt,
            frame=frame,
            detections={"2d": dets_2d, "3d": dets_3d},
            platform=platform,
            tracker="tracker-1",
        )
    assert len(tracks) == n_targs
    for i, trk in enumerate(tracks):
        for det in dets_3d:
            if trk.position.distance(det.box.t) < 2:
                break
        else:
            raise


def test_box_tracker_moving_reference():
    box1 = box_3d.deepcopy()
    dt = 0.1
    frames = 10
    v0 = np.array([5, 0, 0])
    tracker = tracker3d.BasicBoxTracker3D()
    for i in range(frames):
        reference = ReferenceFrame(
            box1.reference.x + v0 * dt * i, box1.reference.q, GlobalOrigin3D
        )
        dets = [
            BoxDetection(
                "boxes", box1.change_reference(reference, inplace=False), reference
            )
        ]
        tracks = tracker(t=dt * i, frame=i, detections=dets, platform=reference)
        if len(tracks) > 0:
            assert not tracks[0].reference.allclose(box1.reference)
    assert len(tracks) == 1
    assert box1.allclose(tracks[0].box.change_reference(box1.reference, inplace=False))
