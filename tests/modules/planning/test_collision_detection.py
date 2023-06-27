# -*- coding: utf-8 -*-
# @Author: Shucheng Z
# @Date:   2022-06-20
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""
import sys

import numpy as np

from avstack import GroundTruthInformation
from avstack.environment.objects import VehicleState
from avstack.geometry import GlobalOrigin3D, bbox, Position, Velocity, Acceleration, AngularVelocity, Attitude, transformations as tforms
from avstack.modules import prediction, tracking
from avstack.modules.planning.components import CollisionDetection


sys.path.append("tests/")
from utilities import get_ego


pos_obj = Position(np.array([0, 0, 0]), GlobalOrigin3D)
rot_obj = Attitude(np.quaternion(1), GlobalOrigin3D)
box_obj = bbox.Box3D(pos_obj, rot_obj, [2,2,5])


def set_seed():
    seed = 10
    np.random.seed(seed)


def get_object_no_collision(ego):
    set_seed()
    ref = ego.position.reference
    pos_obj = ego.position + 10 * np.random.rand(3)
    vel_obj = Velocity(np.random.rand(3), ref)
    acc_obj = Acceleration(np.random.rand(3), ref)
    rot_obj = Attitude(np.quaternion(1), ref)
    ang_obj = AngularVelocity(np.quaternion(*np.random.rand(3)), ref)
    obj = VehicleState("car")
    obj.set(
        0,
        pos_obj,
        box_obj,
        vel_obj,
        acc_obj,
        rot_obj,
        ang_obj,
    )
    return obj


def get_object_collision(ego):
    set_seed()
    ref = ego.position.reference
    pos_obj = (
        ego.position + ego.velocity * 3 + ego.acceleration * 0.5 * 3**2
    )
    vel_obj = Velocity(np.zeros(3), ref)
    acc_obj = Acceleration(np.zeros(3), ref)
    rot_obj = Attitude(np.quaternion(1), ref)
    ang_obj = AngularVelocity(np.quaternion(*np.random.rand(3)), ref)
    obj = VehicleState("car")
    obj.set(
        0,
        pos_obj,
        box_obj,
        vel_obj,
        acc_obj,
        rot_obj,
        ang_obj,
    )
    return obj


def get_object_collision_yaw(ego):
    set_seed()
    ref = ego.position.reference
    pos_obj = (
        ego.position + ego.velocity * 3 + ego.acceleration * 0.5 * 3**2
    )
    vel_obj = Velocity(np.zeros(3), ref)
    acc_obj = Acceleration(np.zeros(3), ref)
    q = tforms.transform_orientation(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]), 'dcm', 'quat')  # Anticlockwise 90 degree
    rot_obj = Attitude(q, ref)
    ang_obj = AngularVelocity(np.quaternion(*np.random.rand(3)), ref)
    obj = VehicleState("car")
    obj.set(
        0,
        pos_obj,
        box_obj,
        vel_obj,
        acc_obj,
        rot_obj,
        ang_obj,
    )
    return obj


def test_collision_detection():
    set_seed()
    predictor = prediction.KinematicPrediction(dt_pred=0.1, t_pred_forward=3)

    ego = get_ego(1)
    obj0 = get_object_no_collision(ego)
    obj1 = get_object_collision(ego)
    obj2 = get_object_collision_yaw(ego)

    frame = timestamp = 0
    ground_truth = GroundTruthInformation(
        frame, timestamp, ego_state=ego, objects=[obj0, obj1, obj2]
    )
    tracker = tracking.tracker3d.GroundTruthTracker()
    tracks = tracker(t=timestamp, frame=frame, detections=None, ground_truth=ground_truth)
    pred_ego = predictor(tracks, frame=frame)
    pred_ego = pred_ego[list(pred_ego.keys())[0]]
    preds_3d = predictor(tracks, frame=frame)
    collision_detection = CollisionDetection(ego, tracks)
    collision_records = collision_detection.collision_monitor(pred_ego, preds_3d)