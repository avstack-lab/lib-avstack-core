# -*- coding: utf-8 -*-
# @Author: Shucheng Z
# @Date:   2022-06-20
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""
import avstack
from avstack.modules.planning.components import CollisionDetection
import numpy as np
import quaternion

from avstack.environment import EnvironmentState
from avstack.modules import planning
from avstack.modules.perception.detections import LaneLineInSpace
from avstack.geometry import bbox, NominalOriginStandard
from avstack import transformations as tforms

from avstack import GroundTruthInformation
from avstack.modules import prediction
from avstack.modules import tracking
from avstack.objects import VehicleState

import sys
sys.path.append('tests/')
from utilities import get_ego


def set_seed():
    seed = 10
    np.random.seed(seed)

box_obj = bbox.Box3D([2,2,5,[0,0,0],np.quaternion(1)], NominalOriginStandard)  # box in local coordinates


def get_object_no_collision(ego):
    set_seed()
    pos_obj = ego.position + 10*np.random.rand(3)
    vel_obj = np.random.rand(3)
    acc_obj = np.random.rand(3)
    rot_obj = np.eye(3)
    ang_obj = np.random.rand(3)
    obj = VehicleState('car')
    obj.set(0, pos_obj, box_obj, vel_obj, acc_obj, rot_obj, ang_obj, origin=NominalOriginStandard)
    return obj


def get_object_collision(ego):
    set_seed()
    pos_obj = ego.position + ego.velocity.vector * 3 + 0.5 * ego.acceleration.vector * 3 ** 2
    vel_obj = np.zeros(3)
    acc_obj = np.zeros(3)
    rot_obj = np.eye(3)
    ang_obj = np.random.rand(3)
    obj = VehicleState('car')
    obj.set(0, pos_obj, box_obj, vel_obj, acc_obj, rot_obj, ang_obj, origin=NominalOriginStandard)
    return obj


def get_object_collision_yaw(ego):
    set_seed()
    pos_obj = ego.position + ego.velocity.vector * 3 + 0.5 * ego.acceleration.vector * 3 ** 2
    vel_obj = np.zeros(3)
    acc_obj = np.zeros(3)
    rot_obj = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) # Anticlockwise 90 degree
    ang_obj = np.random.rand(3)
    obj = VehicleState('car')
    obj.set(0, pos_obj, box_obj, vel_obj, acc_obj, rot_obj, ang_obj, origin=NominalOriginStandard)
    return obj


def test_collision_detection():
    set_seed()
    predictor = prediction.KinematicPrediction(dt_pred=0.1, t_pred_forward=3)

    ego = get_ego(1)
    obj0 = get_object_no_collision(ego)
    obj1 = get_object_collision(ego)
    obj2 = get_object_collision_yaw(ego)

    frame = timestamp = 0
    ground_truth = GroundTruthInformation(frame, timestamp, ego_state=ego, objects=[obj0, obj1, obj2])
    tracker = tracking.tracker3d.GroundTruthTracker()
    tracks = tracker(frame, ground_truth)
    pred_ego = predictor(frame, tracks); pred_ego = pred_ego[list(pred_ego.keys())[0]]
    preds_3d = predictor(frame, tracks)
    collision_detection = CollisionDetection(ego, tracks)
    collision_records = collision_detection.collision_monitor(pred_ego, preds_3d)

    # assert collision_records[1] == 2.5
    # assert collision_records[2] in [1.0, 1.5]
