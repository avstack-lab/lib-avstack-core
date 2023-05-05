# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-10
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""
import sys

import numpy as np

from avstack.environment import EnvironmentState
from avstack.geometry import (
    NominalOriginStandard,
    Translation,
)
from avstack.modules import planning
from avstack.modules.perception.detections import LaneLineInSpace
from avstack.modules.planning.vehicle import components


sys.path.append("tests/")
from utilities import get_ego, get_object_global


def get_lanes():
    l_pts = [Translation([i, 4, 0], origin=NominalOriginStandard) for i in range(20)]
    left_lane = LaneLineInSpace(l_pts)
    r_pts = [Translation([i, -3, 0], origin=NominalOriginStandard) for i in range(20)]
    right_lane = LaneLineInSpace(r_pts)
    return [[left_lane, right_lane]]


def get_tracks(ego, seed):
    trk_0 = get_object_global(seed)
    trk_1 = get_object_global(seed + 1)
    return [ego.global_to_local(trk_0), ego.global_to_local(trk_1)]


def test_random_planner():
    np.random.seed(10)
    plan = planning.WaypointPlan()
    planner = planning.vehicle.RandomPlanner()
    ego_state = get_ego(1)
    plan = planner(plan, ego_state)
    assert len(plan) == 1
    w = plan.pop()
    assert len(plan) == 0
    plan = planner(plan, ego_state)
    assert len(plan) == 1


def test_object_following():
    plan = planning.WaypointPlan()
    planner = planning.vehicle.AdaptiveCruiseControl()
    ego_state = get_ego(1)
    environment = EnvironmentState()
    objects = []
    tracks = {}
    predictions = {}
    lane_lines = get_lanes()
    # -- without objects
    plan = planner(
        plan,
        ego_state=ego_state,
        environment=environment,
        objects=objects,
        objects_3d=tracks,
        objects_2d=[],
        signs=[],
        lights=[],
        lane_lines=lane_lines,
    )
    # -- with objects
    tracks = get_tracks(ego_state, 99)
    track_follow = components.get_object_to_follow(ego_state, tracks, lane_lines)
    assert track_follow == tracks[1]
    plan = planner(
        plan,
        ego_state=ego_state,
        environment=environment,
        objects=objects,
        objects_3d=tracks,
        objects_2d=[],
        signs=[],
        lights=[],
        lane_lines=lane_lines,
    )
