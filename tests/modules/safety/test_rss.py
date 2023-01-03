# -*- coding: utf-8 -*-
# @Author: spencer@primus
# @Date:   2022-06-15
# @Last Modified by:   Spencer H
# @Last Modified time: 2022-06-16

import sys
from copy import copy, deepcopy
import numpy as np
from avstack.objects import VehicleState
from avstack.modules import safety, localization

sys.path.append('tests/')
from utilities import get_test_sensor_data, get_ego, get_object_global
# obj, box_calib, lidar_calib, pc, camera_calib, img, box_2d, box_3d = get_test_sensor_data()

def test_rss_interface():
    if safety.use_rss:
        ego = get_ego(1)
        obj = get_object_global(2)
        rss_eval = safety.RssEvaluator()
        rss_eval(ego, [obj])
    else:
        print('cannot test rss safety')


def get_obj_vector_relative(ego, d, vector, obj_type='car'):
    obj = VehicleState(obj_type)
    pos = ego.position + d * vector
    box = ego.box
    box.t = pos
    obj.set(t=ego.t, position=pos, box=box, velocity=np.zeros((3,)), attitude=ego.attitude)
    assert np.isclose(ego.position.distance(obj.position), d)
    return obj


def get_obj_forward_relative(ego, d_forward, obj_type='car'):
    return get_obj_vector_relative(ego, d_forward, ego.attitude.forward_vector, obj_type=obj_type)


def get_obj_left_relative(ego, d_lateral, obj_type='car'):
    return get_obj_vector_relative(ego, d_lateral, ego.attitude.left_vector, obj_type=obj_type)


# ===========================================
# Testing singular metrics
# ===========================================

def test_rss_forward_unsafe():
    if safety.use_rss:
        ego = get_ego(1)
        obj = get_obj_forward_relative(ego, 3)
        rss_eval = safety.RssEvaluator()
        safety_metric = rss_eval(ego, [obj], verbose=False)
        assert not safety_metric.safe


def test_rss_forward_safe():
    if safety.use_rss:
        ego = get_ego(1)
        obj = get_obj_forward_relative(ego, 30)
        rss_eval = safety.RssEvaluator()
        safety_metric = rss_eval(ego, [obj])
        assert safety_metric.safe


def test_rss_left_unsafe():
    if safety.use_rss:
        ego = get_ego(1)
        obj = get_obj_left_relative(ego, 0.5)
        rss_eval = safety.RssEvaluator()
        safety_metric = rss_eval(ego, [obj], verbose=False)
        assert not safety_metric.safe
        assert safety_metric.dangerous_objects == [obj.ID]


def test_rss_left_safe():
    if safety.use_rss:
        ego = get_ego(1)
        obj = get_obj_left_relative(ego, 5)
        rss_eval = safety.RssEvaluator()
        safety_metric = rss_eval(ego, [obj], verbose=False)
        assert safety_metric.safe


# ===========================================
# Testing aggregate metrics
# ===========================================

def test_rss_agg_all_safe():
    if safety.use_rss:
        ego = get_ego(1)
        d_forward = [29, 30, 40]
        objs = [get_obj_forward_relative(ego, d) for d in d_forward]
        rss_eval = safety.RssEvaluator()
        safety_metric = rss_eval(ego, objs, verbose=False)
        assert safety_metric.safe
        assert len(safety_metric.dangerous_objects) == 0


def test_rss_agg_one_unsafe():
    if safety.use_rss:
        ego = get_ego(1)
        d_forward = [3, 30, 40]
        objs = [get_obj_forward_relative(ego, d) for d in d_forward]
        rss_eval = safety.RssEvaluator()
        safety_metric = rss_eval(ego, objs, verbose=False)
        assert not safety_metric.safe
        assert safety_metric.dangerous_objects == [objs[0].ID]

# ===========================================
# Testing other objects
# ===========================================

def test_rss_car():
    if safety.use_rss:
        import ad_rss as ad
        ego = get_ego(1)
        d_forward = 30
        obj = get_obj_forward_relative(ego, d_forward)
        rss_eval = safety.RssEvaluator()
        assert rss_eval._rss_object(ego, obj).objectType == ad.rss.world.ObjectType.OtherVehicle


def test_rss_ped_safe():
    if safety.use_rss:
        import ad_rss as ad
        ego = get_ego(1)
        d_left = 10
        obj = get_obj_left_relative(ego, d_left, obj_type='pedestrian')
        rss_eval = safety.RssEvaluator()
        assert rss_eval._rss_object(ego, obj).objectType == ad.rss.world.ObjectType.Pedestrian
        safety_metric = rss_eval(ego, [obj], verbose=False)
        assert safety_metric.safe


def test_rss_ped_unsafe():
    if safety.use_rss:
        ego = get_ego(1)
        d_forward = 5
        obj = get_obj_forward_relative(ego, d_forward, obj_type='pedestrian')
        rss_eval = safety.RssEvaluator()
        safety_metric = rss_eval(ego, [obj], verbose=False)
        assert not safety_metric.safe
