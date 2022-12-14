# -*- coding: utf-8 -*-
# @Author: Shucheng Zhang
# @Date:   2022-06-01
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""

import numpy as np
from avstack import GroundTruthInformation
from avstack.geometry import bbox, NominalOriginStandard
from avstack.modules import tracking, perception, prediction
from avstack.objects import VehicleState


import sys
sys.path.append('tests/')
from utilities import get_ego, get_object_global

frame = 0


def test_kinematic_prediction():
    # -- test prediction
    detector = perception.object3d.GroundTruth3DObjectDetector()
    tracker = tracking.tracker3d.Ab3dmotTracker(10)
    predictor = prediction.KinematicPrediction(0.1, 3)

    ego = get_ego(2)
    obj1 = get_object_global(1)
    frame = timestamp = 0
    ground_truth = GroundTruthInformation(frame, timestamp, ego_state=ego, objects=[obj1])

    objects_3d = tracker(frame, detector(frame, ground_truth, identifier='detector-1'), identifier='tracker-1')
    objects_2d = []
    preds_3d = predictor(frame, objects_3d)
    assert len(preds_3d) == 1
    ID = list(preds_3d.keys())[0]
    assert len(preds_3d[ID]) == predictor.t_forward / predictor.dt
    t = predictor.dt
    for t in predictor.dt_predicts:
        assert np.all(preds_3d[ID][t].position == objects_3d[0].position + objects_3d[0].velocity*t)
        assert np.all(preds_3d[ID][t].velocity == objects_3d[0].velocity)
