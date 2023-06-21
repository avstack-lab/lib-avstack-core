# -*- coding: utf-8 -*-
# @Author: Shucheng Zhang
# @Date:   2022-06-01
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""

import sys

import numpy as np

from avstack import GroundTruthInformation
from avstack.modules import perception, prediction, tracking


sys.path.append("tests/")
from utilities import get_ego, get_object_global


frame = 0


def test_kinematic_prediction():
    # -- test prediction
    detector = perception.object3d.GroundTruth3DObjectDetector()
    tracker = tracking.tracker3d.Ab3dmotTracker()
    predictor = prediction.KinematicPrediction(0.1, 3)

    ego = get_ego(2)
    obj1 = get_object_global(1)
    frame = timestamp = 0
    ground_truth = GroundTruthInformation(
        frame, timestamp, ego_state=ego, objects=[obj1]
    )

    objects_3d = tracker(
        t=0.10 * frame,
        frame=frame,
        detections=detector(ground_truth, frame=frame, identifier="detector-1"),
        identifier="tracker-1",
    )
    objects_2d = []
    preds_3d = predictor(objects_3d, frame=frame)
    assert len(preds_3d) == 1
    ID = list(preds_3d.keys())[0]
    assert len(preds_3d[ID]) == predictor.t_forward / predictor.dt
    t = predictor.dt
    for t in predictor.dt_predicts:
        assert np.all(
            preds_3d[ID][t].position
            == objects_3d[0].position + objects_3d[0].velocity * t
        )
        assert np.all(preds_3d[ID][t].velocity == objects_3d[0].velocity)
