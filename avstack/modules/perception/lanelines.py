# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-27
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-28
# @Description:
"""

"""
import os

from avstack.modules.perception import detections
from avstack.modules.perception.base import _PerceptionAlgorithm


class GroundTruthLaneLineDetector(_PerceptionAlgorithm):
    MODE = "lane lines"

    def __call__(self, frame, ground_truth, *args, **kwargs):
        """Wrap ground truths to detections"""
        return ground_truth.lane_lines


class LaneNet(_PerceptionAlgorithm):
    pass
