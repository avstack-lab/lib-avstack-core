# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-20
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-05-20
# @Description:
"""

"""

from enum import Enum


mph_per_mps = 2.23694  # 1 m/s = 2.23694 mph


class TrafficLevel(Enum):
    NONE = 1
    LIGHT = 2
    MODERATE = 3
    HEAVY = 4
