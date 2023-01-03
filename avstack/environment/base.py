# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-20
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-27
# @Description:
"""

"""

from .traffic import TrafficLevel

mph_per_mps = 2.23694  # 1 m/s = 2.23694 mph


class EnvironmentState():
    def __init__(self):
        self.speed_limit = 35 / mph_per_mps
        self.work_zone = False
        self.traffic_level = TrafficLevel.NONE

    @property
    def speed_limit(self):
        return self._speed_limit

    @speed_limit.setter
    def speed_limit(self, speed_limit):
        assert speed_limit > 0
        self._speed_limit = speed_limit

    def update(self, ego_state, objects_3d, objects_2d, lights, signs, lanes):
        pass
        # self._update_traffic(self, )