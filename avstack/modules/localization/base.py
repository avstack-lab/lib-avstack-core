# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-04
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-28
# @Description:
"""

"""
import numpy as np
import quaternion


class _LocalizationAlgorithm:
    def __init__(self, t_init, ego_init, rate=100):
        self.t_last_exec = -np.inf
        self.rate = rate
        self._interval = 1 / rate
        self._last_estimate = None
        self.assign_from_ego(t_init, ego_init)

    def __call__(self, t, *args, **kwargs):
        """main call for localization"""
        assert t > self.t_last_exec, (t, self.t_last_exec)
        if (t - self.t_last_exec) >= (self._interval - 1e-5):
            self.t_last_exec = t
            ego_loc = self.execute(t, *args, **kwargs)
            self._last_estimate = ego_loc
            return ego_loc
        else:
            # -- propagate to queried time
            return self._last_estimate.predict(t - self._last_estimate.t)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = position

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, box):
        self._box = box

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = velocity

    @property
    def acceleration(self):
        return self._acceleration

    @acceleration.setter
    def acceleration(self, acceleration):
        self._acceleration = acceleration

    @property
    def attitude(self):
        return self._attitude

    @attitude.setter
    def attitude(self, attitude):
        self._attitude = attitude

    @property
    def angular_velocity(self):
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, angular_velocity):
        self._angular_velocity = angular_velocity

    def assign_from_ego(self, t, ego):
        self.t = t
        if ego is None:
            self.initialized = False
            self.position = None
            self.box = None
            self.velocity = None
            self.acceleration = None
            self.attitude = None
            self.angular_velocity = None
        else:
            self.initialized = True
            self.position = ego.position
            self.box = ego.box
            self.velocity = ego.velocity
            self.acceleration = ego.acceleration
            self.attitude = ego.attitude
            self.angular_velocity = ego.angular_velocity

    def execute(self, t, *args, **kwargs):
        raise NotImplementedError


class GroundTruthLocalizer(_LocalizationAlgorithm):
    def __init__(self, rate, t_init, ego_init):
        super().__init__(t_init, ego_init, rate)

    def execute(self, t, ground_truth, *args, **kwargs):
        assert t == ground_truth.timestamp, (t, ground_truth.timestamp)
        self.assign_from_ego(ground_truth.timestamp, ground_truth.ego_state)
        return ground_truth.ego_state
