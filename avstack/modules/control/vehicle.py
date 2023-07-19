# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-06
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-27
# @Description:
"""

"""


from avstack.geometry import Pose

from .pid import PIDLateralController, PIDLongitudinalController


class VehicleControlSignal:
    def __init__(
        self,
        throttle=0.0,
        brake=0.0,
        steer=0.0,
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False,
    ):
        self.throttle = throttle
        self.brake = brake
        self.steer = steer
        self.hand_brake = hand_brake
        self.manual_gear_shift = manual_gear_shift
        self.reverse = reverse

    @property
    def throttle(self):
        return self._throttle

    @throttle.setter
    def throttle(self, throttle):
        assert 0 <= throttle <= 1.0, throttle
        self._throttle = throttle

    @property
    def brake(self):
        return self._brake

    @brake.setter
    def brake(self, brake):
        assert 0.0 <= brake <= 1.0, brake
        self._brake = brake

    @property
    def steer(self):
        return self._steer

    @steer.setter
    def steer(self, steer):
        assert -1.0 <= steer <= 1.0, steer
        self._steer = steer

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"VehicleControlSignal -- throttle: {self.throttle}, brake: {self.brake}, steer: {self.steer}"


# ====================================================
# INSTANCES
# ====================================================


class _ControlAlgorithm:
    pass


class VehiclePIDController(_ControlAlgorithm):
    def __init__(
        self,
        args_lateral,
        args_longitudinal,
        max_throttle=0.75,
        max_brake=1.0,
        max_steering=0.8,
        **kwargs,
    ):
        """
        Constructor method.

        :param args_lateral: dictionary of arguments to set the lateral PID controller
        using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal
        PID controller using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        """

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self.past_steering = 0.0
        self._lon_controller = PIDLongitudinalController(**args_longitudinal)
        self._lat_controller = PIDLateralController(**args_lateral)

    def update_coefficients(self, args_lateral, args_longitudinal):
        self._lon_controller.update_coefficients(**args_longitudinal)
        self._lat_controller.update_coefficients(**args_lateral)

    def __call__(self, ego_state, plan):
        """Run lateral and longitudinal PID control

        :ego_state - current vehicle state
        :plan - queue of waypoints for planning
        """

        # -- get details from waypoint
        t = ego_state.t
        waypoint_target = plan.top()[1]
        current_speed = ego_state.velocity.norm()
        target_speed = waypoint_target.target_speed
        current_point = Pose(ego_state.position, ego_state.attitude)
        target_point = waypoint_target.target_point

        # -- apply details
        acceleration = self._lon_controller(t, current_speed, target_speed)
        current_steering = self._lat_controller(t, current_point, target_point)
        if acceleration >= 0.0:
            throttle = min(acceleration, self.max_throt)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.
        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1
        if current_steering >= 0:
            steer = min(self.max_steer, current_steering)
        else:
            steer = max(-self.max_steer, current_steering)
        self.past_steering = steer

        return VehicleControlSignal(throttle=throttle, brake=brake, steer=steer)
