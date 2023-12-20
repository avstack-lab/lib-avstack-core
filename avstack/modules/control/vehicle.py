from avstack.geometry import Pose
from avstack.utils.decorators import apply_hooks

from ..base import BaseModule
from .pid import PIDLateralController, PIDLongitudinalController
from .types import VehicleControlSignal


class VehiclePIDController(BaseModule):
    def __init__(
        self,
        args_lateral,
        args_longitudinal,
        max_throttle=0.75,
        max_brake=1.0,
        max_steering=0.8,
        name="vehiclepid",
        *args,
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
        super().__init__(name=name, *args, **kwargs)

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self.past_steering = 0.0
        self._lon_controller = PIDLongitudinalController(**args_longitudinal)
        self._lat_controller = PIDLateralController(**args_lateral)

    def update_coefficients(self, args_lateral, args_longitudinal):
        self._lon_controller.update_coefficients(**args_longitudinal)
        self._lat_controller.update_coefficients(**args_lateral)

    @apply_hooks
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
