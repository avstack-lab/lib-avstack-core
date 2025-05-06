class VehicleControlSignal:
    def __init__(
        self,
        timestamp=None,
        throttle=0.0,
        brake=0.0,
        steer=0.0,
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False,
        bounded=True,
    ):
        """Define the vehicle control signal"""
        self.bounded = bounded
        self.timestamp = timestamp
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
        if self.bounded:
            assert 0 <= throttle <= 1.0, throttle
        self._throttle = throttle

    @property
    def brake(self):
        return self._brake

    @brake.setter
    def brake(self, brake):
        if self.bounded:
            assert 0.0 <= brake <= 1.0, brake
        self._brake = brake

    @property
    def steer(self):
        return self._steer

    @steer.setter
    def steer(self, steer):
        if self.bounded:
            assert -1.0 <= steer <= 1.0, steer
        self._steer = steer

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"VehicleControlSignal at time {self.timestamp:.3f} -- throttle: {self.throttle}, brake: {self.brake}, steer: {self.steer}"
