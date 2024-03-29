import math
from collections import deque

import numpy as np

from avstack.geometry import Pose, Vector
from avstack.utils.decorators import apply_hooks

from ..base import BaseModule


"""
TODO:
- Derivative smoothing over the buffer window
"""


class PIDBase(BaseModule):
    def __init__(self, K_P, K_D, K_I, buffer_len=10, name="pid", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self.buffer_len = buffer_len
        self.reset_buffer()

    @apply_hooks
    def __call__(self, t, error, clip_low=None, clip_high=None):
        if len(self._t_buffer) > 0:
            assert t >= self._t_buffer[-1], f"{t}, {self._t_buffer[-1]}"
            if t == self._t_buffer[-1]:
                self._t_buffer.pop()
                self._error_buffer.pop()
        self._t_buffer.append(t)
        self._error_buffer.append(error)
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / (
                self._t_buffer[-1] - self._t_buffer[-2]
            )
            _ie = np.trapz(self._error_buffer, self._t_buffer)
        else:
            _de = 0.0
            _ie = 0.0
        value = (self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie)
        if clip_low or clip_high:
            value = np.clip(value, clip_low, clip_high)
        return value

    def update_coefficients(self, K_P, K_D, K_I, buffer_len):
        if self.buffer_len != buffer_len:
            raise NotImplementedError
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I

    def reset_buffer(self):
        self._error_buffer = deque(maxlen=self.buffer_len)
        self._t_buffer = deque(maxlen=self.buffer_len)


class _PIDController(BaseModule):
    def __init__(
        self, K_P, K_D, K_I, buffer_len, name="pidcontroller", *args, **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)
        self.controller = PIDBase(K_P, K_D, K_I, buffer_len)

    def update_coefficients(self, K_P, K_D, K_I, buffer_len=None):
        if buffer_len is None:
            buffer_len = self.buffer_len
        self.controller.update_coefficients(K_P, K_D, K_I, buffer_len)

    def reset_buffer(self):
        self.controller.reset_buffer()


class PIDLongitudinalController(_PIDController):
    def __init__(self, K_P=1.0, K_D=0.0, K_I=0.0, buffer_len=10, *args, **kwargs):
        super().__init__(K_P, K_D, K_I, buffer_len, *args, **kwargs)

    @apply_hooks
    def __call__(self, t, current_speed: float, target_speed: float, debug=False):
        error = target_speed - current_speed
        return self.controller(t, error, clip_low=-1.0, clip_high=1.0)


class PIDLateralController(_PIDController):
    def __init__(self, K_P=1.95, K_D=0.2, K_I=0.05, buffer_len=10, *args, **kwargs):
        super().__init__(K_P, K_D, K_I, buffer_len, *args, **kwargs)
        self.last_error = None

    @apply_hooks
    def __call__(self, t, current: Pose, target: Pose):
        """Run lateral control

        current and target are in global coordinates, standard
        """
        v_begin = current.position
        xyz = [math.cos(current.attitude.yaw), math.sin(target.attitude.yaw), 0]
        v_end = v_begin + Vector(xyz, reference=v_begin.reference)
        v_vec = np.array([v_end.x[0] - v_begin.x[0], v_end.x[1] - v_begin.x[1], 0.0])
        w_vec = np.array(
            [
                target.position.x[0] - v_begin.x[0],
                target.position.x[1] - v_begin.x[1],
                0.0,
            ]
        )
        error = -math.acos(
            np.clip(
                np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)),
                -1.0,
                1.0,
            )
        )
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            error *= -1.0

        # wrap between -pi to pi
        error = (error + math.pi) % (2 * math.pi) - math.pi

        # protect against angle jumps by flushing buffer
        if self.last_error is not None:
            if abs(self.last_error) > math.pi / 2:
                if np.sign(error) != np.sign(self.last_error):
                    self.reset_buffer()
        self.last_error = error

        # protect against unstable behavior
        # TODO
        c_val = self.controller(t, error, clip_low=-0.5, clip_high=0.5)
        return c_val
