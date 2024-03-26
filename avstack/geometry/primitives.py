from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from .frame import FrameTransform, ReferenceFrame, TransformManager

from avstack.exceptions import FrameEquivalenceError

from .frame import WorldFrame


class Vector:
    def __init__(self, x: np.ndarray, frame: "ReferenceFrame" = WorldFrame):
        self.x = x
        self.reference = frame

    def copy(self):
        return Vector(self.x.copy(), self.frame)

    def change_reference(
        self,
        frame: "ReferenceFrame",
        tm: "TransformManager",
        T: "FrameTransform" = None,
    ):
        if not T:
            T = tm.get_transform(from_frame=self.frame, to_stamp=stamp)
        raise NotImplementedError


class Rotation:
    def __init__(self, q: np.array, frame: "ReferenceFrame" = WorldFrame):
        self.q = q
        self.reference = frame

    def copy(self):
        return Rotation(self.q.copy(), self.frame)

    def change_reference(
        self,
        frame: "ReferenceFrame",
        tm: "TransformManager",
        T: "FrameTransform" = None,
    ):
        if not T:
            T = tm.get_transform(from_frame=self.frame, to_stamp=stamp)
        raise NotImplementedError


class Pose:
    def __init__(self, position: "Vector", attitude: "Rotation"):
        self.position = position
        self.attitude = attitude
        if position.frame != attitude.frame:
            raise FrameEquivalenceError(position.frame, attitude.frame)

    def copy(self):
        return Pose(self.position.copy(), self.attitude.copy())

    def change_reference(self, frame: "ReferenceFrame", tm: "TransformManager"):
        T = tm.get_transform(from_frame=self.frame, to_stamp=stamp)  # for efficiency
        self.position.change_reference(frame, tm, T=T)
        self.attitude.change_reference(frame, tm, T=T)


class PointMatrix:
    def __init__(self, x: np.ndarray, frame: "ReferenceFrame") -> None:
        self.x = x
        self.reference = frame

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        if len(x.shape) == 1:
            x = x[:, None]
        self._x = x

    @property
    def shape(self):
        return self.x.shape

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, indices):
        return self.x[indices]

    def copy(self):
        return self.__class__(self.x.copy(), self.frame)


class PointMatrix3D(PointMatrix):
    pass


class PointMatrix2D(PointMatrix):
    pass
