from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .primitives import Pose
    from .frame import ReferenceFrame, TransformManager

import numpy as np

from avstack.exceptions import FrameEquivalenceError


class BoxSize:
    def __init__(self, height, width, length=None):
        self.height = height
        self.width = width
        self.length = length

    @property
    def area(self):
        if self.length:
            raise ValueError("Cannot call area when length is not None")
        else:
            return self.height * self.width

    @property
    def volume(self):
        if not self.length:
            raise ValueError("Cannot call volume when length is None")
        else:
            return self.height * self.width * self.length


class BoundingBoxDecoder:
    pass


class BoundingBox2D:
    def __init__(self, xyxy: np.ndarray, frame: "ReferenceFrame"):
        self.xyxy = xyxy
        self.reference = frame

    @property
    def center(self):
        return np.array(
            [(self.xyxy[2] + self.xyxy[0]) / 2, (self.xyxy[3] + self.xyxy[1]) / 2]
        )

    @property
    def xmin(self):
        return self.xyxy[0]

    @property
    def ymin(self):
        return self.xyxy[1]

    @property
    def xmax(self):
        return self.xyxy[2]

    @property
    def ymax(self):
        return self.xyxy[3]


class BoundingBox2Dcwh(BoundingBox2D):
    def __init__(
        self, center: np.ndarray, width: float, height: float, frame: "ReferenceFrame"
    ):
        xyxy = np.array(
            [
                center[0] - width / 2,
                center[1] - height / 2,
                center[0] + width / 2,
                center[1] + height / 2,
            ]
        )
        super().__init__(xyxy, frame)


class BoundingBox2Dxyxy(BoundingBox2D):
    def __init__(self, xyxy: np.ndarray, frame: "ReferenceFrame"):
        super().__init__(xyxy, frame)


class BoundingBox3D:
    def __init__(self, pose: "Pose", box: "BoxSize"):
        self.pose = pose
        self.box = box

    @property
    def corners(self):
        raise NotImplementedError

    @property
    def frame(self):
        return self.pose.frame

    def IoU(self, other: "BoundingBox3D"):
        if self.frame != other.frame:
            raise FrameEquivalenceError(self.frame, other.frame)
        else:
            raise NotImplementedError

    def change_reference(self, frame: "ReferenceFrame", tm: "TransformManager"):
        self.pose.change_reference(frame, tm)


class SegMask2D:
    pass
