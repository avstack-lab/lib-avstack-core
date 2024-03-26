from dataclasses import dataclass

import numpy as np

from avstack.time import Stamp


class TransformManagerDecoder:
    pass


class ReferenceFrameDecoder:
    pass


class TransformManager:
    def __init__(self):
        self.transforms = [WorldFrame]

    def add_transform(self, transform: "FrameTransform"):
        self.transforms.append(transform)

    def get_transform(self, from_frame, to_frame, time) -> "FrameTransform":
        raise NotImplementedError


@dataclass(frozen=True)
class ReferenceFrame:
    name: str
    stamp: Stamp


@dataclass(frozen=True)
class FrameTransform:
    from_frame: str
    to_frame: str
    transform: "Transform"


@dataclass(frozen=True)
class Transform:
    x: np.ndarray
    q: np.ndarray
    v: np.ndarray = None


WorldFrame = ReferenceFrame("world", None)
