import numpy as np
import quaternion  # noqa

from .bbox import (
    BoundingBox2D,
    BoundingBox2Dcwh,
    BoundingBox2Dxyxy,
    BoundingBox3D,
    BoundingBoxDecoder,
    BoxSize,
)
from .coordinates import CameraCoordinates, StandardCoordinates
from .fov import Circle, Shape, Sphere, Vesica, Wedge
from .frame import (
    FrameTransform,
    ReferenceFrame,
    ReferenceFrameDecoder,
    Transform,
    TransformManager,
    TransformManagerDecoder,
    WorldFrame,
)
from .primitives import (
    PointMatrix,
    PointMatrix2D,
    PointMatrix3D,
    Pose,
    Rotation,
    RotationDecoder,
    Vector,
    VectorDecoder,
)


R_stan_to_cam = StandardCoordinates.get_conversion_matrix(CameraCoordinates)
q_stan_to_cam = quaternion.from_rotation_matrix(R_stan_to_cam)
R_cam_to_stan = R_stan_to_cam.T
q_cam_to_stan = q_stan_to_cam.conjugate()


__all__ = [
    "BoundingBox2D",
    "BoundingBox2Dcwh",
    "BoundingBox2Dxyxy",
    "BoundingBox3D",
    "BoundingBoxDecoder",
    "BoxSize",
    "PointMatrix",
    "PointMatrix3D",
    "PointMatrix2D",
    "Pose",
    "ReferenceFrame",
    "ReferenceFrameDecoder",
    "Rotation",
    "TransformManager",
    "TransformManagerDecoder",
    "Vector",
    "WorldFrame",
]
