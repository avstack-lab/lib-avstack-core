import quaternion

from .base import q_mult_vec
from .bbox import Box2D, Box3D, BoxDecoder, SegMask2D
from .coordinates import CameraCoordinates, StandardCoordinates
from .datastructs import (
    Acceleration,
    AngularVelocity,
    Attitude,
    PointMatrix2D,
    PointMatrix3D,
    Pose,
    Position,
    RotationDecoder,
    Twist,
    VectorDecoder,
    Velocity,
)
from .fov import Circle, Shape, Sphere, Vesica, Wedge
from .planes import GroundPlane, plane_2_transform
from .refchoc import (
    GlobalOrigin3D,
    PassiveReferenceFrame,
    ReferenceDecoder,
    ReferenceFrame,
    Rotation,
    RotationEncoder,
    Vector,
    VectorEncoder,
    VectorHeadTail,
)
from .transformations import transform_orientation


R_stan_to_cam = StandardCoordinates.get_conversion_matrix(CameraCoordinates)
q_stan_to_cam = quaternion.from_rotation_matrix(R_stan_to_cam)
R_cam_to_stan = R_stan_to_cam.T
q_cam_to_stan = q_stan_to_cam.conjugate()


__all__ = [
    "Acceleration",
    "AngularVelocity",
    "Attitude",
    "Box2D",
    "Box3D",
    "BoxDecoder",
    "Circle",
    "GlobalOrigin3D",
    "GroundPlane",
    "plane_2_transform",
    "PointMatrix2D",
    "PointMatrix3D",
    "Pose",
    "Position",
    "q_mult_vec",
    "PassiveReferenceFrame",
    "ReferenceDecoder",
    "ReferenceFrame",
    "Rotation",
    "RotationDecoder",
    "RotationEncoder",
    "SegMask2D",
    "Shape",
    "Sphere",
    "transform_orientation",
    "Twist",
    "Vector",
    "VectorDecoder",
    "VectorEncoder",
    "VectorHeadTail",
    "Velocity",
    "Vesica",
    "Wedge",
]
