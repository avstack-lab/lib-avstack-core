# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-03-23
# @Last Modified by:   spencer@primus
# @Last Modified date: 2022-09-08
# @Description:
"""

"""

import quaternion

from . import transformations
from .base import q_mult_vec
from .bbox import Box2D, Box3D, SegMask2D
from .coordinates import CameraCoordinates, StandardCoordinates
from .datastructs import (
    Acceleration,
    AngularVelocity,
    Attitude,
    PointMatrix2D,
    PointMatrix3D,
    Pose,
    Position,
    Twist,
    Velocity,
)
from .planes import GroundPlane, plane_2_transform
from .refchoc import (
    GlobalOrigin3D,
    ReferenceFrame,
    Rotation,
    Vector,
    VectorHeadTail,
    get_reference_from_line,
)
from .transformations import transform_orientation


R_stan_to_cam = StandardCoordinates.get_conversion_matrix(CameraCoordinates)
q_stan_to_cam = quaternion.from_rotation_matrix(R_stan_to_cam)
R_cam_to_stan = R_stan_to_cam.T
q_cam_to_stan = q_stan_to_cam.conjugate()
