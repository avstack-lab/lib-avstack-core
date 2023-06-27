# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-03-23
# @Last Modified by:   spencer@primus
# @Last Modified date: 2022-09-08
# @Description:
"""

"""

import quaternion
from .coordinates import StandardCoordinates, CameraCoordinates
from .planes import GroundPlane, plane_2_transform
from .datastructs import PointMatrix2D, PointMatrix3D, Pose, Twist, Position, Velocity, Acceleration, Attitude, AngularVelocity
from .refchoc import ReferenceFrame, GlobalOrigin3D, Rotation, Vector, VectorHeadTail, get_reference_from_line
from .transformations import transform_orientation
from .base import q_mult_vec
from .bbox import Box2D, Box3D, SegMask2D
from . import transformations

R_stan_to_cam = StandardCoordinates.get_conversion_matrix(CameraCoordinates)
q_stan_to_cam = quaternion.from_rotation_matrix(R_stan_to_cam)
R_cam_to_stan = R_stan_to_cam.T
q_cam_to_stan = q_stan_to_cam.conjugate()