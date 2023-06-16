# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-03-23
# @Last Modified by:   spencer@primus
# @Last Modified date: 2022-09-08
# @Description:
"""

"""

import numpy as np
import quaternion

from .coordinates import (
    CameraCoordinates,
    CarlaCoordinates,
    Coordinates,
    EnuCoordinates,
    LidarCoordinates,
    LidarCoordinatesYForward,
    StandardCoordinates,
    cross_product_coord,
    get_coordinates_from_string,
)
from .planes import GroundPlane, plane_2_transform
from .primitives import (
    Origin,
    Pose,
    Rotation,
    Transform,
    Translation,
    Twist,
    VectorDirMag,
    VectorHeadTail,
    get_origin_from_line,
    get_transform_from_line,
)
from .transformations import transform_orientation


R_stan_to_cam = StandardCoordinates.get_conversion_matrix(CameraCoordinates)
q_stan_to_cam = quaternion.from_rotation_matrix(R_stan_to_cam)
R_cam_to_stan = R_stan_to_cam.T
q_cam_to_stan = q_stan_to_cam.conjugate()

NominalOriginStandard = Origin(np.zeros((3,)), np.quaternion(1))
NominalOriginCamera = Origin(np.zeros((3,)), q_stan_to_cam)

NominalRotation = Rotation(np.quaternion(1), origin=NominalOriginStandard)
NominalTranslation = Translation([0, 0, 0], origin=NominalOriginStandard)
NominalTransform = Transform(NominalRotation, NominalTranslation)


from .base import q_mult_vec
from .bbox import Box2D, Box3D, SegMask2D
