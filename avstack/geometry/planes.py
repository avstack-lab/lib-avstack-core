# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-19
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""
import numpy as np
import quaternion

from .refchoc import PassiveReferenceFrame, ReferenceFrame, Rotation, Vector


def plane_2_transform(plane):
    """Convert a plane-based transform to a transform object

    Transform is "sensor to ground"
    """
    dz = plane.p[3]
    trans_sens_to_ground = Vector([0, 0, dz], reference=plane.reference)
    up = plane.p[:3]
    forward_tmp = np.array([1, 0, 0])
    left = np.cross(up, forward_tmp)
    forward = np.cross(left, up)
    R_sensor_2_ground = np.vstack((forward, left, up)).T
    q_sensor_2_ground = quaternion.from_rotation_matrix(R_sensor_2_ground)
    rotation = Rotation(q_sensor_2_ground, reference=plane.reference)
    return Transform(rotation, trans_sens_to_ground)


# class Plane3D():
#     def __init__(self, plane_coeffs, calib):
#         """
#         Creates plane equations in the different reference frames

#         Assumption: plane_coeffs coming in are in image frame
#         """
#         self.p_image = plane_coeffs
#         self.p_lidar = np.asarray([plane_coeffs[2], -plane_coeffs[0], -plane_coeffs[1], plane_coeffs[3] + 0.08])


class GroundPlane:
    def __init__(self, plane_coeffs, reference):
        """Creates plane equation in reference frame

        First 3 elements define normal vector of the plane
        Last element describes the height of the sensor relative to the plane
        """
        self.p = np.asarray(plane_coeffs)
        assert isinstance(reference, (PassiveReferenceFrame, ReferenceFrame))
        self.reference = reference
        self.normal = self.p[:3] / np.linalg.norm(self.p[:3])

    def __str__(self):
        return f"Ground Plane with cooefficients:\n{self.p} in {self.reference}"

    def angle_between(self, other):
        """Gets the angle between normal vectors of two planes"""
        if isinstance(other, GroundPlane):
            assert (
                self.reference == other.reference
            ), "For now references must be the same"
            v1 = self.normal
            v2 = other.normal
            angle = np.arccos(np.dot(v1, v2))
        else:
            raise NotImplementedError
        return angle

    def as_reference(self):
        """Convert a plane-based transform to a transform object

        Transform is "sensor to ground"
        """
        dz = self.p[3]
        trans_sens_to_ground = Vector([0, 0, dz], reference=self.reference)
        up = self.p[:3]
        forward_tmp = np.array([1, 0, 0])
        left = np.cross(up, forward_tmp)
        forward = np.cross(left, up)
        R_sensor_2_ground = np.vstack((forward, left, up)).T
        q_sensor_2_ground = quaternion.from_rotation_matrix(R_sensor_2_ground)
        rotation = Rotation(q_sensor_2_ground, reference=self.reference)
        return ReferenceFrame(trans_sens_to_ground.x, rotation.q, self.reference)
