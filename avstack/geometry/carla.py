# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-04
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-05-04
# @Description:
"""

"""

import numpy as np


##############################################


def carla_location_to_numpy_vector(carla_location):
    """
    Convert a carla location to a ROS vector3
    Considers the conversion from left-handed system (unreal) to right-handed
    system (ROS)
    :param carla_location: the carla location
    :type carla_location: carla.Location
    :return: a numpy.array with 3 elements
    :rtype: numpy.array
    """
    return np.array([carla_location.x, -carla_location.y, carla_location.z])


def carla_rotation_to_RPY(carla_rotation):
    """
    Convert a carla rotation to a roll, pitch, yaw tuple
    Considers the conversion from left-handed system (unreal) to right-handed
    system (ROS).
    Considers the conversion from degrees (carla) to radians (ROS).
    :param carla_rotation: the carla rotation
    :type carla_rotation: carla.Rotation
    :return: a tuple with 3 elements (roll, pitch, yaw)
    :rtype: tuple
    """
    roll = math.radians(carla_rotation.roll)
    pitch = -math.radians(carla_rotation.pitch)
    yaw = -math.radians(carla_rotation.yaw)
    return (roll, pitch, yaw)


def carla_rotation_to_rotation_matrix(carla_rotation):
    """
    Convert a carla rotation to a rotation matrix
    Considers the conversion from left-handed system (unreal) to right-handed
    Considers the conversion from degrees (carla) to radians (ROS).
    :param carla_rotation: the carla rotation
    :type carla_rotation: carla.Rotation
    :return: a numpy.array with 3x3 elements
    :rtype: numpy.array
    """
    roll, pitch, yaw = carla_rotation_to_RPY(carla_rotation)
    return tforms.transform_orientation(np.array([roll, pitch, yaw]), "euler", "DCM")


def carla_rotation_to_quaternion(carla_rotation):
    """
    Convert a carla rotation to a ROS quaternion
    Considers the conversion from left-handed system (unreal) to right-handed
    system (ROS).
    Considers the conversion from degrees (carla) to radians (ROS).
    :param carla_rotation: the carla rotation
    :type carla_rotation: carla.Rotation
    :return: a numpy.array with 3x3 elements
    :rtype: numpy.array
    """
    roll, pitch, yaw = carla_rotation_to_RPY(carla_rotation)
    return tforms.transform_orientation(np.array([roll, pitch, yaw]), "euler", "quat")
