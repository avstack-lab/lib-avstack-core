# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-03
# @Last Modified by:   spencer@primus
# @Last Modified date: 2022-08-07
# @Description:
"""

"""

import numpy as np

import avstack.geometry


class Coordinates:
    """Coordinate conventions"""

    def __init__(self, forward, left, up):
        self.forward = forward
        self.left = left
        self.up = up

        self._parity = [forward[0], left[0], up[0]]
        self._axis = [forward[1], left[1], up[1]]
        self._parity_map = {"-": -1, "+": 1}
        self._axis_map = {"x": 0, "y": 1, "z": 2}
        self._vec_map = {
            "x": np.array([1, 0, 0], dtype=int),
            "y": np.array([0, 1, 0], dtype=int),
            "z": np.array([0, 0, 1], dtype=int),
        }

    def _vector(self, idx):
        return self._parity_map[self._parity[idx]] * self._vec_map[self._axis[idx]]

    def _index(self, idx):
        return abs(self._axis_map[self._axis[idx]])

    @property
    def forward_vector(self):
        return self._vector(0)

    @property
    def left_vector(self):
        return self._vector(1)

    @property
    def up_vector(self):
        return self._vector(2)

    @property
    def forward_index(self):
        return self._index(0)

    @property
    def left_index(self):
        return self._index(1)

    @property
    def up_index(self):
        return self._index(2)

    @property
    def matrix(self):
        """Gets global 2 local conversion matrix"""
        return np.vstack((self.forward_vector, self.left_vector, self.up_vector)).T

    def __eq__(self, other):
        return (
            (self.forward == other.forward)
            & (self.left == other.left)
            & (self.up == other.up)
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Coordinate Object With: Forward={self.forward}, Left={self.left}, Up={self.up}"

    def get_conversion_matrix(self, other):
        """Convert rotation matrix between coordinates"""
        assert isinstance(other, Coordinates)
        if self == other:
            return np.eye(3)
        else:
            # gets local_self 2 local_other matrix via o_g2l @ s_g2l.T
            return other.matrix @ self.matrix.T

    def convert(self, data, other):
        """Convert a numpy array to another coordinate system"""
        assert isinstance(other, Coordinates)
        if self == other:
            newdata = data
        elif isinstance(data, avstack.geometry.Vector):
            newdata = avstack.geometry.Vector(
                other, self.convert(data.vector, other)
            )
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                data = np.asarray(data)[:, None].T
                do_squeeze = True
            elif len(data.shape) == 2:
                # assert data.shape[1] == 3, f'Incompatible data shape of {data.shape}'
                do_squeeze = False
            else:
                raise NotImplementedError
            newdata = data.copy()
            cardinal = ["x", "y", "z"]
            for i, s in enumerate(cardinal):
                newdata[:, i] = data[
                    :, cardinal.index(self._axis[other._axis.index(s)])
                ]
                if (
                    self._parity[other._axis.index(s)]
                    != other._parity[other._axis.index(s)]
                ):
                    newdata[:, i] *= -1
            if do_squeeze:
                newdata = np.squeeze(newdata)
        else:
            raise NotImplementedError(
                f"Converting data of type {type(data)} not implemented"
            )

        return newdata

    def project(self, data, projection):
        """Projects into a particular system

        fv -- [left, up]
        bev -- [forward, left]
        """
        # get the projection data
        assert projection in ["fv", "bev"]
        if isinstance(data, avstack.geometry.Vector):
            data = data.vector
        elif isinstance(data, np.ndarray):
            pass
        else:
            raise NotImplementedError(
                f"Projecting data of type {type(data)} not implemented"
            )

        # do the projection
        if projection == "fv":
            newdata = np.array([data[self.left_index], data[self.up_index]])
        elif projection == "bev":
            newdata = np.array([data[self.forward_index], data[self.left_index]])
        else:
            raise NotImplementedError
        return newdata


class StandardCoordinates_(Coordinates):
    """Standard 'lidar' coordinate system"""

    def __init__(self):
        super().__init__("+x", "+y", "+z")

    def __str__(self):
        return "StandardCoordinates"


class LidarCoordinates_(StandardCoordinates_):
    pass


class LidarCoordinatesYForward_(Coordinates):
    """Standard 'lidar' coordinate system"""

    def __init__(self):
        super().__init__("+y", "-x", "+z")

    def __str__(self):
        return "StandardCoordinatesRotated90"


class CameraCoordinates_(Coordinates):
    """Standard 'camera' coordinate system"""

    def __init__(self):
        super().__init__("+z", "-x", "-y")

    def __str__(self):
        return "CameraCoordinates"


class CarlaCoordinates_(Coordinates):
    """Carlas left handed coordinate system"""

    def __init__(self):
        super().__init__("+x", "-y", "+z")

    def __str__(self):
        return "CarlaCoordinates"


class EnuCoordinates_(Coordinates):
    """East-North-Up coordinate system"""

    def __init__(self):
        super().__init__("+y", "-x", "+z")

    def __str__(self):
        return "ENUCoordinates"


def get_coordinate_class(forward, left, up):
    if (forward == "+x") and (left == "+y") and (up == "+z"):
        return StandardCoordinates
    elif (forward == "+z") and (left == "-x") and (up == "-y"):
        return CameraCoordinates
    else:
        raise RuntimeError(f"Cannot find coordinates for (){forward}, {left}, {up})")


StandardCoordinates = StandardCoordinates_()
LidarCoordinates = LidarCoordinates_()
LidarCoordinatesYForward = LidarCoordinatesYForward_()
CameraCoordinates = CameraCoordinates_()
CarlaCoordinates = CarlaCoordinates_()
EnuCoordinates = EnuCoordinates_()


def cross_product_coord(c1, c2):
    """Assumes that this is c1 x c2 = c3...user can mod parity outside"""
    if c1[0] == c2[0]:
        parity = "+"
    else:
        parity = "-"
    dirs = ["x", "y", "z"]
    dirs.remove(c1[1])
    dirs.remove(c2[1])
    return parity + dirs[0]


def get_coordinates_from_string(c_string):
    if c_string.lower() == "standardcoordinates":
        C = LidarCoordinates
    elif c_string.lower() == "cameracoordinates":
        C = CameraCoordinates
    else:
        raise NotImplementedError(c_string.lower())
    return C
