from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from avstack.geometry import ReferenceFrame

import numpy as np

from avstack.geometry import conversions


class _Detection:
    def __init__(
        self,
        data: Any,
        frame: "ReferenceFrame",
        score: float = None,
        object_class: str = None,
    ):
        self.data = data
        self.reference = frame
        self.score = score
        self.object_class = object_class


class MaskDetection(_Detection):
    pass


class BoxDetection(_Detection):
    pass


class Box2DDetection(BoxDetection):
    pass


class Box3DDetection(BoxDetection):
    pass


class CartesianDetection(_Detection):
    pass


class XyDetection(CartesianDetection):
    pass


class XyzDetection(CartesianDetection):
    pass


class SphericalDetection(_Detection):
    pass


class RazDetection(SphericalDetection):
    @property
    def xy(self):
        return conversions.spherical_to_cartesian_2d(self.data)

    @xy.setter
    def xy(self, xy: np.ndarray):
        self.data = conversions.cartesian_to_spherical_2d(xy)


class RazelDetection(SphericalDetection):
    @property
    def xyz(self):
        return conversions.spherical_to_cartesian_3d(self.data)

    @xyz.setter
    def xyz(self, xyz: np.ndarray):
        self.data = conversions.cartesian_to_spherical_3d(xyz)


class RazelRrtDetection(RazelDetection):
    @property
    def xyzrrt(self):
        return np.array([*self.xyz, self.data[3]])
