import numpy as np
from copy import deepcopy
from .refchoc import ReferenceFrame, Vector, Rotation
from .base import q_mult_vec
from . import transformations as tforms
import avstack


class Position(Vector):
    def _pull_from_reference(self, reference: ReferenceFrame):
        return reference.x

    @staticmethod
    def factory():
        return Position


class Velocity(Vector):
    def _pull_from_reference(self, reference: ReferenceFrame):
        return reference.v

    @staticmethod
    def factory():
        return Velocity


class Acceleration(Vector):
    def _pull_from_reference(self, reference: ReferenceFrame):
        return reference.acc

    @staticmethod
    def factory():
        return Acceleration


class Attitude(Rotation):
    def _pull_from_reference(self, reference: ReferenceFrame):
        return reference.q

    @staticmethod
    def factory():
        return Attitude


class AngularVelocity(Rotation):
    def _pull_from_reference(self, reference: ReferenceFrame):
        return reference.ang

    @staticmethod
    def factory():
        return AngularVelocity


class Pose:
    def __init__(self, position: Vector, rotation: Rotation) -> None:
        self.position = position
        self.rotation = rotation


class Twist:
    def __init__(self, linear: Vector, angular: Vector) -> None:
        self.linear = linear
        self.angular = angular


class _PointMatrix:
    def __init__(self, x: np.ndarray, calibration) -> None:
        self.x = x
        if isinstance(calibration, ReferenceFrame):
            calibration = avstack.calibration.Calibration(calibration)
        self.calibration = calibration

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, x):
        if len(x.shape) == 1:
            x = x[:,None]
        self._x = x

    @property
    def reference(self):
        return self.calibration.reference

    @property
    def shape(self):
        return self.x.shape

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, indices):
        return self.x[indices]


class PointMatrix3D(_PointMatrix):

    def change_calibration(self, calibration, inplace: bool=False):
        return self.change_reference(calibration.reference, inplace)

    def change_reference(self, reference, inplace: bool=False):
        """Change of reference frame of a vector

        Step 1: compute the differential
        Step 2: apply the differential to this object

        self.x : x_ref1_to_point_in_ref1
        diff.x : x_ref1_to_ref2_in_ref1
        diff.q : q_ref1_to_ref2

        x : x_ref2_to_point_in_ref2 <-- diff.q * (self.x - diff.x)
        """
        diff = self.calibration.reference.differential(reference, in_self=True)  # self to other
        x = q_mult_vec(diff.q, self.x[:,:3] - diff.x)
        if inplace:
            self.x = x
            self.calibration.reference = reference
        else:
            calib = deepcopy(self.calibration)
            calib.reference = reference
            return PointMatrix3D(x, calib)
        
    def filter(self, mask):
        return PointMatrix3D(self.x[mask,:], self.calibration)

    def project_to_2d(self, calibration):
        pts_3d_img = self.change_calibration(calibration).x[:,:3]
        pts_3d_hom = tforms.cart2hom(pts_3d_img)
        pts_2d = np.dot(pts_3d_hom, np.transpose(calibration.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        pts_2d_cam = pts_2d[:, 0:2]
        return PointMatrix2D(pts_2d_cam, calibration)
    

class PointMatrix2D(_PointMatrix):

    @property
    def angles(self):
        """Takes pixel coordinates and get angles

        assumes x_pixels has (0,0) in the top left

        Returns the (az, el) angles
        """
        x_centered = self.calibration.principal_point - self.x
        # azel = x_centered * np.array([self.pixel_size_u, self.pixel_size_v])
        if len(x_centered.shape) == 1:
            azel = np.array(
                [
                    np.arctan2(x_centered[0], self.calibration.f_u),
                    np.arctan2(x_centered[1], self.calibration.f_v),
                ]
            )
        else:
            azel = np.zeros_like(x_centered)[:, :2]
            azel[:, 0] = np.arctan2(x_centered[:, 0], self.calibration.f_u)
            azel[:, 1] = np.arctan2(x_centered[:, 1], self.calibration.f_v)
        return azel