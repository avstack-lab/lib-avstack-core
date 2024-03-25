import json

import numpy as np

import avstack

from . import transformations as tforms
from .base import fastround, q_mult_vec
from .frame import ReferenceFrame, ReferenceFrameDecoder, TransformManager


class VectorEncoder(json.JSONEncoder):
    def default(self, o):
        v_dict = {
            "x": o.x.tolist(),
            "reference": o.reference.encode(),
            "n_prec": o.n_prec,
        }
        return {type(o).__name__.lower(): v_dict}


class RotationEncoder(json.JSONEncoder):
    def default(self, o):
        q_dict = {
            "qw": o.q.w,
            "qv": o.q.vec.tolist(),
            "reference": o.reference.encode(),
            "n_prec": o.n_prec,
        }
        return {type(o).__name__.lower(): q_dict}


class VectorDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "vector" in json_object:
            json_object = json_object["vector"]
            factory = Vector
        elif "position" in json_object:
            json_object = json_object["position"]
            factory = Position
        elif "velocity" in json_object:
            json_object = json_object["velocity"]
            factory = Velocity
        elif "acceleration" in json_object:
            json_object = json_object["acceleration"]
            factory = Acceleration
        else:
            return json_object
        if json_object is None:
            return None
        reference = json.loads(json_object["reference"], cls=ReferenceFrameDecoder)
        return factory(
            x=np.array(json_object["x"]),
            reference=reference,
            n_prec=json_object["n_prec"],
        )


class RotationDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "rotation" in json_object:
            json_object = json_object["rotation"]
            factory = Rotation
        elif "attitude" in json_object:
            json_object = json_object["attitude"]
            factory = Attitude
        elif "angular" in json_object:
            json_object = json_object["angular"]
            factory = AngularVelocity
        elif "angularvelocity" in json_object:
            json_object = json_object["angularvelocity"]
            factory = AngularVelocity
        else:
            return json_object
        if json_object is None:
            return None
        reference = json.loads(json_object["reference"], cls=ReferenceFrameDecoder)
        return factory(
            q=np.quaternion(json_object["qw"], *json_object["qv"]),
            reference=reference,
            n_prec=json_object["n_prec"],
        )


class Vector:
    def __init__(
        self, x: np.ndarray, reference: ReferenceFrame, n_prec: int = 8
    ) -> None:
        self.n_prec = n_prec
        self.x = np.asarray(x, dtype=float)
        self.reference = reference

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        y = np.empty_like(x)
        self._x = fastround(x, self.n_prec, y)

    @property
    def finite(self):
        return np.all(np.isfinite(self.x))

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, reference):
        assert isinstance(reference, ReferenceFrame)
        self._reference = reference

    def __str__(self):
        return f"{type(self)} - {self.x}, {self.reference}"

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        return self.x

    def __getitem__(self, key: int):
        return self.x[key]

    def __setitem__(self, key: int, value: float):
        self.x[key] = np.round(value, self.n_prec)

    def __neg__(self):
        return self.factory()(-self.x, self.reference, n_prec=self.n_prec)

    def __add__(self, other: "Vector", inplace: bool = False):
        # Perform wrapping
        if isinstance(other, Vector):
            if self.reference != other.reference:
                raise RuntimeError(
                    f"Reference frames are not equivalent - {self.reference} vs {other.reference}"
                )
            other = other.x
        elif isinstance(other, (int, np.ndarray, float)):
            pass
        else:
            raise NotImplementedError(type(other))

        # Perform addition
        if inplace:
            self.x = self.x + other
        else:
            return self.factory()(self.x + other, self.reference, self.n_prec)

    def __sub__(self, other: "Vector"):
        return -(-self + other)  # have to do this weird order!!

    def __mul__(self, other: "Vector", inplace: bool = False):
        # Perform wrapping
        if isinstance(other, Vector):
            if self.reference != other.reference:
                raise RuntimeError(
                    f"Reference frames are not equivalent - {self.reference} vs {other.reference}"
                )
            other = other.x
        elif isinstance(other, (int, np.ndarray, float)):
            pass
        else:
            raise NotImplementedError(type(other))

        # Perform multiplication
        if inplace:
            self.x = self.x * other
        else:
            return self.factory()(self.x * other, self.reference, self.n_prec)

    def __truediv__(self, other, inplace: bool = False):
        # Wrapping
        if isinstance(other, (int, np.ndarray, float)):
            pass
        else:
            raise NotImplementedError(type(other))

        # Perform division
        if inplace:
            self.x = self.x / other
        else:
            return self.factory()(self.x / other, self.reference, self.n_prec)

    def __matmul__(self, other):
        # Perform wrapping
        if isinstance(other, Vector):
            if self.reference != other.reference:
                other = other.change_reference(self.reference, inplace=False)
            other = other.x
        elif isinstance(other, (np.ndarray)):
            pass
        else:
            raise NotImplementedError(type(other))

        # Perform dot product
        return self.x @ other

    def encode(self):
        return json.dumps(self, cls=VectorEncoder)

    def allclose(self, other: "Vector"):
        if self.reference == other.reference:
            return np.allclose(self.x, other.x)
        else:
            other = other.change_reference(self.reference, inplace=False)
            return np.allclose(self.x, other.x)

    def change_reference(
        self,
        tm: TransformManager,
        reference: ReferenceFrame,
        inplace: bool,
        angle_only: bool = False,
    ):
        """Change of reference frame of a vector

        Step 1: compute the differential (self to other)
        Step 2: apply the differential to this object

        self.x : x_ref1_to_point_in_ref1
        diff.x : x_ref1_to_ref2_in_ref1
        diff.q : q_ref1_to_ref2

        x : x_ref2_to_point_in_ref2 <-- diff.q * (self.x - diff.x)
        """
        diff = self.reference.differential(reference, in_self=True)  # self to other
        diff_x = self._pull_from_reference(diff)
        if angle_only:
            x = q_mult_vec(diff.q, self.x)
        else:
            x = q_mult_vec(diff.q, self.x - diff_x)
        if inplace:
            self.x = x
            self.reference = reference
        else:
            return self.factory()(x, reference, self.n_prec)

    @staticmethod
    def factory():
        return Vector

    def _pull_from_reference(self, reference: ReferenceFrame):
        return reference.x

    def sqrt(self):
        return np.sqrt(self.x)

    def norm(self):
        return np.linalg.norm(self.x)

    def unit(self):
        return self.factory()(self.x / np.linalg.norm(self.x), self.reference)

    def distance(self, other):
        return (self - other).norm()


class VectorHeadTail:
    def __init__(
        self,
        head: np.ndarray,
        tail: np.ndarray,
        reference: ReferenceFrame,
        n_prec: int = 8,
    ) -> None:
        self.n_prec = n_prec
        self.head = Vector(head, reference, n_prec=n_prec)
        self.tail = Vector(tail, reference, n_prec=n_prec)

    def change_reference(
        self, reference: ReferenceFrame, inplace: bool, angle_only: bool = False
    ):
        if inplace:
            self.head.change_reference(
                reference, inplace=inplace, angle_only=angle_only
            )
            self.tail.change_reference(
                reference, inplace=inplace, angle_only=angle_only
            )
        else:
            head = self.head.change_reference(
                reference, inplace=inplace, angle_only=angle_only
            )
            tail = self.head.change_reference(
                reference, inplace=inplace, angle_only=angle_only
            )
            return VectorHeadTail(head.x, tail.x, reference)


class Spherical(Vector):
    def __init__(
        self, x: np.ndarray, reference: ReferenceFrame, n_prec: int = 8, wrapping="v1"
    ) -> None:
        """

        wrapping:
            - v1: azimuth wraps to [-pi, pi]
                  elevation wraps [-pi/2, pi/2]
            - v2: azimuth wraps to [0, 2pi]
                  elevation wraps to [-pi/2, pi/2]
        """
        super().__init__(x, reference, n_prec)
        assert wrapping in ["v1", "v2"]
        if self.wrapping == "v1":
            self.wrap_az = lambda az: (az + np.pi) % (2 * np.pi) - np.pi
            self.wrap_el = lambda el: (el + np.pi / 2) % np.pi - np.pi / 2
        elif self.wrapping == "v2":
            self.wrap_az = lambda az: az % (2 * np.pi)
            self.wrap_el = lambda el: (el + np.pi / 2) % np.pi - np.pi / 2
        else:
            raise NotImplementedError(wrapping)
        self.wrapping = wrapping

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        if len(x) > 1:
            x[1] = self.wrap_az(x[1])
        if len(x) > 2:
            x[2] = self.wrap_el(x[2])
        y = np.empty_like(x)
        self._x = fastround(x, self.n_prec, y)


class Rotation:
    """
    Rotation is defined as q_reference_to_object
    """

    def __init__(self, q: np.quaternion, reference: ReferenceFrame, n_prec=8) -> None:
        self.n_prec = n_prec
        if isinstance(q, np.quaternion):
            pass
        elif isinstance(q, np.ndarray) and ((q.shape == (3, 3)) or (q.shape == (2, 2))):
            q = tforms.transform_orientation(q, "dcm", "quat")
        else:
            raise ValueError(f"{type(q)} must be quaternion or 3x3 or 2x2")
        self.q = q
        self.reference = reference

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, reference):
        assert isinstance(reference, ReferenceFrame)
        self._reference = reference

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        y = np.empty_like(q.vec)
        self._q = np.quaternion(
            np.round(q.w, self.n_prec), *fastround(q.vec, self.n_prec, y)
        )

    @property
    def qw(self):
        return self.q.w

    @property
    def qx(self):
        return self.q.x

    @property
    def qy(self):
        return self.q.y

    @property
    def qz(self):
        return self.q.z

    @property
    def R(self):
        return tforms.transform_orientation(self.q, "quat", "dcm")

    @property
    def euler(self):
        return tforms.transform_orientation(self.q, "quat", "euler")

    @property
    def forward_vector(self):
        return self.R[0, :]

    @property
    def left_vector(self):
        return self.R[1, :]

    @property
    def up_vector(self):
        return self.R[2, :]

    @property
    def yaw(self):
        return self.euler[2]

    def conjugate(self):
        return self.factory()(self.q.conjugate(), self.reference)

    def __str__(self):
        return f"{type(self)} - {self.q}, {self.reference}"

    def __repr__(self):
        return str(self)

    def __mul__(self, other: "Rotation", inplace: bool = False):
        if self.reference != other.reference:
            other = other.change_reference(self.reference, inplace=False)
        if inplace:
            self.q = self.q * other.q
        else:
            return self.factory()(self.q * other.q, self.reference, self.n_prec)

    def __matmul__(self, other):
        raise NotImplementedError

    def encode(self):
        return json.dumps(self, cls=RotationEncoder)

    def allclose(self, other: "Rotation"):
        if self.reference == other.reference:
            return np.allclose(self.q.vec, other.q.vec)
        else:
            other = other.change_reference(self.reference, inplace=False)
            return np.allclose(self.q.vec, other.q.vec)

    def angle_between(self, other: "Rotation"):
        if self.reference != other.reference:
            other = other.change_reference(self.reference, inplace=False)
        return 2 * np.arcsin(np.linalg.norm((self.q * other.q.conjugate()).vec))

    def change_reference(self, reference: ReferenceFrame, inplace: bool):
        """Change of reference frame of a vector

        Step 1: compute the differential
        Step 2: apply the differential to this object
        TODO: could do angles only...

        self.q : q_ref1_to_point
        diff.q : q_ref1_to_ref2

        q : q_ref2_to_point <-- q_ref1_to_point * q_ref1_to_ref2.conjugate()
        """
        diff = self.reference.differential(reference, in_self=True)  # self to other
        diff_q = self._pull_from_reference(diff)
        q = self.q * diff_q.conjugate()
        if inplace:
            self.q = q
            self.reference = reference
        else:
            return self.factory()(q, reference, self.n_prec)

    def _pull_from_reference(self, reference: ReferenceFrame):
        return reference.q

    @staticmethod
    def factory():
        return Rotation


class Position(Vector):
    def _pull_from_reference(self, reference: ReferenceFrame):
        return reference.x

    @property
    def position(self):
        return self

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
    def __init__(self, position: Position, attitude: Attitude) -> None:
        self.position = position
        self.attitude = attitude

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        if not isinstance(position, Position):
            raise TypeError(f"Input position type {type(position)} not allowed")
        self._position = position

    @property
    def attitude(self):
        return self._attitude

    @attitude.setter
    def attitude(self, attitude):
        if not isinstance(attitude, Attitude):
            raise TypeError(f"Input position type {type(attitude)} not allowed")
        self._attitude = attitude

    @property
    def matrix(self):
        return np.block(
            [[self.attitude.R, self.position.x], [np.zeros((1, 3)), np.ones((1, 1))]]
        )

    def change_reference(self, reference, inplace: bool):
        if inplace:
            self.position.change_reference(reference=reference, inplace=inplace)
            self.attitude.change_reference(reference=reference, inplace=inplace)
        else:
            position = self.position.change_reference(
                reference=reference, inplace=inplace
            )
            attitude = self.attitude.change_reference(
                reference=reference, inplace=inplace
            )
            return Pose(position, attitude)


class Twist:
    def __init__(self, linear: Velocity, angular: AngularVelocity) -> None:
        self.linear = linear
        self.angular = angular

    @property
    def linear(self):
        return self._linear

    @linear.setter
    def linear(self, linear):
        if not isinstance(linear, Velocity):
            raise TypeError(f"Input position type {type(linear)} not allowed")
        self._linear = linear

    @property
    def angular(self):
        return self._angular

    @angular.setter
    def angular(self, angular):
        if not isinstance(angular, AngularVelocity):
            raise TypeError(f"Input position type {type(angular)} not allowed")
        self._angular = angular

    def change_reference(self, reference, inplace: bool):
        if inplace:
            self.linear.change_reference(reference=reference, inplace=inplace)
            self.angular.change_reference(reference=reference, inplace=inplace)
        else:
            linear = self.linear.change_reference(reference=reference, inplace=inplace)
            angular = self.angular.change_reference(
                reference=reference, inplace=inplace
            )
            return Twist(linear, angular)


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
            x = x[:, None]
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

    def copy(self):
        return self.__class__(self.x.copy(), self.calibration)


class PointMatrix3D(_PointMatrix):
    def change_calibration(self, calibration, inplace: bool):
        return self.change_reference(calibration.reference, inplace)

    def change_reference(self, reference, inplace: bool):
        """Change of reference frame of a vector

        Step 1: compute the differential
        Step 2: apply the differential to this object

        self.x : x_ref1_to_point_in_ref1
        diff.x : x_ref1_to_ref2_in_ref1
        diff.q : q_ref1_to_ref2

        x : x_ref2_to_point_in_ref2 <-- diff.q * (self.x - diff.x)
        """
        diff = self.calibration.reference.differential(
            reference, in_self=True
        )  # self to other
        x = q_mult_vec(diff.q, self.x[:, :3] - diff.x)
        if self.x.shape[1] > 3:
            x = np.hstack((x, self.x[:, 3:]))
        if inplace:
            self.x = x
            self.calibration.reference = reference
        else:
            try:
                calib = self.calibration.__class__(reference)
            except TypeError as e:
                calib = self.calibration.__class__(
                    reference, self.calibration.P, self.calibration.img_shape
                )
            return PointMatrix3D(x, calib)

    def filter(self, mask):
        return PointMatrix3D(self.x[mask, :], self.calibration)

    def project_to_2d(self, calibration):
        pts_3d_img = self.change_calibration(calibration, inplace=False).x[:, :3]
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
