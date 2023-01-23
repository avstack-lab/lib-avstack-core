# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-03
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-10-22
# @Description:
"""

"""

from copy import copy, deepcopy

import numpy as np
import quaternion
from numba import jit
from numba.types import float64, int64

from avstack import transformations as tforms
from avstack.geometry import CameraCoordinates, StandardCoordinates

from .base import q_mult_vec


def get_origin_from_line(line):
    items = line.split()
    assert items[0] == "origin", items
    x = np.array([float(r) for r in items[1:4]])
    q = np.quaternion(*[float(w) for w in items[4:8]])
    return Origin(x, q)


def get_transform_from_line(line):
    items = line.split()
    assert items[0] == "transform", items
    idx_tr = items.index("translation")
    R = get_rotation_from_line(" ".join(items[1:idx_tr]))
    Tr = get_translation_from_line(" ".join(items[idx_tr:]))
    return Transform(R, Tr)


def get_rotation_from_line(line):
    items = line.split()
    assert items[0] == "rotation", items
    q = np.quaternion(*[float(r) for r in items[1:5]])
    origin = get_origin_from_line(" ".join(items[5:]))
    return Rotation(q, origin=origin)


def get_translation_from_line(line):
    items = line.split()
    assert items[0] == "translation", items
    v = [float(x) for x in items[1:4]]
    origin = get_origin_from_line(" ".join(items[4:]))
    return Translation(v, origin=origin)


@jit(float64[:](float64[:], int64, float64[:]), nopython=True)
def fastround(arr, ndec, out):
    for i in range(len(arr)):
        if np.abs(arr[i]) < 1e-12:
            arr[i] = 0.0
    return np.round_(arr, ndec, out)


class Origin:
    """An origin is like a transform but avoids circular reference

    An origin can be attached to a vector to say where the "zero" of
    that vector is in the global space. An origin can be attached
    to sensor data to show where the zero of that sensor is in the
    global space.

    x := translation from 'global/reference' to 'local' in 'global/reference' coordinates
        --> this is like x_origin_2_object in origin's frame
    q := rotation from 'global/reference' to 'local'
        --> this is like q_origin_2_object in origin's frame

    """

    def __init__(self, x: np.ndarray, q: np.quaternion, n_prec=8):
        # -- parse quaternion/rotation
        if isinstance(x, np.quaternion) and isinstance(q, np.ndarray):
            x, q = q, x
        if isinstance(q, np.quaternion):
            self.q = q
        elif isinstance(q, (list, np.ndarray)) and (len(q) == 4):
            self.q = quaternion.from_float_array(q)
        elif isinstance(q, np.ndarray) and q.shape == (3, 3):
            self.q = tforms.transform_orientation(q, "dcm", "quat")
        else:
            raise ValueError(f"{type(q)} must be quaternion or 3x3\n{q}")

        # -- parse translation
        if isinstance(x, np.ndarray) and (len(x) == 3):
            self.x = x.astype(np.float64)
        elif isinstance(x, list) and (len(x) == 3):
            self.x = np.array(x, dtype=np.float64)
        else:
            raise ValueError(f"{x} must be a list of ndaray of len 3")

        # -- round results
        y = np.empty_like(self.x)
        self.x = fastround(self.x, n_prec, y)
        y = np.empty_like(self.x)
        self.q = np.quaternion(
            np.round(self.q.w, n_prec), *fastround(self.q.vec, n_prec, y)
        )

    def __hash__(self):
        return hash(self.x.tobytes() + self.q.vec.tobytes())

    def allclose(self, other):
        if isinstance(other, Origin):
            if hash(self) == hash(other):
                return True
            else:
                return quaternion.allclose(self.q, other.q) and np.allclose(
                    self.x, other.x
                )
        else:
            raise NotImplementedError(
                f"Cannot check equality between origin and {type(other)}"
            )

    def __eq__(self, other):
        if isinstance(other, Origin):
            if hash(self) == hash(other):
                return True
            else:
                return False
                # c1 = quaternion.allclose(self.q, other.q)
                # c2 = np.all(self.x == other.x)
                # return c1 and c2
        else:
            raise NotImplementedError(
                f"Cannot check equality between origin and {type(other)}"
            )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"origin of x:{self.x}, q:{self.q}"

    @property
    def x_b(self):
        """Translation from a to b in b's frame

        NOTE:
        x_a_to_b_in_a == -x_b_2_a_in_a
        x_a_to_b_in_b == q_a_2_b * x_a_to_b_in_a

        The elements are as follows:
        self.x   := x_OR1_2_O_in_OR1
        self.q   := q_OR1_2_O
        self.x_b := x_O_2_OR1_in_O

        Therefore, the computation is as follows:
        x_O_2_OR1_in_OR1 = -x_OR1_2_O_in_OR1
        x_O_2_OR1_in_O = q_OR1_2_O * x_O_2_OR1_in_OR1
        self.x_b = self.q * -self.x
        """
        return q_mult_vec(self.q, -self.x)

    @property
    def R(self):
        return quaternion.as_rotation_matrix(self.q)

    @property
    def euler(self):
        return tforms.transform_orientation(self.q, "quat", "euler")

    @property
    def rotation(self):
        nom_org = Origin(np.zeros((3,)), np.eye(3))
        return Rotation(self.R, origin=nom_org)

    @property
    def translation(self):
        nom_org = Origin(np.zeros((3,)), np.eye(3))
        return Translation(self.x, origin=nom_org)

    @property
    def transform(self):
        nom_org = Origin(np.zeros((3,)), np.eye(3))
        return Transform(self.rotation, self.translation)

    @property
    def matrix(self):
        return np.block(
            [[self.R, self.x_b[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]
        )

    def translate(self, other: np.ndarray):
        return Origin(self.x + other, self.q)

    def rotate(self, other):
        """
        Rotate the origin via the following:

        self.q := q_OR1_2_O
        other  := q_O_2_ON

        The operation is therefore:
        q_OR1_2_ON = q_O_2_ON * q_OR1_2_O

        Which is the same as:
        other * self.q
        """
        if isinstance(other, np.quaternion):
            pass
        elif isinstance(other, np.ndarray) and other.shape == (3, 3):
            other = quaternion.from_rotation_matrix(other)
        else:
            raise NotImplementedError(type(other))
        return Origin(q_mult_vec(other, self.x), other * self.q)

    def inv(self):
        """Invert the origin

        The purpose of this is to be able to take a point that
        was in the origin system and bring it back to the
        "global" reference system

        The elements are as follows:
        self.x   := x_OR1_2_O_in_OR1
        self.x_b := x_O_2_OR1_in_O
        self.q   := q_OR1_2_O
        x_new    := x_O_2_OR1_in_O
        q_new    := q_O_2_OR1

        Thus, the computation is:
        x_new = self.x_b
        q_new = self.q.conjugate()
        """
        return Origin(self.x_b, self.q.conjugate())

    def change_points_origin(self, points, origin_new):
        """
        Change the origin of the points as follows:

        Assumption: origins share the same reference origin

        points       := x_O_to_pts_in_O
        self.x       := x_OR1_to_O_in_OR1
        self.q       := q_OR1_to_O
        origin_new.x := x_OR1_to_ON_in_OR1
        origin_new.q := q_OR1_to_ON

        To compute (new_data := x_ON_to_pts_in_ON), do the following
        q_O_to_ON = q_OR1_to_ON * q_OR1_to_O.conjugate()
        x_ON_to_O_in_OR1 = x_OR1_to_O_in_OR1 - x_OR1_to_ON_in_OR1
        x_ON_to_O_in_O = q_mult_vec(q_OR1_to_O, x_ON_to_O_in_OR1)
        x_ON_to_pts_in_O = x_O_to_pts_in_O + x_ON_to_O_in_O
        x_ON_to_pts_in_ON = q_mult_vec(q_O_to_ON, x_ON_to_pts_in_O)
        """
        # -- pull off details
        x_O_to_pts_in_O = points
        x_OR1_to_O_in_OR1 = self.x
        q_OR1_to_O = self.q
        x_OR1_to_ON_in_OR1 = origin_new.x
        q_OR1_to_ON = origin_new.q

        # -- perform calculations
        q_O_to_ON = q_OR1_to_ON * q_OR1_to_O.conjugate()
        x_ON_to_O_in_OR1 = x_OR1_to_O_in_OR1 - x_OR1_to_ON_in_OR1
        x_ON_to_O_in_O = q_mult_vec(q_OR1_to_O, x_ON_to_O_in_OR1)
        x_ON_to_pts_in_O = x_O_to_pts_in_O + x_ON_to_O_in_O
        x_ON_to_pts_in_ON = q_mult_vec(q_O_to_ON, x_ON_to_pts_in_O)
        return x_ON_to_pts_in_ON

    def __matmul__(self, other):
        """Called when doing self @ other

        This is called with the intention of putting a point into
        a new origin system.

        This works as follows for different classes

        self.q := q_OR1_2_O
        self.x := x_OR1_2_O_in_OR1

        ---- if input is numpy array
            other: either [n, 3] or [3,]
            other  := x_OR1_2_pts_in_OR1 --> ASSUMES POINTS IN OR1

            Thus, the computation is:
            x_O_2_OR1_in_OR1 = -x_OR1_2_O_in_OR1   --> -self.x
            x_O_2_pts_in_OR1 = x_OR1_2_pts_in_OR1 + x_O_2_OR1_in_OR1  --> other - self.x
            x_O_2_pts_in_O = q_OR1_2_O * x_O_2_pts_in_OR1

            Which becomes:
            v_out = self.q * (other - self.x)
        """
        if isinstance(other, np.ndarray):
            if ((len(other.shape) == 1) or (other.shape[0] == 3)) or (
                (len(other.shape) == 2) and (other.shape[1] == 3)
            ):
                x_d = (
                    (other - self.x)
                    if (len(other.shape) == 1 or other.shape[0] == 3)
                    else (other - self.x[:, None].T)
                )
                return q_mult_vec(self.q, x_d)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def format_as_string(self):
        return (
            f"origin {self.x[0]} {self.x[1]} {self.x[2]} "
            f"{self.q.w} {self.q.x} {self.q.y} {self.q.z}"
        )


R_stan_to_cam = StandardCoordinates.get_conversion_matrix(CameraCoordinates)
q_stan_to_cam = quaternion.from_rotation_matrix(R_stan_to_cam)
NominalOriginStandard = Origin(np.zeros((3,)), np.quaternion(1))
NominalOriginCamera = Origin(np.zeros((3,)), q_stan_to_cam)


class Rotation:
    def __init__(self, q, origin=NominalOriginStandard, n_prec=8):
        """Assumes R is something like R_global_2_local"""
        if isinstance(q, np.quaternion):
            pass
        elif isinstance(q, np.ndarray) and q.shape == (3, 3):
            q = tforms.transform_orientation(q, "dcm", "quat")
        else:
            raise ValueError(f"{type(q)} must be quaternion or 3x3")
        y = np.empty_like(q.vec)
        self.q = np.quaternion(np.round(q.w, n_prec), *fastround(q.vec, n_prec, y))
        assert isinstance(origin, Origin), origin
        self.origin = origin

    @property
    def R(self):
        return quaternion.as_rotation_matrix(self.q)

    @property
    def euler(self):
        return tforms.transform_orientation(self.q, "quat", "euler")

    @property
    def T(self):
        return Rotation(self.q.conjugate(), self.origin)

    @property
    def T_(self):
        self.q = self.q.conjugate()
        return self

    @property
    def q_by_origin(self):
        return self.q * self.origin.q

    @property
    def forward_vector(self):
        return tforms.transform_orientation(self.q_by_origin, "quat", "dcm")[:, 0]

    @property
    def left_vector(self):
        return tforms.transform_orientation(self.q_by_origin, "quat", "dcm")[:, 1]

    @property
    def up_vector(self):
        return tforms.transform_orientation(self.q_by_origin, "quat", "dcm")[:, 2]

    @property
    def yaw(self):
        return tforms.transform_orientation(self.q_by_origin, "quat", "euler")[2]

    def __str__(self):
        return f"Rotation with quaternion: {self.q}"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return 3

    def allclose(self, other):
        if isinstance(other, np.quaternion):
            q1 = self.q
            q2 = other
        elif isinstance(other, Rotation):
            if self.origin != other.origin:
                q1 = self.q_by_origin
                q2 = other.q_by_origin
            else:
                q1 = self.q
                q2 = other.q
        else:
            raise NotImplementedError(type(other))
        if hash(q1.tobytes()) == hash(q2.tobytes()):
            return True
        else:
            return quaternion.allclose(q1, q2)

    def __eq__(self, other):
        if isinstance(other, np.quaternion):
            q1 = self.q
            q2 = other
        elif isinstance(other, Rotation):
            if self.origin != other.origin:
                q1 = self.q_by_origin
                q2 = other.q_by_origin
            else:
                q1 = self.q
                q2 = other.q
        else:
            raise NotImplementedError(type(other))
        if hash(q1.tobytes()) == hash(q2.tobytes()):
            return True
        else:
            return quaternion.allclose(q1, q2)

    def __matmul__(self, other):
        """Called when doing self @ other"""
        if isinstance(other, Rotation):
            if self.origin != other.origin:
                other.change_origin(self.origin)
            return Rotation(self.q * other.q, origin=self.origin)
        elif isinstance(other, Translation):
            if self.origin != other.origin:
                other.change_origin(self.origin)
            return Translation(q_mult_vec(self.q, other.vector), origin=self.origin)
        elif isinstance(other, np.ndarray):
            if len(other.shape) == 1 or other.shape[0] == 3:
                return q_mult_vec(self.q, other)
            elif len(other.shape) == 2 and other.shape[1] == 3:
                return quaternion.rotate_vectors(self.q, other)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def copy(self):
        return Rotation(self.q.copy(), self.origin)

    def change_origin(self, origin_new):
        """
        self.q        := q_O_2_S
        self.origin.q := q_OR1_2_O
        origin_new.q  := q_OR2_2_ON

        where:
            -- S   := self (this object's body frame)
            -- O   := origin
            -- OR  := origin reference (where x, q are relative to)
            -- ON  := origin new
            -- ONR := origin new reference

        assumption: both O and ON have the same reference!! So for now this
        means that OR == ONR. Eventually this could change by introducing the
        reference chain of command (RefChoc)

        Thus, the calculation performed is akin to:
        q_OR1_2_OR2 --> must calculate this using the RefChoc
        q_OR1_2_ON = q_OR2_2_ON * q_OR1_2_OR2
        q_ON_2_O   = q_OR1_2_O * q_OR1_2_ON.conjugate()
        q_ON_2_S   = q_O_2_S * q_ON_2_O

        And since assumption states OR == ONR, this simplies to
        q_ON_2_O = q_OR1_2_O * q_OR2_2_ON.conjugate()
        q_ON_2_S = q_O_2_S * q_ON_2_O

        And in one line:
        q_ON_2_S = q_O_2_S * q_OR1_2_O * q_OR2_2_ON.conjugate()

        Thus in regular notation, this becomes:
        self.q = self.q * self.origin.q * origin_new.q.conjugate()
        """
        if origin_new != self.origin:
            q = self.q * self.origin.q * origin_new.q.conjugate()
            self.__init__(q, origin=origin_new)

    def format_as_string(self):
        return f"rotation {self.q.w} {self.q.x} {self.q.y} {self.q.z} {self.origin.format_as_string()}"


class Translation:
    TYPE = "Translation"

    def __init__(self, *args, origin=NominalOriginStandard, n_prec=8):
        if len(args) == 1:
            assert len(args[0]) == 3, f"Actual length: {len(args[0])}, {args[0]}"
            self.x, self.y, self.z = args[0]
        elif len(args) == 2:
            assert isinstance(args[1], Origin), args[1]
            self.x, self.y, self.z = args[0]
            origin = args[1]
        elif len(args) == 3:
            self.x, self.y, self.z = args
        elif len(args) == 4:
            assert isinstance(args[3], Origin), args[3]
            self.x, self.y, self.z, origin = args
        else:
            raise NotImplementedError(args)
        self.x = np.round(self.x, n_prec)
        self.y = np.round(self.y, n_prec)
        self.z = np.round(self.z, n_prec)
        self.origin = origin

    def copy(self):
        return Translation(self.x, self.y, self.z, origin=self.origin)

    @property
    def vector(self):
        return np.array([self.x, self.y, self.z])

    @vector.setter
    def vector(self, vector):
        self.x, self.y, self.z = vector

    @property
    def vector_global(self):
        return self.origin.inv() @ self.vector

    @property
    def range(self):
        return np.linalg.norm(self.vector)

    @property
    def finite(self):
        return np.all(np.isfinite(self.vector))

    def __str__(self):
        return f"{self.TYPE} at [{self.x}, {self.y}, {self.z}] with {self.origin}"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return 3

    def __iter__(self):
        return iter(self.vector)

    def __getitem__(self, key):
        return self.vector[key]

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        elif key == 2:
            self.z = value
        else:
            raise KeyError

    def allclose(self, other):
        if isinstance(other, np.ndarray):
            v1 = self.vector
            v2 = other
        else:
            if self.origin != other.origin:
                v1 = self.origin.inv() @ self.vector
                v2 = other.origin.inv() @ other.vector
            else:
                v1 = self.vector
                v2 = other.vector
        if hash(v1.tobytes()) == hash(v2.tobytes()):
            return True
        else:
            return np.allclose(v1, v2)

    def __eq__(self, other):
        if isinstance(other, np.ndarray):
            v1 = self.vector
            v2 = other
        else:
            if self.origin != other.origin:
                v1 = self.origin.inv() @ self.vector
                v2 = other.origin.inv() @ other.vector
            else:
                v1 = self.vector
                v2 = other.vector
        if hash(v1.tobytes()) == hash(v2.tobytes()):
            return True
        else:
            return np.all(v1 == v2)

    def __neg__(self):
        """Called when doing -self"""
        return Translation(-self.x, -self.y, -self.z, origin=self.origin)

    def __add__(self, other):
        """Called when doing self + other"""
        if isinstance(other, Translation):
            if self.origin != other.origin:
                other.change_origin(self.origin)
            return Translation(self.vector + other.vector, origin=self.origin)
        elif isinstance(other, (np.ndarray, int, float)):
            if isinstance(other, (int, float)) or len(other.shape) == 1:
                return Translation(other + self.vector, origin=self.origin)
            elif len(other.shape) == 2:
                if other.shape[0] == 3:
                    return other + self.vector
                elif other.shape[1] == 3:
                    return other + self.vector[:, None].T
                else:
                    raise NotImplementedError
            else:
                raise RuntimeError
        else:
            raise NotImplementedError(type(other))

    def __sub__(self, other):
        """Called when doing self - other"""
        return self + -other

    def __mul__(self, other):
        """Called when doing self * other"""
        if isinstance(other, int) or isinstance(other, float):
            x = self.vector * other
        elif isinstance(other, Translation):
            x = self.vector * other.vector
        else:
            raise NotImplementedError(type(other))
        return Translation(x, origin=self.origin)

    def __truediv__(self, other):
        """Called when doing self / other"""
        if isinstance(other, int) or isinstance(other, float):
            x = self.vector / other
        elif isinstance(other, Translation):
            x = self.vector / other.vector
        else:
            raise NotImplementedError(type(other))
        return Translation(x, origin=self.origin)

    def __matmul__(self, other):
        """called when doing self @ other"""
        if isinstance(other, Rotation):
            if self.origin != other.origin:
                other.change_origin(self.origin)
            x = self.vector @ other.R
        elif isinstance(other, np.ndarray):
            assert other.shape == (3, 3)
            x = other @ self.vector
        else:
            raise NotImplementedError(type(other))
        return Translation(x, origin=self.origin)

    def change_origin(self, origin_new):
        """Change origin of the translation

        Need to perform the following operations:

        self.x        := x_O_2_S_in_O  (in O's frame)
        self.origin.x := x_OR1_2_O_in_OR1  (in OR1's frame)
        self.origin.q := q_OR1_2_O
        origin_new.x  := x_OR2_2_ON_in_OR2  (in OR2's frame)
        origin_new.q  := q_OR2_2_ON

        where:
            -- S   := self (this object's body frame)
            -- O   := origin
            -- OR1 := origin reference (where x, q are relative to)
            -- ON  := origin new
            -- OR2 := origin new reference

        assumption: both O and ON have the same reference!! So for now this
        means that OR1 == OR2. Eventually this could change by introducing the
        reference chain of command (RefChoc)

        Thus, the calculation performed is akin to:
        q_O_2_OR2 = q_OR1_2_O.conjugate() --> this is an assumption!!!!
        x_O_2_S_in_OR2 = q_O_2_OR2 * x_O_2_S_in_O
        x_ON_2_OR2_in_OR2 = -x_OR2_2_ON_in_OR2
        x_OR2_2_O_in_OR2 = x_OR1_2_O_in_OR1  --> this is an assumption!!!!
        x_ON_2_O_in_OR2 = x_OR2_2_O_in_OR2 + x_ON_2_OR2_in_OR2
        x_ON_2_S_in_OR2 = x_O_2_S_in_OR2 + x_ON_2_O_in_OR2
        x_ON_2_S_in_ON = q_OR2_2_ON * x_ON_2_S_in_OR2

        So, putting this all together, we get:
        x_ON_2_S_in_OR2 = origin_new.q *((q_OR1_2_O.conjugate() * x_O_2_S_in_O) + x_OR1_2_O_in_OR1 + -x_OR2_2_ON_in_OR2)

        Which becomes:
        x_out = (self.origin.q.conjugate() * self.x) + self.origin.x - origin_new.x
        """
        if origin_new != self.origin:
            # x_global = self.origin.R.T @ self.vector + self.origin.x
            # x_new = origin_new.R @ (x_global - origin_new.x)
            x_out = q_mult_vec(
                origin_new.q,
                q_mult_vec(self.origin.q.conjugate(), self.vector)
                + self.origin.x
                - origin_new.x,
            )
            self.__init__(x_out, origin=origin_new)

    def sqrt(self):
        return np.sqrt(self.vector)

    def norm(self):
        return np.linalg.norm(self.vector)

    def distance(self, other):
        if isinstance(other, Translation):
            dist = np.linalg.norm(self.vector_global - other.vector_global)
        elif isinstance(other, (np.ndarray, list)):
            if isinstance(other, list):
                other = np.asarray(other)
            dist = np.linalg.norm(self.vector - other)
        else:
            try:
                dist = np.linalg.norm(self.vector - other.translation.vector)
            except Exception as e:
                raise NotImplementedError(f"Type {type(other)} not able for dist calc")
        return dist

    def format_as_string(self):
        return f"translation {self.vector[0]} {self.vector[1]} {self.vector[2]} {self.origin.format_as_string()}"


class Vector(Translation):
    TYPE = "Vector"
    """
    Unlike translation, a vector is NOT related to the origin's translation
    so only the rotation aspect is applied
    """

    def __eq__(self, other):
        if isinstance(other, np.ndarray):
            v1 = self.vector
            v2 = other
        else:
            v1 = self.origin.R.T @ self.vector
            v2 = other.origin.R.T @ other.vector
        if hash(v1.tobytes()) == hash(v2.tobytes()):
            return True
        else:
            return np.allclose(v1, v2)

    @property
    def vector_global(self):
        return self.origin.R.T @ self.vector

    def change_origin(self, origin_new):
        if origin_new != self.origin:
            v_new = origin_new.R @ (self.origin.R.T @ self.vector)
            self.__init__(v_new, origin=origin_new)

    def format_as_string(self):
        return f"vector {self.vector[0]} {self.vector[1]} {self.vector[2]} {self.origin.format_as_string()}"


class Transform:
    def __init__(self, rotation: Rotation, translation: Translation):
        self.rotation = rotation
        self.translation = translation
        assert self.rotation.origin.allclose(self.translation.origin)
        self.origin = self.rotation.origin

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Transform with: {self.rotation}, {self.translation}"

    def copy(self):
        return Transform(self.rotation.copy(), self.translation.copy())

    @property
    def rotation(self):
        """Rotation from a to b frame"""
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        assert isinstance(rotation, Rotation), f"{type(rotation)}"
        self._rotation = rotation

    @property
    def translation(self):
        """Translation from a to b in a's frame"""
        return self._translation

    @translation.setter
    def translation(self, translation):
        assert isinstance(translation, Translation)
        self._translation = translation

    @property
    def translation_b(self):
        """Translation from a to b in b's frame"""
        return self.rotation @ self.translation

    @translation_b.setter
    def translation_b(self, translation_b):
        assert isinstance(translation_b, Translation)
        self.translation = self.rotation.R.T @ translation_b

    @property
    def T(self):
        return Transform(self.rotation.T, self.rotation @ -self.translation)

    @property
    def matrix(self):
        return np.block(
            [
                [self.rotation.R, self.translation_b.vector[:, None]],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]
        )

    @property
    def inverse_matrix(self):
        return self.T.matrix

    def __eq__(self, other):
        return (self.rotation == other.rotation) and (
            self.translation == other.translation
        )

    def __matmul__(self, other):
        """called when doing self @ other"""
        if isinstance(other, np.ndarray):
            if len(other.shape) == 1 or other.shape[0] == 3:
                return self.rotation @ (-self.translation + other)
            elif len(other.shape) == 2 and other.shape[1] == 3:
                return self.rotation @ (-self.translation + other)
            else:
                raise NotImplementedError
        else:
            if isinstance(other, Translation):
                return self.rotation @ (-self.translation + other)
            elif isinstance(other, Transform):
                R = self.rotation @ other.rotation
                Tr = self.translation @ other.rotation + other.translation
                return Transform(R, Tr)
            elif isinstance(other, Origin):
                raise
                # origin = other.change_coordinates(self.coordinates)
                # R = Rotation(self.rotation @ origin.R, origin)
                # Tr = Translation(self.translation @ origin.R + origin.x, origin)
                # return Transform(R, Tr)
            else:
                raise NotImplementedError(type(other))

    def as_origin(self):
        return Origin(self.translation.vector, self.rotation.q)

    def change_origin(self, origin_new):
        self.rotation.change_origin(origin_new)
        self.translation.change_origin(origin_new)

    def format_as_string(self):
        return f"transform {self.rotation.format_as_string()} {self.translation.format_as_string()}"


class PointCloud:
    def __init__(self, points, origin):
        self.points = points
        self.origin = origin
        self.coordinates = origin.coordinates
