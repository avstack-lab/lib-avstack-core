from typing import TYPE_CHECKING

from numba import jit


if TYPE_CHECKING:
    from .primitives import Pose
    from .frame import ReferenceFrame, TransformManager

import numpy as np

from avstack.exceptions import FrameEquivalenceError

from .primitives import PointMatrix2D, PointMatrix3D
from .utils import _q_mult_vec


class BoxSize:
    def __init__(self, height: float, width: float, length: float = None):
        self.height = float(height)
        self.width = float(width)
        self.length = float(length)

    @property
    def area(self):
        if self.length:
            raise ValueError("Cannot call area when length is not None")
        else:
            return self.height * self.width

    @property
    def volume(self):
        if not self.length:
            raise ValueError("Cannot call volume when length is None")
        else:
            return self.height * self.width * self.length


class BoundingBoxDecoder:
    pass


class BoundingBox2D:
    def __init__(
        self,
        xyxy: np.ndarray,
        reference: "ReferenceFrame",
        ID: int = None,
        obj_class: str = None,
    ):
        self.xyxy = xyxy
        self.reference = reference
        self.obj_class = obj_class
        self.ID = ID

    @property
    def center(self):
        return np.array(
            [(self.xyxy[2] + self.xyxy[0]) / 2, (self.xyxy[3] + self.xyxy[1]) / 2]
        )

    @property
    def xmin(self):
        return self.xyxy[0]

    @property
    def ymin(self):
        return self.xyxy[1]

    @property
    def xmax(self):
        return self.xyxy[2]

    @property
    def ymax(self):
        return self.xyxy[3]

    @property
    def corners(self):
        return PointMatrix2D(
            x=np.array(
                [
                    [self.xmin, self.ymin],
                    [self.xmin, self.ymax],
                    [self.xmax, self.ymax],
                    [self.xmax, self.ymin],
                ]
            ),
            reference=self.reference,
        )


class BoundingBox2Dcwh(BoundingBox2D):
    def __init__(
        self,
        center: np.ndarray,
        width: float,
        height: float,
        reference: "ReferenceFrame",
        ID: int = None,
        obj_class: str = None,
    ):
        xyxy = np.array(
            [
                center[0] - width / 2,
                center[1] - height / 2,
                center[0] + width / 2,
                center[1] + height / 2,
            ]
        )
        super().__init__(xyxy, reference, ID, obj_class)


class BoundingBox2Dxyxy(BoundingBox2D):
    def __init__(
        self,
        xyxy: np.ndarray,
        reference: "ReferenceFrame",
        ID: int = None,
        obj_class: str = None,
    ):
        super().__init__(xyxy, reference, ID, obj_class)


class BoundingBox3D:
    def __init__(
        self, pose: "Pose", box: "BoxSize", ID: int = None, obj_class: str = None
    ):
        self.pose = pose
        self.box = box
        self.ID = ID
        self.obj_class = obj_class

    @property
    def center(self):
        return self.pose.position.x.astype(float)

    @property
    def height(self):
        return self.box.height

    @property
    def width(self):
        return self.box.width

    @property
    def length(self):
        return self.box.length

    @property
    def corners(self):
        raise NotImplementedError

    @property
    def reference(self):
        return self.pose.reference

    @property
    def corners(self):
        """A wrapper around the jit compiled method"""
        qcon = self.pose.attitude.q.conjugate()
        qcs = qcon.w
        qcr = qcon.vec
        qcm = qcon.w**2 + qcon.x**2 + qcon.y**2 + qcon.z**2
        corners = _compute_box_3d_corners(
            self.pose.position.x,
            self.height,
            self.width,
            self.length,
            qcs,
            qcr,
            qcm,
            "center",
        )
        return PointMatrix3D(corners, self.reference)

    def IoU(self, other: "BoundingBox3D"):
        if self.reference != other.reference:
            raise FrameEquivalenceError(self.reference, other.reference)
        else:
            raise NotImplementedError

    def change_reference(self, reference: "ReferenceFrame", tm: "TransformManager"):
        self.pose.change_reference(reference, tm)


class SegMask2D:
    pass


@jit(nopython=True, parallel=True)
def _compute_box_3d_corners(
    t: np.ndarray,
    h: float,
    w: float,
    l: float,
    qcs: float,
    qcr: float,
    qcm: float,
    where_is_t: str,
):
    """Computes the 3D bounding box in the box's coordinate frame

    X - forward, Y - left, Z - up
    Corners start in upper top left and go counterclockwise on top plane
    then lower top left and go counterclockwise on bottom plane


          0-----3
         -     --
        -     - -
       -     -  -
      -     -   -
     -     -    -
    1-----2     7
    -     -    -
    -     -   -
    -     -  -
    -     - -
    -     --
    5-----6

    Starts with box in standard frame, then converts to reference

    Conversion process is as follows:

    box3d.q         := q_O_2_V --> reference to vehicle body frame
    box3d.t         := x_O_2_V_in_O -- > reference to vehicle body frame in reference frame
    corners_3d_base := x_V_2_pts_in_V --> vehicle body frame to corner points in body frame

    To get corners in the reference frame, do the following:
    x_V_2_pts_in_O = q_O_2_V.conjugate() * x_V_2_pts_in_V
    x_O_2_pts_in_O = x_V_2_pts_in_O + x_O_2_V_in_O

    This expands to:
    corners_out =  box3d.q.conjugate() * corners_3d_base + box3d.t

    """
    x_corners = np.array(
        [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2], dtype=np.float64
    )
    y_corners = np.array(
        [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2], dtype=np.float64
    )
    z_corners = np.array([h, h, h, h, 0, 0, 0, 0], dtype=np.float64)
    if where_is_t == "bottom":
        pass
    elif where_is_t == "center":
        z_corners -= h / 2
    else:
        raise NotImplementedError("Cannot handle this t position")

    # -- convert to the desired coordinates in the box's frame
    corners_3d_base = np.column_stack((x_corners, y_corners, z_corners))
    # NOTE: no need to do anything with the new reference here...corners will assume
    # the reference of the vehicle

    # -- apply box rotation quaternion, then translation
    corners_3d = _q_mult_vec(qcs, qcr, qcm, corners_3d_base) + t

    return corners_3d
