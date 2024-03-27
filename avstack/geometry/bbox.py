import json
from typing import TYPE_CHECKING

from numba import jit


if TYPE_CHECKING:
    from .primitives import Pose
    from .frame import ReferenceFrame, TransformManager

import numpy as np

from avstack.exceptions import FrameEquivalenceError

from .frame import ReferenceFrameDecoder
from .primitives import PointMatrix2D, PointMatrix3D, PoseDecoder
from .utils import _q_mult_vec


# =============================================
# ENCODING/DECODING
# =============================================

#####################
# ENCODERS
#####################


class BoxSizeEncoder(json.JSONEncoder):
    def default(self, o):
        box_size_dict = {
            "height": o.height,
            "width": o.width,
            "length": o.length,
        }
        return {"boxsize": box_size_dict}


class BoundingBoxEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, BoundingBox2Dxyxy):
            box_dict = {
                "xyxy": o.xyxy.tolist(),
                "reference": o.reference.encode(),
                "obj_class": o.obj_class,
                "ID": o.ID,
            }
            return {"boundingbox2dxyxy": box_dict}
        elif isinstance(o, BoundingBox2Dcwh):
            box_dict = {
                "center": o.xyxy.tolist(),
                "width": o.width,
                "height": o.height,
                "reference": o.reference.encode(),
                "obj_class": o.obj_class,
                "ID": o.ID,
            }
            return {"boundingbox2dcwh": box_dict}
        elif isinstance(o, BoundingBox3D):
            box_dict = {
                "pose": o.pose.encode(),
                "box": o.box.encode(),
                "obj_class": o.obj_class,
                "ID": o.ID,
            }
            return {"boundingbox3d": box_dict}
        elif isinstance(o, SegMask2D):
            seg_dict = {
                "shape": o.sparse_mask.shape,
                "rows": o.sparse_mask.row.tolist(),
                "cols": o.sparse_mask.col.tolist(),
                "data": o.sparse_mask.data.tolist(),
                "calibration": o.calibration.encode(),
            }
            return {"segmask2d": seg_dict}
        else:
            raise NotImplementedError(f"{type(0)}, {o}")


#####################
# DECODERS
#####################


class BoxSizeDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "boxsize" in json_object:
            json_object = json_object["boxsize"]
            return BoxSize(
                json_object["height"], json_object["width"], json_object["length"]
            )
        else:
            return json_object


class BoundingBoxDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        from ..calibration import CalibrationDecoder

        if "boundingbox2dxyxy" in json_object:
            json_object = json_object["boundingbox2dxyxy"]
            reference = json.loads(json_object["reference"], cls=ReferenceFrameDecoder)
            return BoundingBox2D(
                xyxy=json_object["xyxy"],
                reference=reference,
                ID=json_object["ID"],
                obj_class=json_object["obj_class"],
            )
        elif "boundingbox2dcwh" in json_object:
            json_object = json_object["boundingbox2dcwh"]
            reference = json.loads(json_object["reference"], cls=ReferenceFrameDecoder)
            return BoundingBox2D(
                center=json_object["center"],
                width=json_object["width"],
                height=json_object["height"],
                reference=reference,
                ID=json_object["ID"],
                obj_class=json_object["obj_class"],
            )
        elif "boundingbox3d" in json_object:
            json_object = json_object["boundingbox3d"]
            pose = json.loads(json_object["pose"], cls=PoseDecoder)
            box = json.loads(json_object["box"], cls=BoxSizeDecoder)
            return BoundingBox3D(
                pose=pose,
                box=box,
                obj_class=json_object["obj_class"],
                ID=json_object["ID"],
            )
        elif "segmask2d" in json_object:
            json_object = json_object["segmask2d"]
            calibration = json.loads(json_object["calibration"], cls=CalibrationDecoder)
            data = np.array(json_object["data"])
            rows = np.array(json_object["rows"])
            cols = np.array(json_object["cols"])
            mask = sparse.coo_matrix(
                (data, (rows, cols)), shape=json_object["shape"]
            ).toarray()
            return SegMask2D(mask=mask, calibration=calibration)
        else:
            return json_object


# =============================================
# CLASSES
# =============================================


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

    def encode(self):
        return json.dumps(self, cls=BoxSizeEncoder)


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

    def encode(self):
        return json.dumps(self, cls=BoundingBoxEncoder)


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

    def encode(self):
        return json.dumps(self, cls=BoundingBoxEncoder)


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

    def encode(self):
        return json.dumps(self, cls=BoundingBoxEncoder)

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
