import json
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from .frame import FrameTransform, ReferenceFrame, TransformManager

from avstack.exceptions import FrameEquivalenceError

from . import conversions
from .frame import ReferenceFrameDecoder, WorldFrame


# =============================================
# ENCODING/DECODING
# =============================================

#####################
# ENCODERS
#####################


class PoseEncoder(json.JSONEncoder):
    def default(self, o):
        p_dict = {
            "position": o.position.encode(),
            "attitude": o.attitude.encode(),
        }
        return {type(o).__name__.lower(): p_dict}


class VectorEncoder(json.JSONEncoder):
    def default(self, o):
        v_dict = {
            "x": o.x.tolist(),
            "reference": o.reference.encode(),
        }
        return {type(o).__name__.lower(): v_dict}


class RotationEncoder(json.JSONEncoder):
    def default(self, o):
        q_dict = {
            "qw": o.q.w,
            "qv": o.q.vec.tolist(),
            "reference": o.reference.encode(),
        }
        return {type(o).__name__.lower(): q_dict}


#####################
# DECODERS
#####################


class VectorDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "vector" in json_object:
            json_object = json_object["vector"]
            reference = json.loads(json_object["reference"], cls=ReferenceFrameDecoder)
            return Vector(
                x=np.array(json_object["x"]),
                reference=reference,
            )
        else:
            return json_object


class RotationDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "rotation" in json_object:
            json_object = json_object["rotation"]
            reference = json.loads(json_object["reference"], cls=ReferenceFrameDecoder)
            return Rotation(
                q=np.quaternion(json_object["qw"], *json_object["qv"]),
                reference=reference,
            )
        else:
            return json_object


class PoseDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "pose" in json_object:
            json_object = json_object["pose"]
            return Pose(
                json.loads(json_object["position"], cls=VectorDecoder),
                json.loads(json_object["attitude"], cls=RotationDecoder),
            )
        else:
            return json_object


# =============================================
# CLASSES
# =============================================


class Vector:
    def __init__(self, x: np.ndarray, reference: "ReferenceFrame" = WorldFrame):
        self.x = np.asarray(x)
        self.reference = reference

    def copy(self):
        return Vector(self.x.copy(), self.frame)

    def encode(self):
        return json.dumps(self, cls=VectorEncoder)

    def change_reference(
        self,
        reference: "ReferenceFrame",
        tm: "TransformManager",
        T: "FrameTransform" = None,
    ):
        if not T:
            T = tm.get_transform(from_frame=self.reference, to_frame=reference)
        raise NotImplementedError


class Rotation:
    def __init__(self, q: np.array, reference: "ReferenceFrame" = WorldFrame):
        self.q = q
        self.reference = reference

    @property
    def R(self):
        return conversions.transform_orientation(self.q, "quat", "dcm")

    def copy(self):
        return Rotation(self.q.copy(), self.frame)

    def encode(self):
        return json.dumps(self, cls=RotationEncoder)

    def change_reference(
        self,
        frame: "ReferenceFrame",
        tm: "TransformManager",
        T: "FrameTransform" = None,
    ):
        if not T:
            T = tm.get_transform(from_frame=self.frame, to_frame=frame)
        raise NotImplementedError


class Pose:
    def __init__(self, position: "Vector", attitude: "Rotation"):
        self.position = position
        self.attitude = attitude
        if position.reference != attitude.reference:
            raise FrameEquivalenceError(position.reference, attitude.reference)

    @property
    def reference(self):
        return self.position.reference

    def copy(self):
        return Pose(self.position.copy(), self.attitude.copy())

    def encode(self):
        return json.dumps(self, cls=PoseEncoder)

    def change_reference(self, reference: "ReferenceFrame", tm: "TransformManager"):
        T = tm.get_transform(
            from_frame=self.reference, to_frame=reference
        )  # for efficiency
        self.position.change_reference(reference, tm, T=T)
        self.attitude.change_reference(reference, tm, T=T)


class PointMatrix:
    def __init__(self, x: np.ndarray, reference: "ReferenceFrame") -> None:
        self.x = x
        self.reference = reference

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        if len(x.shape) == 1:
            x = x[:, None]
        self._x = x

    @property
    def shape(self):
        return self.x.shape

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, indices):
        return self.x[indices]

    def copy(self):
        return self.__class__(self.x.copy(), self.reference)


class PointMatrix3D(PointMatrix):
    pass


class PointMatrix2D(PointMatrix):
    pass
