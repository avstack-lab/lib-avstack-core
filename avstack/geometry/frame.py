import json
from dataclasses import dataclass

import numpy as np

from avstack.time import Stamp, StampDecoder


# =============================================
# ENCODING/DECODING
# =============================================

#####################
# ENCODERS
#####################


class ReferenceFrameEncoder(json.JSONEncoder):
    def default(self, o):
        v_dict = {
            "name": o.name,
            "stamp": o.stamp.encode() if o.stamp is not None else "",
        }
        return {"referenceframe": v_dict}


class FrameTransformEncoder(json.JSONEncoder):
    def default(self, o):
        f_dict = {
            "from_frame": o.from_frame,
            "to_frame": o.to_frame,
            "transform": o.transform.encode(),
        }
        return {"frametransform": f_dict}


class TransformEncoder(json.JSONEncoder):
    def default(self, o):
        t_dict = {
            "x": o.x.tolist(),
            "qw": o.q.w,
            "qv": o.q.vec.tolist(),
            "v": o.v.tolist() if o.v is not None else None,
        }
        return {"transform": t_dict}


class TransformManagerEncoder(json.JSONEncoder):
    def default(self, o):
        raise


#####################
# DECODERS
#####################


class TransformManagerDecoder:
    pass


class ReferenceFrameDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "referenceframe" in json_object:
            json_object = json_object["referenceframe"]
            return ReferenceFrame(
                name=json_object["name"],
                stamp=json.loads(json_object["stamp"], cls=StampDecoder)
                if json_object["stamp"]
                else None,
            )
        else:
            return json_object


class FrameTransformDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "frametransform" in json_object:
            json_object = json_object["frametransform"]
            return FrameTransform(
                from_frame=json_object["from_frame"],
                to_frame=json_object["to_frame"],
                transform=json.loads(json_object["transform"], cls=TransformDecoder),
            )
        else:
            return json_object


class TransformDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "transform" in json_object:
            json_object = json_object["transform"]
            return Transform(
                x=np.array(json_object["x"]),
                q=np.quaternion(json_object["qw"], *json_object["qv"]),
                v=np.array(json_object["v"]) if json_object["v"] is not None else None,
            )
        else:
            return json_object


# =============================================
# CLASSES
# =============================================


class TransformManager:
    def __init__(self):
        self.transforms = [WorldFrame]

    def encode(self):
        return json.dumps(self, cls=TransformManagerEncoder)

    def add_transform(self, transform: "FrameTransform"):
        self.transforms.append(transform)

    def get_transform(
        self, from_frame: str, to_frame: str, time: float = None
    ) -> "FrameTransform":
        raise NotImplementedError


@dataclass(frozen=True)
class ReferenceFrame:
    name: str
    stamp: Stamp

    def encode(self):
        return json.dumps(self, cls=ReferenceFrameEncoder)


@dataclass(frozen=True)
class FrameTransform:
    from_frame: str
    to_frame: str
    transform: "Transform"

    def encode(self):
        return json.dumps(self, cls=FrameTransformEncoder)


@dataclass(frozen=True)
class Transform:
    x: np.ndarray
    q: np.ndarray
    v: np.ndarray = None

    def encode(self):
        return json.dumps(self, cls=TransformEncoder)


WorldFrame = ReferenceFrame("world", None)
