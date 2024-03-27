import json
from dataclasses import dataclass


class StampDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "stamp" in json_object:
            json_object = json_object["stamp"]
            return Stamp(
                timestamp=float(json_object["stamp"]),
                frame=int(json_object["frame"]) if json_object["frame"] != "" else None,
            )
        else:
            return json_object


class StampEncoder(json.JSONEncoder):
    def default(self, o):
        v_dict = {
            "timestamp": o.timestamp,
            "frame": o.frame if o.frame is not None else -1,
        }
        return {"stamp": v_dict}


@dataclass
class Stamp:
    timestamp: float
    frame: int = None

    def encode(self):
        return json.dumps(self, cls=StampEncoder)
