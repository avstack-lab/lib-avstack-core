class TransformManager:
    pass


class ReferenceFrameDecoder:
    pass


class ReferenceFrameEncoder:
    pass


class ReferenceFrame:
    def __init__(self, name: str, time: float):
        self.name = name
        self.time = time


WorldFrame = ReferenceFrame("world", None)


class FrameTransform:
    def __init__(
        self, to_frame: str, from_frame: str, time: float, transform: "Transform"
    ):
        pass


class Transform:
    def __init__(self, x, q, v):
        self.x = x
        self.q = q
        self.v = v
