from dataclasses import dataclass


class StampDecoder:
    pass


@dataclass
class Stamp:
    stamp: float
    frame: int = None
