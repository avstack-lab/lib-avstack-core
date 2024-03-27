import numpy as np

from avstack.geometry import (
    BoundingBox2Dcwh,
    BoundingBox2Dxyxy,
    BoundingBox3D,
    BoxSize,
    Pose,
    Rotation,
    Vector,
    WorldFrame,
)


def test_init_box2d_cwh():
    center = np.array([1, 2, 3, 4])
    width, height = 100, 200
    reference = WorldFrame
    box = BoundingBox2Dcwh(center, width, height, reference)


def test_init_box2d_xyxy():
    xyxy = np.array([1, 2, 3, 4])
    frame = WorldFrame
    box = BoundingBox2Dxyxy(xyxy, frame)


def test_init_box3d():
    position = Vector(np.random.randn(3), reference=WorldFrame)
    attitude = Rotation(np.array([1, 0, 0, 0]), reference=WorldFrame)
    pose = Pose(position, attitude)
    size = BoxSize(1, 2, 3)
    box = BoundingBox3D(pose, size)
