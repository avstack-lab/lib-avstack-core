import json
import sys

import numpy as np

from avstack.geometry import (
    FrameTransform,
    ReferenceFrame,
    Transform,
    TransformManager,
    conversions,
)
from avstack.objects import ObjectStateDecoder


sys.path.append("tests/")
from utilities import camera_frame, camera_tform, get_ego


def test_encode_decode_object():
    obj_1 = get_ego(seed=1)
    obj_2 = json.loads(obj_1.encode(), cls=ObjectStateDecoder)
    assert obj_1.allclose(obj_2)


def test_change_reference():
    tm = TransformManager()

    # -- add world to new frame
    t_world_to_new = Transform(
        x=np.array([1, 2, 3]),
        q=conversions.transform_orientation([1, -1, 0.1], "euler", "quat"),
    )
    T_world_to_new = FrameTransform(
        from_frame="world", to_frame="newframe", transform=t_world_to_new
    )
    tm.add_transform(T_world_to_new)
    O_new = ReferenceFrame(name="newframe", stamp=None)

    # -- add world to camera frame
    tm.add_transform(camera_tform)

    # -- transform the object
    obj1 = get_ego(seed=1, reference=camera_frame)
    obj1_copy = get_ego(seed=1, reference=camera_frame)
    obj1.change_reference(O_new, tm=tm)
    obj1.change_reference(camera_frame, tm=tm)
    assert obj1.box.allclose(obj1_copy.box)


# def test_object_as_reference():
#     obj1 = get_ego(seed=1)
#     obj_ref = obj1.as_reference()
#     assert np.allclose(obj_ref.x, obj1.position.x)
#     assert np.allclose(obj_ref.v, obj1.velocity.x)


# def test_object_transform_reference():
#     obj1 = get_ego(seed=1)
#     obj1.attitude = Rotation(np.quaternion(1), obj1.attitude.reference)
#     obj2 = get_ego(seed=2)
#     obj1.attitude = Rotation(np.quaternion(1), obj1.attitude.reference)
#     obj2_in_1 = obj2.change_reference(obj1)
#     assert np.allclose(obj2_in_1.position.x, obj2.position.x - obj1.position.x)
#     assert np.allclose(obj2_in_1.velocity.x, obj2.velocity.x - obj1.velocity.x)
