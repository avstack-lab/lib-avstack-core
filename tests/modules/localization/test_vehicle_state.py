import sys

from avstack.environment.objects import VehicleState


sys.path.append("tests/")
from utilities import get_ego


def test_vehicle_state_init():
    VS = get_ego(1)
    assert isinstance(VS, VehicleState)


# def test_vehicle_state_convert():
#     VS_1 = get_ego(1)
#     VS_2 = get_ego(2)
#     VS_2_in_1 = VS_2.change_reference(VS_1, inplace=False)
#     v1 = VS_2_in_1.change_reference(GlobalOrigin3D, inplace=False).position
#     v2 = VS_2.change_reference(GlobalOrigin3D, inplace=False).position
#     assert v1.allclose(v2)


def test_vehicle_state_velocity_diff():
    pass
