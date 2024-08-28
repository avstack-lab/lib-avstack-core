import sys

from avstack.geometry import GlobalOrigin3D
from avstack.modules import perception
from avstack.sensors import LidarData


sys.path.append("tests/")
from utilities import get_test_sensor_data


(
    obj,
    box_calib,
    lidar_calib,
    pc,
    camera_calib,
    img,
    radar_calib,
    rad,
    box_2d,
    box_3d,
) = get_test_sensor_data()
pc_bev = LidarData(pc.timestamp, pc.frame, pc.data[:, [1, 2]], lidar_calib, 100)


def test_percep_base():
    try:
        percep = perception.object2dbev.Lidar2dCentroidDetector()
        output = percep(pc_bev, GlobalOrigin3D, "lidar-detector")
    except (NameError, ModuleNotFoundError):
        pass
