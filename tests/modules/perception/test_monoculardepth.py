import sys

from avstack.modules.perception.monoculardepth import MidasDepthEstimator
from avstack.sensors import DepthImageData


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


def test_midas_inference():
    dataset = None
    model = "dpt_beit_base_384"
    try:
        depthestim = MidasDepthEstimator(model=model)
    except ModuleNotFoundError:
        print("Cannot run mmdet test without the module")
    except FileNotFoundError:
        print(f"Cannot find ({model}, {dataset}) model file for mmdet3d test")
    else:
        depth_out = depthestim(img)
        assert isinstance(depth_out, DepthImageData)
