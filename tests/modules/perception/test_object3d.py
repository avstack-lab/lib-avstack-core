import sys

from avstack.datastructs import DataContainer
from avstack.modules import perception


sys.path.append("tests/")
from utilities import get_object_global, get_test_sensor_data


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
frame = 0


def test_passthrough_perception_box():
    frame = timestamp = 0
    objs = [get_object_global(seed=i) for i in range(4)]
    data = DataContainer(
        frame=frame, timestamp=timestamp, data=objs, source_identifier="sensor-1"
    )
    percep = perception.object3d.Passthrough3DObjectDetector()
    detections = percep(data, frame=frame)
    assert len(detections) == len(data)


def test_passthrough_perception_centroid():
    frame = timestamp = 0
    objs = [get_object_global(seed=i).position for i in range(4)]
    data = DataContainer(
        frame=frame, timestamp=timestamp, data=objs, source_identifier="sensor-1"
    )
    percep = perception.object3d.Passthrough3DObjectDetector()
    detections = percep(data, frame=frame)
    assert len(detections) == len(data)


class LidarMeasurement:
    """To emulate the carla measurements"""

    def __init__(self, raw_data: memoryview) -> None:
        assert isinstance(raw_data, memoryview)
        self.raw_data = raw_data


def run_mmdet3d(datatype, model, dataset, as_memoryview=False):
    try:
        detector = perception.object3d.MMDetObjectDetector3D(
            model=model, dataset=dataset
        )
    except ModuleNotFoundError:
        print("Cannot run mmdet test without the module")
    except FileNotFoundError:
        print(f"Cannot find ({model}, {dataset}) model file for mmdet3d test")
    else:
        if datatype == "lidar":
            data = pc
            if as_memoryview:
                data.data = LidarMeasurement(memoryview(data.data.x))
        elif datatype == "image":
            data = img
        else:
            raise NotImplementedError(datatype)
        _ = detector(data, frame=frame)


# def test_mmdet_3d_pgd_kitti():
#     run_mmdet3d("image", "pgd", "kitti")


# def test_mmdet_3d_pillars_kitti():
#     run_mmdet3d("lidar", "pointpillars", "kitti")


# def test_mmdet_3d_pillars_kitti_memoryview():
#     run_mmdet3d("lidar", "pointpillars", "kitti", as_memoryview=True)
