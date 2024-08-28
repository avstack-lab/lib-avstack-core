import logging
import sys

from avstack.modules import perception


sys.path.append("tests/")
from utilities import get_test_sensor_data


LOGGER = logging.getLogger(__name__)

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


def run_mmdet2d(model, dataset, img, frame, deploy, deploy_runtime=None):
    try:
        detector = perception.object2dfv.MMDetObjectDetector2D(
            model=model,
            dataset=dataset,
            deploy=deploy,
            deploy_runtime=deploy_runtime,
            name="camera_objects_2d",
        )
    except ModuleNotFoundError:
        print("Cannot run mmdet test without the module")
    except FileNotFoundError as e:
        LOGGER.warning(e)
    except ImportError as e:
        LOGGER.warning(e)
    else:
        dets = detector(img, frame=frame)
        if dataset != "coco-person":
            assert len(dets) > 0


# def test_mmdet_2d_perception_from_deploy():
#     frame = 0
#     try:
#         pass
#     except ModuleNotFoundError as e:
#         print("Cannot run mmdet test without the module")
#     else:

#         model_dataset_pairs = [
#             ("cascade_rcnn", "coco", "tensorrt"),
#         ]

#         for model, dataset, runtime in model_dataset_pairs:
#             run_mmdet2d(model, dataset, img, frame, deploy=True, deploy_runtime=runtime)


def test_mmdet_2d_perception_from_checkpoint():
    frame = 0
    try:
        pass
    except ModuleNotFoundError as e:
        print("Cannot run mmdet test without the module")
    else:

        model_dataset_pairs = [
            ("fasterrcnn", "kitti"),
            ("fasterrcnn", "cityscapes"),
            ("fasterrcnn", "coco-person"),
            ("fasterrcnn", "carla-vehicle"),
            ("cascadercnn", "carla-vehicle"),
            ("rtmdet", "coco"),
        ]

        for model, dataset in model_dataset_pairs:
            run_mmdet2d(model, dataset, img, frame, deploy=False)
