# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-27
# @Last Modified by:   spencer@primus
# @Last Modified date: 2022-07-28
# @Description:
"""

"""
import hashlib
import os
import tempfile
import time

import numpy as np
from cv2 import imwrite

import avstack
from avstack.datastructs import DataContainer
from avstack.geometry import StandardCoordinates, bbox
from avstack.modules.perception import detections, utils
from avstack.modules.perception.base import _MMObjectDetector, _PerceptionAlgorithm


# ===========================================================================
# TRUTH OPERATIONS
# ===========================================================================


class GroundTruth2DFvObjectDetector(_PerceptionAlgorithm):
    MODE = "object_2d"

    def __call__(self, ground_truth, *args, **kwargs):
        raise NotImplementedError


# ===========================================================================
# CUSTOM OPERATIONS
# ===========================================================================


class InfrastructureCameraTo3d(_PerceptionAlgorithm):
    """
    This algorithm is suitable for 2D infrastructure sensors that have some
    non-zero pitch angle. Under the assumption of a flat ground plane and
    precise knowledge of the calibration matrix for the sensor, this algorithm
    computes the full 3D positioning information assuming the object is on
    the ground plane. It then uses and optimization procedure to fit a 3D
    bounding box that matches the 2D detection from the camera

    The procedure is as follows:
    1. Get 2D mask detections from a camera algorithm
    2. From the centroid of the mask, get azimuth and elevation angles
    3. Approximating the ground as flat, use sensor calibration to predict range
    4. Fit 3D bounding box corners that are consistent with this 2D box
    5. Build a box object from the corners
    """

    MODE = "object_3d"

    def __init__(self, detector, algorithm="range_from_ground"):
        self.detector = detector
        self.algorithm = algorithm

    def _execute(self, data, identifier, *args, **kwargs):
        # -- get detections from mmdet
        detections = self.detector(data, identifier, *args, **kwargs)

        # -- use approximation of a flat ground plane to get 3D positioning
        # TODO: add possibility of roll angle?
        # TODO: account for RefChoc
        z = data.origin.x[2]
        pitch = data.origin.euler[1]  # pitch down is positive

        # -- from the detections, compute the azimuth and elevation angles
        for det in detections:
            az, el = det.box.angles
            view_angle = pitch + el  # TODO: check this out...
            rng = z / np.sin(view_angle)

        # -- fit a 3D bounding box that is consistent with the 2D projection
        raise

        # -- from the 3D bounding box corners, build the box object
        raise


# ===========================================================================
# MM DETECTION OPERATIONS
# ===========================================================================


class MMDetObjectDetector2D(_MMObjectDetector):
    MODE = "object_2d"

    def __init__(
        self,
        model="fasterrcnn",
        dataset="kitti",
        threshold=None,
        gpu=0,
        epoch="latest",
        **kwargs,
    ):
        super().__init__(model, dataset, gpu, epoch, threshold, **kwargs)
        from mmdet.apis import inference_detector, init_detector
        from mmdet.utils import register_all_modules
        register_all_modules(init_default_scope=True)
        self.inference_detector = inference_detector
        self.model = init_detector(self.mod_path, self.chk_path, device=f"cuda:{gpu}")

    def _execute(self, data, identifier, is_rgb=True, eval_method="data", **kwargs):
        from mmdet.utils import register_all_modules
        register_all_modules(init_default_scope=True)

        # -- inference
        result_ = self.run_mm_inference(
            self.inference_detector, self.model, data, is_rgb, eval_method
        )

        # -- postprocess objects
        detections = utils.convert_mm2d_to_avstack(
            result_,
            data.calibration,
            self.model,
            identifier,
            self.label_dataset_override,
            self.threshold,
            self.whitelist,
            self.class_names,
        )
        return DataContainer(data.frame, data.timestamp, detections, identifier)

    @staticmethod
    def run_mm_inference(inference_detector, model, data, is_rgb, eval_method):
        if eval_method == "file":
            with tempfile.TemporaryDirectory() as temp_dir:
                fd_data, data_file = tempfile.mkstemp(suffix=".png", dir=temp_dir)
                os.close(fd_data)  # need to start with the file closed...
                if is_rgb:
                    imwrite(data_file, data.data[:, :, ::-1])
                else:
                    imwrite(data_file, data.data)
                result_ = inference_detector(model, data_file)
        elif eval_method == "data":
            result_ = inference_detector(
                model, data.data if not is_rgb else data.data[:, :, ::-1]
            )
        else:
            raise NotImplementedError(eval_method)
        return result_

    @staticmethod
    def parse_mm_object_classes(dataset):
        if dataset == "kitti":
            all_objs = ["Car", "Pedestrian", "Cyclist"]
            whitelist = all_objs
        elif dataset == "nuscenes":
            all_objs = [
                "person",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
            ]
            whitelist = all_objs
        elif dataset == "nuimages":
            all_objs = (
                "car",
                "truck",
                "trailer",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "pedestrian",
                "traffic_cone",
                "barrier",
            )
            whitelist = (
                "car",
                "truck",
                "trailer",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "pedestrian",
            )
        elif dataset in ["carla", "carla-infrastructure"]:
            all_objs = ["car", "bicycle", "truck", "motorcycle"]
            whitelist = all_objs
        elif dataset == "cityscapes":
            all_objs = [
                "person",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
            ]
            whitelist = all_objs
        elif dataset == "coco-person":
            all_objs = ["person"]
            whitelist = all_objs
        elif dataset == "coco":
            all_objs = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
            whitelist = ["person", "bicycle", "car"]
        else:
            raise NotImplementedError(dataset)
        return all_objs, whitelist

    @staticmethod
    def parse_mm_model(model, dataset, epoch):
        input_data = "camera"
        label_dataset_override = dataset
        if model == "yolov3":
            raise NotImplementedError("yolo not trained yet")
        elif model == "rtmdet":
            if dataset == "coco":
                threshold = 0.5
                config_file = "configs/rtmdet/rtmdet_m_8xb32-300e_coco.py"
                checkpoint_file = "checkpoints/rtmdet/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
            else:
                raise NotImplementedError(f"{model}, {dataset} not compatible yet")
        elif model == "fasterrcnn":
            if dataset == "kitti":
                threshold = 0.5
                config_file = "configs/cityscapes/faster-rcnn_r50_fpn_1x_cityscapes.py"
                checkpoint_file = "checkpoints/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
                label_dataset_override = "cityscapes"
            elif dataset == "nuscenes":
                threshold = 0.7
                config_file = "work_dirs/nuscenes/faster_rcnn_r50_fpn_1x_nuscenes.py"
                checkpoint_file = (
                    "work_dirs/nuscenes/faster_rcnn_r50_fpn_1x_nuscenes.pth"
                )
            elif dataset == "carla":
                threshold = 0.7
                config_file = "work_dirs/carla/faster_rcnn_r50_fpn_1x_carla.py"
                checkpoint_file = "work_dirs/carla/faster_rcnn_r50_fpn_1x_carla.pth"
            elif dataset == "carla-infrastructure":
                threshold = 0.7
                config_file = (
                    "work_dirs/carla/faster_rcnn_r50_fpn_1x_carla_infrastructure.py"
                )
                checkpoint_file = (
                    "work_dirs/carla/faster_rcnn_r50_fpn_1x_carla_infrastructure.pth"
                )
            elif dataset == "cityscapes":
                threshold = 0.5
                config_file = "configs/cityscapes/faster-rcnn_r50_fpn_1x_cityscapes.py"
                checkpoint_file = "checkpoints/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
            elif dataset == "coco-person":
                threshold = 0.25
                config_file = "configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py"
                checkpoint_file = "checkpoints/coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
            else:
                raise NotImplementedError(f"{model}, {dataset} not compatible yet")
        elif model == "htc":
            if dataset == "nuimages":
                threshold = 0.7
                config_file = "configs/nuimages/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim.py"
                checkpoint_file = "checkpoints/nuimages/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim_20201008_211222-0b16ac4b.pth"
            else:
                raise NotImplementedError(f"{model}, {dataset} not compatible yet")
        elif model == "cascade-mask-rcnn":
            if dataset in [
                "nuimages",
                "carla",
                "kitti",
                "nuscenes",
            ]:  # TODO eventually separate these
                threshold = 0.7
                config_file = (
                    "configs/nuimages/cascade-mask-rcnn_r50_fpn_coco-20e-1x_nuim.py"
                )
                checkpoint_file = "checkpoints/nuimages/cascade_mask_rcnn_r50_fpn_coco-20e_1x_nuim_20201009_124158-ad0540e3.pth"
                label_dataset_override = "nuimages"
            else:
                raise NotImplementedError(f"{model}, {dataset} not compatible yet")
        else:
            raise NotImplementedError(model)
        return (
            threshold,
            config_file,
            checkpoint_file,
            input_data,
            label_dataset_override,
        )
