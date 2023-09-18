# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-28
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-28
# @Description:
"""

"""

import itertools
import os
import shutil

import avstack


class _PerceptionAlgorithm:
    next_id = itertools.count()

    def __init__(self, save_output=False, save_folder="", **kwargs):
        self.ID = next(self.next_id)
        self.save = save_output
        # TODO: self.MODE is not the best way to do this
        self.save_folder = os.path.join(save_folder, "perception", self.MODE)
        if save_output:
            if os.path.exists(self.save_folder):
                shutil.rmtree(self.save_folder)
            os.makedirs(self.save_folder)
        self.iframe = -1

    def __call__(self, data, frame=-1, identifier="", *args, **kwargs):
        self.iframe += 1
        if data is None:
            return None
        else:
            detections = self._execute(
                data, frame=frame, identifier=identifier, *args, **kwargs
            )
            if self.save:
                fname = os.path.join(self.save_folder, "%06i.txt" % frame)
                with open(fname, "w") as f:
                    f.write(detections.encode())
            return detections


mm2d_root = os.path.join(
    os.path.dirname(os.path.dirname(avstack.__file__)), "third_party", "mmdetection"
)
mm3d_root = os.path.join(
    os.path.dirname(os.path.dirname(avstack.__file__)), "third_party", "mmdetection3d"
)


class _MMObjectDetector(_PerceptionAlgorithm):
    def __init__(self, model, dataset, gpu=0, epoch="latest", threshold=None, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset.lower()
        self.algorithm = model

        # Initialize model
        (
            self.threshold,
            config_file,
            checkpoint_file,
            self.input_data,
            label_dataset_override,
        ) = self.parse_mm_model(model, dataset, epoch)
        self.class_names = self.parse_mm_object_classes(label_dataset_override)[0]

        # Get label mapping
        all_objs, _ = self.parse_mm_object_classes(label_dataset_override)
        self.obj_map = {i: n for i, n in enumerate(all_objs)}
        _, self.whitelist = self.parse_mm_object_classes(label_dataset_override)
        self.label_dataset_override = label_dataset_override

        # Find model and checkpoint paths
        if threshold is not None:
            print(f"Overriding default threshold of {self.threshold} with {threshold}")
            self.threshold = threshold
        self.model_name = model
        mod_path = os.path.join(mm2d_root, config_file)

        # HACK: map 'latest' to the checkpoint
        if not os.path.exists(mod_path):
            mod_path = os.path.join(mm3d_root, config_file)
            if "latest" in checkpoint_file:
                chk_path = map_checkpoint_to_latest(mm3d_root, checkpoint_file)
            else:
                chk_path = os.path.join(mm3d_root, checkpoint_file)
            if not os.path.exists(mod_path):
                raise FileNotFoundError(f"Cannot find {config_file} config")
            if not os.path.exists(chk_path):
                raise FileNotFoundError(f"Cannot find {checkpoint_file} checkpoint")
        else:
            if "latest" in checkpoint_file:
                chk_path = map_checkpoint_to_latest(mm2d_root, checkpoint_file)
            else:
                chk_path = os.path.join(mm2d_root, checkpoint_file)
        if not os.path.exists(chk_path):
            raise FileNotFoundError(f"Cannot find {checkpoint_file} checkpoint")
        self.mod_path = mod_path
        self.chk_path = chk_path

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
            all_objs = [
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "couch",
                "potted plant",
                "bed",
                "dining table",
                "toilet",
                "tv",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush",
            ]
            whitelist = ["person", "bicycle", "car"]
        else:
            raise NotImplementedError(dataset)
        return all_objs, whitelist


def map_checkpoint_to_latest(mm_root, checkpoint_file):
    if os.path.exists(os.path.dirname(os.path.join(mm_root, checkpoint_file))):
        with open(
            os.path.join(
                os.path.dirname(os.path.join(mm_root, checkpoint_file)),
                "last_checkpoint",
            ),
            "r",
        ) as f:
            chk_path = f.readlines()[0]
        return chk_path
    else:
        return None
