import json
import os
import tempfile

from cv2 import imwrite

from avstack.config import MODELS
from avstack.datastructs import DataContainer
from avstack.environment.objects import ObjectState
from avstack.geometry import Box3D
from avstack.modules.perception import detections, utils
from avstack.modules.perception.base import _MMObjectDetector, _PerceptionAlgorithm


# ===========================================================================
# GROUND TRUTH OPERATIONS
# ===========================================================================


@MODELS.register_module()
class GroundTruth3DObjectDetector(_PerceptionAlgorithm):
    MODE = "object_3d"

    def _execute(self, ground_truth, identifier, *args, **kwargs):
        """Wrap ground truths to detections relative to ego location"""
        dets = []
        if ground_truth.objects is not None:
            for obj in ground_truth.objects:
                if hasattr(obj, "box"):
                    obj_in_ego = obj.change_reference(
                        ground_truth.ego_state, inplace=False
                    )
                    det = detections.BoxDetection(
                        self.MODE,
                        obj_in_ego.box,
                        obj_in_ego.box.reference,
                        obj.obj_type,
                    )
                else:
                    raise NotImplementedError(obj)
                dets.append(det)
        return DataContainer(
            ground_truth.frame, ground_truth.timestamp, dets, identifier
        )


@MODELS.register_module()
class Passthrough3DObjectDetector(_PerceptionAlgorithm):
    MODE = "object_3d"

    def _execute(self, data, identifier, *args, **kwargs):
        dets = []
        for obj in data:
            if isinstance(obj, detections.Detection_):
                det = obj
            elif isinstance(obj, ObjectState):
                det = detections.BoxDetection(
                    source_identifier=self.MODE,
                    box=obj.box,
                    reference=obj.reference,
                    obj_type=obj.obj_type,
                )
            elif isinstance(obj, Box3D):
                det = detections.BoxDetection(
                    source_identifier=self.MODE,
                    box=obj,
                    reference=obj.reference,
                    obj_type=obj.obj_type,
                )
            elif hasattr(obj, "box"):
                det = detections.BoxDetection(
                    source_identifier=self.MODE,
                    box=obj.box,
                    reference=obj.reference,
                    obj_type=obj.obj_type,
                )
            else:
                det = detections.CentroidDetection(
                    centroid=obj.x,
                    source_identifier=self.MODE,
                    reference=obj.reference,
                    obj_type=obj.obj_type if hasattr(obj, "obj_type") else None,
                )
            dets.append(det)
        return DataContainer(data.frame, data.timestamp, dets, identifier)


# ===========================================================================
# MM DETECTION OPERATIONS
# ===========================================================================


@MODELS.register_module()
class MMDetObjectDetector3D(_MMObjectDetector):
    MODE = "object_3d"

    def __init__(
        self,
        model="pointpillars",
        dataset="kitti",
        deploy=False,
        front_only=False,
        gpu=0,
        epoch="latest",
        threshold=None,
        deploy_runtime="tensorrt",
        prune_duplicate=True,
        thresh_duplicate=0.5,
        **kwargs,
    ):
        super().__init__(
            model=model,
            dataset=dataset,
            deploy=deploy,
            deploy_runtime=deploy_runtime,
            threshold=threshold,
            gpu=gpu,
            epoch=epoch,
            **kwargs,
        )
        self.front_only = front_only
        self.prune_duplicate = prune_duplicate
        self.thresh_duplicate = thresh_duplicate

    def _execute(self, data, identifier, eval_method="file", **kwargs):
        from mmdet3d.utils import register_all_modules

        register_all_modules(init_default_scope=True)

        # -- inference
        result_ = self.run_mm_inference(
            self.inference_detector,
            self.model,
            data,
            self.input_data,
            eval_method,
            do_projection=self._do_projection,
        )

        # -- postprocess objects
        detections = utils.convert_mm3d_to_avstack(
            result_,
            data.calibration,
            self.model,
            self.obj_map,
            self.whitelist,
            self.input_data,
            identifier,
            self.dataset,
            self.threshold,
            do_projection=self._do_projection,
            front_only=self.front_only,
            prune_low=(self.model_name in ["pgd"]),
            prune_duplicate=self.prune_duplicate,
            thresh_duplicate=self.thresh_duplicate,
            **kwargs,
        )
        return DataContainer(data.frame, data.timestamp, detections, identifier)

    @staticmethod
    def run_mm_inference(
        inference_detector, model, data, input_data, eval_method, do_projection=False
    ):
        if eval_method == "file":
            with tempfile.TemporaryDirectory() as temp_dir:
                suffix = ".bin" if input_data == "lidar" else ".png"
                fd_data, data_file = tempfile.mkstemp(suffix=suffix, dir=temp_dir)
                os.close(fd_data)  # need to start with the file closed...
                if input_data == "lidar":
                    # project if necessary
                    if do_projection:
                        data = data.transform_to_ground()
                    data.save_to_file(data_file)
                    result_, _ = inference_detector(model, data_file)
                elif input_data == "camera":
                    imwrite(data_file, data.data)
                    P = data.calibration.P.tolist()
                    if (len(P) == 3) and (len(P[0]) == 4):
                        P.append([0, 0, 0, 1.0])
                    fd_json, json_file = tempfile.mkstemp(suffix=".json", dir=temp_dir)
                    os.close(fd_json)
                    json_data = {
                        "images": [{"file_name": data_file, "cam_intrinsic": P}]
                    }
                    with open(json_file, "w") as f:
                        json.dump(json_data, f)
                    result_ = inference_detector(model, data_file, json_file)
                else:
                    raise NotImplementedError(input_data)
        elif eval_method == "data":
            if input_data == "lidar":
                raise NotImplementedError(eval_method + " " + input_data)
            elif input_data == "camera":
                raise NotImplementedError(eval_method + " " + input_data)
        else:
            raise NotImplementedError(eval_method)
        return result_

    @staticmethod
    def parse_mm_model_from_checkpoint(model, dataset, epoch):
        dataset = dataset.lower()
        epoch_str = "latest" if epoch == "latest" else "epoch_{}".format(epoch)
        obj_class_dataset_override = dataset
        do_projection = False
        if model == "second":
            if dataset == "kitti":
                threshold = 0.5
                config_file = (
                    "configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py"
                )
                checkpoint_file = "checkpoints/kitti/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth"
            else:
                raise NotImplementedError(f"{model}, {dataset} not compatible yet")
            input_data = "lidar"
        elif model == "pointpillars":
            if dataset == "kitti":
                threshold = 0.5
                config_file = "configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py"
                checkpoint_file = "checkpoints/kitti/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"
            elif dataset == "nuscenes":
                threshold = 0.4
                config_file = (
                    "configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py"
                )
                checkpoint_file = "checkpoints/nuscenes/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth"
            elif dataset == "carla-joint":
                threshold = 0.5
                config_file = "work_dirs/pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-joint/pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-joint.py"
                checkpoint_file = f"work_dirs/pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-joint/{epoch_str}.pth"
            elif dataset == "carla-vehicle":
                threshold = 0.5
                config_file = "work_dirs/pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-vehicle/pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-vehicle.py"
                checkpoint_file = f"work_dirs/pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-vehicle/{epoch_str}.pth"
            elif dataset == "carla-infrastructure":
                threshold = 0.5
                config_file = "work_dirs/pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-infrastructure/pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-infrastructure.py"
                checkpoint_file = f"work_dirs/pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-infrastructure/{epoch_str}.pth"
                do_projection = True
            else:
                raise NotImplementedError(f"{model}, {dataset} not compatible yet")
            input_data = "lidar"
        elif model == "3dssd":
            if dataset == "kitti":
                threshold = 0.2
                config_file = "configs/3dssd/3dssd_4x4_kitti-3d-car.py"
                checkpoint_file = "checkpoints/kitti/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth"
            elif dataset == "carla":
                threshold = 0.5
                config_file = "configs/carla/3dssd_4x4_carla-3d.py"
                checkpoint_file = f"work_dirs/3dssd_4x4_carla-3d/{epoch_str}.pth"
            elif dataset == "carla-infrastructure":
                threshold = 0.5
                config_file = "configs/carla/3dssd_4x4_carla-infrastructure-3d.py"
                checkpoint_file = (
                    f"work_dirs/3dssd_4x4_carla-infrastructure-3d/{epoch_str}.pth"
                )
                do_projection = True
            else:
                raise NotImplementedError(f"{model}, {dataset} not compatible yet")
            input_data = "lidar"
        elif model == "ssn":
            if dataset == "kitti":
                obj_class_dataset_override = "carla"
                threshold = 0.3
                config_file = (
                    "work_dirs/carla/hv_ssn_secfpn_sbn-all_2x16_2x_carla-3d.py"
                )
                checkpoint_file = (
                    f"work_dirs/carla/hv_ssn_secfpn_sbn-all_2x16_2x_carla-3d.pth"
                )
            elif dataset == "carla":
                threshold = 0.5
                config_file = (
                    "work_dirs/carla/hv_ssn_secfpn_sbn-all_2x16_2x_carla-3d.py"
                )
                checkpoint_file = (
                    "work_dirs/carla/hv_ssn_secfpn_sbn-all_2x16_2x_carla-3d.pth"
                )
            elif dataset == "carla-infrastructure":
                raise NotImplementedError
            elif dataset == "nuscenes":
                threshold = 0.5
                config_file = "configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d.py"
                checkpoint_file = f"checkpoints/nuscenes/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d_20210830_101351-51915986.pth"
            else:
                raise NotImplementedError(f"{model}, {dataset} not compatible yet")
            input_data = "lidar"
        elif model == "pgd":
            if dataset == "kitti":
                threshold = 5
                config_file = (
                    "configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py"
                )
                checkpoint_file = "checkpoints/kitti/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608-8a97533b.pth"
            elif dataset == "nuscenes":
                threshold = 0.1
                config_file = "configs/pgd/pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_finetune.py"
                checkpoint_file = "checkpoints/nuscenes/pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_finetune_20211114_162135-5ec7c1cd.pth"
            else:
                raise NotImplementedError(f"{model}, {dataset} not compatible yet")
            input_data = "camera"
        else:
            raise NotImplementedError(model)
        return (
            threshold,
            config_file,
            checkpoint_file,
            input_data,
            obj_class_dataset_override,
            do_projection,
        )
