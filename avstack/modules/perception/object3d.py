# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-27
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-28
# @Description:
"""

"""

import os
import numpy as np
import json
from avstack.modules.perception.base import _PerceptionAlgorithm, _MMObjectDetector
from avstack.modules.perception import detections, utils
from avstack.geometry import bbox
from avstack.datastructs import DataContainer
from cv2 import imwrite
import hashlib, time


# ===========================================================================
# GROUND TRUTH OPERATIONS
# ===========================================================================

class GroundTruth3DObjectDetector(_PerceptionAlgorithm):
    MODE = 'object_3d'
    def _execute(self, ground_truth, identifier, *args, **kwargs):
        """Wrap ground truths to detections relative to ego location"""
        dets = []
        if ground_truth.objects is not None:
            for obj in ground_truth.objects:
                if hasattr(obj, 'box'):
                    obj_in_ego = ground_truth.ego_state.global_to_local(obj)
                    det = detections.BoxDetection(self.MODE, obj_in_ego.box, obj.obj_type)
                else:
                    raise NotImplementedError(obj)
                dets.append(det)
        return DataContainer(ground_truth.frame, ground_truth.timestamp, dets, identifier)


# ===========================================================================
# MM DETECTION OPERATIONS
# ===========================================================================

class MMDetObjectDetector3D(_MMObjectDetector):
    MODE = 'object_3d'
    def __init__(self, model='pointpillars', dataset='kitti', front_only=False,
            gpu=0, epoch='latest', threshold=None, **kwargs):
        super().__init__(model, dataset, gpu, epoch, threshold, **kwargs)
        if self.input_data == 'camera':
            from mmdet3d.apis import inference_mono_3d_detector
            self.inference_detector = inference_mono_3d_detector
            self.inference_mode = 'from_mono'
        elif self.input_data == 'lidar':
            from mmdet3d.apis import inference_detector
            self.inference_detector = inference_detector
            self.inference_mode = 'from_lidar'
        else:
            raise NotImplementedError(self.input_data)
        self.front_only = front_only

        if model == '3dssd':
            assert gpu == 0, 'For some reason, 3dssd must be on gpu 0'

    def _execute(self, data, identifier, eval_method='file', **kwargs):
        # -- inference
        result_ = self.run_mm_inference(self.inference_detector, self.model, data, self.input_data, eval_method)

        # -- postprocess objects
        detections = utils.convert_mm3d_to_avstack(result_, data.calibration,
            self.model, self.obj_map, self.whitelist, self.input_data,
            identifier, self.dataset, self.threshold, front_only=self.front_only,
            prune_low=(self.algorithm in ['pgd']),
            prune_close=(self.algorithm in ['pgd']), **kwargs)
        return DataContainer(data.frame, data.timestamp, detections, identifier)

    @staticmethod
    def run_mm_inference(inference_detector, model, data, input_data, eval_method):
        if eval_method == 'file':
            hash_id = int(abs(hash(time.time())) % 1e8)
            temp_dir = 'temp'
            os.makedirs(temp_dir, exist_ok=True)
            if input_data == 'lidar':
                file = os.path.join(temp_dir, f'temp_lidar_{hash_id}.bin')
                data.data.tofile(file)
                try:
                    result_, _ = inference_detector(model, file)
                finally:
                    os.remove(file)
            elif input_data == 'camera':
                file = os.path.join(temp_dir, f'temp_image_{hash_id}.png')
                imwrite(file, data.data)
                P = data.calibration.P.tolist()
                if (len(P)==3) and (len(P[0])==4):
                    P.append([0,0,0,1.])
                json_file = os.path.join(temp_dir, f'temp_intrinsics_{hash_id}.json')
                json_data = {'images':[{'file_name':file, 'cam_intrinsic':P}]}
                with open(json_file, 'w') as f:
                    json.dump(json_data, f)
                try:
                    result_, _ = inference_detector(model, file, json_file)
                finally:
                    os.remove(file)
                    os.remove(json_file)
            else:
                raise NotImplementedError(input_data)
        elif eval_method == 'data':
            if input_data == 'lidar':
                raise NotImplementedError(eval_method + ' ' + input_data)
            elif input_data == 'camera':
                raise NotImplementedError(eval_method + ' ' + input_data)
        else:
            raise NotImplementedError(eval_method)
        return result_

    @staticmethod
    def parse_mm_object_classes(dataset):
        dataset = dataset.lower()
        if dataset.lower() == 'kitti':
            all_objs = ['Car', 'Pedestrian', 'Cyclist']
            whitelist = all_objs
        elif dataset.lower() == 'nuscenes':
            all_objs = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
                'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
            whitelist = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'pedestrian']
        elif dataset.lower() in ['carla', 'carla-infrastructure']:
            all_objs = ['car', 'bicycle', 'truck', 'motorcycle']
            whitelist = all_objs
        else:
            raise NotImplementedError(dataset)
        return all_objs, whitelist

    @staticmethod
    def parse_mm_model(model, dataset, epoch):
        dataset = dataset.lower()
        epoch_str = 'latest' if epoch == 'latest' else 'epoch_{}'.format(epoch)
        obj_class_dataset_override = dataset
        if model == 'second':
            if dataset == 'kitti':
                threshold = 0.5
                config_file = 'configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py'
                checkpoint_file = 'checkpoints/kitti/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth'
            else:
                raise NotImplementedError(f'{model}, {dataset} not compatible yet')
            input_data = 'lidar'
        elif model == 'pointpillars':
            if dataset == 'kitti':
                threshold = 0.5
                config_file = 'configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
                checkpoint_file = 'checkpoints/kitti/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
            elif dataset == 'nuscenes':
                threshold = 0.4
                config_file = 'configs/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d.py'
                checkpoint_file = 'checkpoints/nuscenes/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth'
            elif dataset == 'carla':
                threshold = 0.5
                config_file = 'configs/carla/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_carla-3d.py'
                checkpoint_file = f'work_dirs/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_carla-3d/{epoch_str}.pth'
            elif dataset == 'carla-infrastructure':
                threshold = 0.5
                config_file = 'configs/carla/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_carla-infrastructure-3d.py'
                checkpoint_file = f'work_dirs/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_carla-infrastructure-3d/{epoch_str}.pth'
            else:
                raise NotImplementedError(f'{model}, {dataset} not compatible yet')
            input_data = 'lidar'
        elif model == '3dssd':
            if dataset == 'kitti':
                threshold = 0.2
                config_file = 'configs/3dssd/3dssd_4x4_kitti-3d-car.py'
                checkpoint_file = 'checkpoints/kitti/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth'
            elif dataset == 'carla':
                threshold = 0.5
                config_file = 'configs/carla/3dssd_4x4_carla-3d.py'
                checkpoint_file = f'work_dirs/3dssd_4x4_carla-3d/{epoch_str}.pth'
            elif dataset == 'carla-infrastructure':
                threshold = 0.5
                config_file = 'configs/carla/3dssd_4x4_carla-infrastructure-3d.py'
                checkpoint_file = f'work_dirs/3dssd_4x4_carla-infrastructure-3d/{epoch_str}.pth'
            else:
                raise NotImplementedError(f'{model}, {dataset} not compatible yet')
            input_data = 'lidar'
        elif model == 'ssn':
            if dataset == 'kitti':
                obj_class_dataset_override = 'carla'
                threshold = 0.3
                config_file = 'work_dirs/hv_ssn_secfpn_sbn-all_2x16_2x_carla-3d/hv_ssn_secfpn_sbn-all_2x16_2x_carla-3d.py'
                checkpoint_file = f'work_dirs/hv_ssn_secfpn_sbn-all_2x16_2x_carla-3d/{epoch_str}.pth'
            elif dataset == 'carla':
                threshold = 0.5
                config_file = 'work_dirs/hv_ssn_secfpn_sbn-all_2x16_2x_carla-3d/hv_ssn_secfpn_sbn-all_2x16_2x_carla-3d.py'
                checkpoint_file = f'work_dirs/hv_ssn_secfpn_sbn-all_2x16_2x_carla-3d/{epoch_str}.pth'
            elif dataset == 'carla-infrastructure':
                raise NotImplementedError
            elif dataset == 'nuscenes':
                threshold = 0.5
                config_file = 'configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d.py'
                checkpoint_file = f'checkpoints/nuscenes/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d_20210830_101351-51915986.pth'
            else:
                raise NotImplementedError(f'{model}, {dataset} not compatible yet')
            input_data = 'lidar'
        elif model == 'pgd':
            if dataset == 'kitti':
                threshold = 5
                config_file = 'configs/pgd/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d.py'
                checkpoint_file ='checkpoints/kitti/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608-8a97533b.pth'
            elif dataset == 'nuscenes':
                threshold = 0.1
                config_file = 'configs/pgd/pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_finetune.py'
                checkpoint_file = 'checkpoints/nuscenes/pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_finetune_20211114_162135-5ec7c1cd.pth'
            else:
                raise NotImplementedError(f'{model}, {dataset} not compatible yet')
            input_data = 'camera'
        else:
            raise NotImplementedError(model)
        return threshold, config_file, checkpoint_file, input_data, obj_class_dataset_override
