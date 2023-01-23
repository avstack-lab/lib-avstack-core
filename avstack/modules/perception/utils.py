# -*- coding: utf-8 -*-
# @Author: spencer@primus
# @Date:   2022-06-07
# @Last Modified by:   spencer@primus
# @Last Modified time: 2022-09-14


import numpy as np

import avstack
from avstack.geometry import q_mult_vec


car_classes = ["car", "Car", "vehicle"]
ped_classes = ["pedestrian", "walker", "person", "Pedestrian"]
bic_classes = ["bicycle", "cyclist", "Cyclist", "cycler"]
ignore_classes = ["traffic_cone", "barrier", "trailer"]

k_classes = [
    ("Car", car_classes + ["bus", "truck"]),
    ("Pedestrian", ped_classes),
    ("Cyclist", bic_classes),
    ("ignore", ignore_classes + ["motorcycle"]),
]

nu_classes = [
    ("car", car_classes),
    ("truck", ["truck"]),
    ("construction_vehicle", ["construction_vehicle"]),
    ("van", ["van"]),
    ("bicycle", bic_classes),
    ("motorcycle", ["motorcycle"]),
    ("bus", ["bus"]),
    ("person", ped_classes),
    ("ignore", ignore_classes),
]

carla_clases = [
    ("car", car_classes),
    ("truck", ["truck", "van", "bus"]),
    ("bicycle", bic_classes),
    ("motorcycle", ["motorcycle"], ("ignore", ignore_classes)),
]

class_maps = {
    "kitti": {k: ks[0] for ks in k_classes for k in ks[1]},
    "nuscenes": {k: ks[0] for ks in nu_classes for k in ks[1]},
    "nuimages": {k: ks[0] for ks in nu_classes for k in ks[1]},
    "carla": {k: ks[0] for ks in carla_clases for k in ks[1]},
}


def convert_mm2d_to_avstack(
    result_,
    calib,
    model_2d,
    source_identifier,
    dataset,
    score_thresh,
    whitelist,
    class_names,
):
    if isinstance(result_, tuple):
        bbox_result, segms = result_
        if isinstance(segms, tuple):
            segms = segms[0]  # ms rcnn
    else:
        bbox_result, segm_result = result_, None
        segms = None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    if score_thresh > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        scores_pre = scores.copy()
        inds = scores > score_thresh
        scores = scores[inds]
        bboxes = bboxes[inds, :-1]
        labels = labels[inds]
        if segms is not None:
            segms = [s for seg in segms for s in seg]
            segms = [s for i, s in enumerate(segms) if inds[i]]

    # -- object types
    obj_type_text = [
        class_maps[dataset][class_names[label]]
        if (class_names is not None) and (class_names[label] in class_maps[dataset])
        else f"class {label}"
        for label in labels
    ]

    # -- make objects
    if segms is None:
        dets = [
            avstack.modules.perception.detections.BoxDetection(
                source_identifier, avstack.geometry.Box2D(bbox, calib), obj_type, score
            )
            for bbox, obj_type, score in zip(bboxes, obj_type_text, scores)
            if obj_type in whitelist
        ]
    else:
        dets = [
            avstack.modules.perception.detections.MaskDetection(
                source_identifier,
                avstack.geometry.Box2D(bbox, calib),
                avstack.geometry.SegMask2D(segm, calib),
                obj_type,
                score,
            )
            for bbox, segm, obj_type, score in zip(bboxes, segms, obj_type_text, scores)
            if obj_type in whitelist
        ]

    return dets


def convert_mm3d_to_avstack(
    result_,
    calib,
    model_3d,
    obj_map,
    whitelist,
    input_data,
    source_identifier,
    dataset,
    thresh=0.5,
    dist_min=2.0,
    front_only=False,
    prune_low=False,
    thresh_low=-1,
    prune_close=False,
    thresh_close=1.5,
    verbose=False,
    **kwargs,
):
    dets = []
    # -- parse object information
    if "lidar" in input_data.lower():
        obj_base = result_[0]
        if "pts_bbox" in obj_base:
            obj_base = obj_base["pts_bbox"]
    elif ("cam" in input_data.lower()) or ("image" in input_data.lower()):
        obj_base = result_[0]["img_bbox"]
    else:
        raise NotImplementedError(input_data)

    # -- convert boxes
    prev_locs = []
    for i_box in range(len(obj_base["boxes_3d"])):
        # if obj_base['labels_3d'] not in obj_map:
        #     continue
        obj_type = class_maps[dataset][obj_map[obj_base["labels_3d"][i_box].item()]]
        if obj_type in whitelist:
            if obj_base["scores_3d"][i_box] > thresh:
                # get info from detections
                box = obj_base["boxes_3d"][i_box]
                ten = box.tensor[0]
                cent = np.array([c.item() for c in box.center[0]])
                if np.linalg.norm(cent) < dist_min:
                    continue
                if (
                    ("pointpillars" in model_3d.cfg.filename)
                    or ("3dssd" in model_3d.cfg.filename)
                    or ("ssn" in model_3d.cfg.filename)
                ):
                    h, w, l = ten[5].item(), ten[4].item(), ten[3].item()
                    if dataset == "kitti":
                        yaw = box.yaw.item()
                        q_S_2_obj = avstack.transformations.transform_orientation(
                            [0, 0, yaw], "euler", "quat"
                        )
                        q_O_2_obj = q_S_2_obj  # sensor is our origin
                        x_O_2_obj_in_O = cent
                        if "ssn" in model_3d.cfg.filename:
                            x_O_2_obj_in_O[2] += h  # whoops
                            where_is_t = "center"
                        else:
                            where_is_t = "bottom"
                        origin = calib.origin
                    elif dataset == "nuscenes":
                        yaw = box.yaw.item()
                        if "pointpillars" in model_3d.cfg.filename:
                            q_O1_2_obj = avstack.transformations.transform_orientation(
                                [0, 0, -yaw], "euler", "quat"
                            )
                            q_O_2_O1 = avstack.transformations.transform_orientation(
                                [0, 0, -np.pi / 2], "euler", "quat"
                            )
                            q_O_2_obj = q_O1_2_obj * q_O_2_O1
                        else:
                            q_O1_2_obj = avstack.transformations.transform_orientation(
                                [0, 0, yaw], "euler", "quat"
                            )
                            q_O_2_obj = q_O1_2_obj
                        x_O_2_obj_in_O = cent
                        where_is_t = "bottom"
                        origin = calib.origin
                    elif dataset == "carla":
                        yaw = box.yaw.item()
                        q_O_2_obj = avstack.transformations.transform_orientation(
                            [0, 0, yaw], "euler", "quat"
                        )
                        x_O_2_obj_in_O = cent
                        x_O_2_obj_in_O[2] += h  # whoops
                        where_is_t = "center"
                        origin = calib.origin
                    elif dataset == "carla-infrastructure":
                        yaw = box.yaw.item()  # yaw is in O's frame here!!
                        q_O_2_obj = avstack.transformations.transform_orientation(
                            [0, 0, yaw], "euler", "quat"
                        )
                        x_O_2_obj_in_O = cent
                        x_O_2_obj_in_O[2] += 1.8
                        where_is_t = "bottom"
                        origin = calib.origin
                    else:
                        raise NotImplementedError(dataset)
                elif "pgd" in model_3d.cfg.filename:
                    w, h, l = ten[5].item(), ten[4].item(), ten[3].item()
                    if dataset == "kitti":
                        yaw = box.yaw.item() - np.pi / 2
                        q_L_2_obj = avstack.transformations.transform_orientation(
                            [0, 0, -yaw], "euler", "quat"
                        )
                        q_C_2_obj = q_L_2_obj * avstack.geometry.q_cam_to_stan
                        x_C_2_obj_in_C = cent
                        where_is_t = "bottom"
                        origin = calib.origin  # camera origin
                    elif dataset == "nuscenes":
                        yaw = box.yaw.item() - np.pi / 2
                        q_L_2_obj = avstack.transformations.transform_orientation(
                            [0, 0, -yaw], "euler", "quat"
                        )
                        q_C_2_obj = q_L_2_obj * avstack.geometry.q_cam_to_stan
                        x_C_2_obj_in_C = cent
                        where_is_t = "bottom"
                        origin = calib.origin
                    else:
                        raise NotImplementedError(dataset)
                    x_O_2_obj_in_O = x_C_2_obj_in_C
                    q_O_2_obj = q_C_2_obj
                else:
                    raise NotImplementedError(model_3d.cfg.filename)

                # make box output
                box3d = avstack.geometry.bbox.Box3D(
                    [h, w, l, x_O_2_obj_in_O, q_O_2_obj],
                    origin=origin,
                    where_is_t=where_is_t,
                )

                # -- pruning
                # ---- too low
                if prune_low and (box3d.t.vector_global[2] < thresh_low):
                    if verbose:
                        print("Box registered uncharacteristically low...skipping")
                    continue
                # ---- too close to another object
                if len(prev_locs) > 0:
                    too_close = False
                    for p_loc in prev_locs:
                        if box3d.t.distance(p_loc) < thresh_close:
                            too_close = True
                            break
                    if too_close:
                        if verbose:
                            print("Object was too close to another")
                        continue
                # ---- we made it!
                prev_locs.append(x_O_2_obj_in_O)
                score = obj_base["scores_3d"][i_box].item()
                dets.append(
                    avstack.modules.perception.detections.BoxDetection(
                        source_identifier, box3d, obj_type, score
                    )
                )
    return dets
