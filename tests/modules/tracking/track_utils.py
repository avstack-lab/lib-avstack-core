import sys
from copy import deepcopy

import numpy as np

from avstack.datastructs import DataContainer
from avstack.geometry.transformations import cartesian_to_spherical, xyzvel_to_razelrrt
from avstack.modules.perception.detections import (
    BoxDetection,
    CentroidDetection,
    RazDetection,
    RazelDetection,
    RazelRrtDetection,
)


sys.path.append("tests/")
from utilities import get_ego, get_object_local, get_test_sensor_data


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
T_lidar = lidar_calib.reference
T_camera = camera_calib.reference

name_3d = "lidar"
name_2d = "image"


def make_kitti_tracking_data(
    dt=0.1, n_frames=10, n_targs=4, det_type="box", shuffle=False
):
    ego = get_ego(1)
    objects = [get_object_local(ego, i + 10) for i in range(n_targs)]
    t = 0
    dets_3d_all = []
    for i in range(n_frames):
        dets = deepcopy(objects)
        dets_class = []
        for det in dets:
            det.box3d.t += t * det.velocity.x
            # -- change to lidar reference frame
            reference = lidar_calib.reference
            det.change_reference(reference, inplace=True)
            if det_type == "box":
                det = BoxDetection(name_3d, det.box3d, reference, det.obj_type)
            elif det_type == "centroid":
                det = CentroidDetection(name_3d, det.box3d.t.x, reference, det.obj_type)
            elif det_type == "raz":
                razel = cartesian_to_spherical(det.box3d.t.x)
                det = RazDetection(name_3d, razel[:2], reference, det.obj_type)
            elif det_type == "razel":
                razel = cartesian_to_spherical(det.box3d.t.x)
                det = RazelDetection(name_3d, razel, reference, det.obj_type)
            elif det_type == "razelrrt":
                razelrrt = xyzvel_to_razelrrt(
                    np.array([*det.box3d.t.x, *det.velocity.x])
                )
                det = RazelRrtDetection(name_3d, razelrrt, reference, det.obj_type)
            elif det_type == "xyz":
                raise
            elif det_type == "xy":
                raise
            else:
                raise NotImplementedError
            dets_class.append(det)
        # shuffle the detection order
        if shuffle:
            np.random.shuffle(dets_class)
        detections = DataContainer(i, t, dets_class, source_identifier=name_3d)
        dets_3d_all.append(detections)
        t += dt
    return dets_3d_all


def make_kitti_2d_3d_tracking_data(dt=0.1, n_frames=10, n_targs=4):
    dets_3d_all = make_kitti_tracking_data(dt, n_frames, n_targs=n_targs)
    dets_2d_all = []
    reference = camera_calib.reference
    for i, dets_3d in enumerate(dets_3d_all):
        d_2d = [
            BoxDetection(
                "detector-2d",
                d.box.project_to_2d_bbox(camera_calib),
                reference,
                d.obj_type,
            )
            for d in dets_3d
        ]
        dets_2d_all.append(
            DataContainer(i, dets_3d.timestamp, d_2d, source_identifier=name_2d)
        )
        assert isinstance(dets_3d, DataContainer)
    return dets_2d_all, dets_3d_all
