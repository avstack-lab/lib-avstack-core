import sys
from copy import deepcopy

import numpy as np

from avstack.calibration import CameraCalibration
from avstack.datastructs import DataContainer
from avstack.geometry import (
    Attitude,
    Box2D,
    Box3D,
    GlobalOrigin3D,
    Position,
    ReferenceFrame,
    q_stan_to_cam,
)
from avstack.geometry.transformations import (
    cartesian_to_spherical,
    transform_orientation,
    xyzvel_to_razelrrt,
)
from avstack.maskfilters import box_in_fov
from avstack.modules.perception.detections import (
    BoxDetection,
    CentroidDetection,
    RazDetection,
    RazelDetection,
    RazelRrtDetection,
)


sys.path.append("tests/")
from utilities import get_ego, get_object_local, get_test_sensor_data


try:
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
except FileNotFoundError:
    print("Cannot find data files but continuing anyway")

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
                det = BoxDetection(
                    data=det.box3d,
                    noise=np.array([1, 1, 1, 0.5, 0.5, 0.5]) ** 2,
                    source_identifier=name_3d,
                    reference=reference,
                    obj_type=det.obj_type,
                )
            elif det_type in ["xy", "xyz", "centroid"]:
                if det_type in ["xy"]:
                    centroid = det.box3d.t.x[:2]
                    noise = np.array([1, 1]) ** 2
                else:
                    centroid = det.box3d.t.x
                    noise = np.array([1, 1, 1]) ** 2
                det = CentroidDetection(
                    data=centroid,
                    noise=noise,
                    source_identifier=name_3d,
                    reference=reference,
                    obj_type=det.obj_type,
                )
            elif det_type == "raz":
                razel = cartesian_to_spherical(det.box3d.t.x)
                det = RazDetection(
                    data=razel[:2],
                    noise=np.array([1, 0.1]) ** 2,
                    source_identifier=name_3d,
                    reference=reference,
                    obj_type=det.obj_type,
                )
            elif det_type == "razel":
                razel = cartesian_to_spherical(det.box3d.t.x)
                det = RazelDetection(
                    data=razel,
                    noise=np.array([1, 0.1, 0.1]) ** 2,
                    source_identifier=name_3d,
                    reference=reference,
                    obj_type=det.obj_type,
                )
            elif det_type == "razelrrt":
                razelrrt = xyzvel_to_razelrrt(
                    np.array([*det.box3d.t.x, *det.velocity.x])
                )
                det = RazelRrtDetection(
                    data=razelrrt,
                    noise=np.array([1, 1e-2, 5e-2, 10]) ** 2,
                    source_identifier=name_3d,
                    reference=reference,
                    obj_type=det.obj_type,
                )
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


def make_3d_tracking_data(dt=0.1, n_frames=50, n_targs=4):
    ext = [(-40, 40), (-40, 40), (-2, 2)]
    detections = [
        [
            (
                np.random.uniform(low=ext[0][0], high=ext[0][1]),
                3 * np.random.randn(),
                np.random.uniform(low=ext[1][0], high=ext[1][1]),
                3 * np.random.randn(),
                np.random.uniform(low=ext[2][0], high=ext[2][1]),
                0.1 * np.random.randn(),
                np.random.uniform(low=1, high=4),
                np.random.uniform(low=1, high=4),
                np.random.uniform(low=1, high=4),
            )
            for _ in range(n_targs)
        ]
    ]

    # propagate objects in time
    for i in range(1, n_frames):
        detections.append(
            [
                (
                    detections[i - 1][j][0] + detections[i - 1][j][1] * dt,
                    detections[i - 1][j][1] + 0.1 * np.random.randn(),
                    detections[i - 1][j][2] + detections[i - 1][j][3] * dt,
                    detections[i - 1][j][3] + 0.1 * np.random.randn(),
                    detections[i - 1][j][4] + detections[i - 1][j][5] * dt,
                    detections[i - 1][j][5] + 0.1 * np.random.randn(),
                    detections[i - 1][j][6] + 0.1 * np.random.randn(),
                    detections[i - 1][j][7] + 0.1 * np.random.randn(),
                    detections[i - 1][j][8] + 0.1 * np.random.randn(),
                )
                for j in range(n_targs)
            ]
        )

    # make detection objects
    dets_3d_all = [
        DataContainer(
            frame=frame,
            timestamp=frame * dt,
            data=[
                BoxDetection(
                    source_identifier="detector-3d",
                    box=Box3D(
                        position=Position([det[0], det[2], det[4]], GlobalOrigin3D),
                        attitude=Attitude(
                            transform_orientation(
                                [0, 0, np.arctan2(det[1], det[3])], "euler", "quat"
                            ),
                            GlobalOrigin3D,
                        ),
                        hwl=det[6:9],
                    ),
                    noise=np.array([1, 1, 1, 0.5, 0.5, 0.5]) ** 2,
                    reference=GlobalOrigin3D,
                    obj_type="car",
                )
                for det in dets
            ],
            source_identifier=name_3d,
        )
        for frame, dets in enumerate(detections)
    ]
    return dets_3d_all


def make_2d_tracking_data(dt=0.1, n_frames=50, n_targs=4):
    # initialize objects
    ref_camera = ReferenceFrame(
        x=np.array([0.27, 0.06, 1.65]), q=q_stan_to_cam, reference=GlobalOrigin3D
    )
    P_cam = np.array(
        [
            [
                7.215377000000e02,
                0.000000000000e00,
                6.095593000000e02,
                4.485728000000e01,
            ],
            [
                0.000000000000e00,
                7.21537000000e02,
                1.728540000000e02,
                2.163791000000e-01,
            ],
            [
                0.000000000000e00,
                0.000000000000e00,
                1.000000000000e00,
                2.745884000000e-03,
            ],
        ]
    )
    img_shape = (375, 1242, 3)
    cam_calib = CameraCalibration(ref_camera, P_cam, img_shape)
    height, width = cam_calib.img_shape[:2]
    detections = [
        [
            (
                np.random.randint(low=20, high=width - 20),
                10 * np.random.randn(),
                np.random.randint(low=20, high=height - 20),
                10 * np.random.randn(),
                np.random.randint(low=100, high=300),
                np.random.randint(low=100, high=200),
            )
            for _ in range(n_targs)
        ]
    ]

    # propagate objects in time
    for i in range(1, n_frames):
        detections.append(
            [
                (
                    detections[i - 1][j][0] + detections[i - 1][j][1] * dt,
                    detections[i - 1][j][1] + 0.1 * np.random.randn(),
                    detections[i - 1][j][2] + detections[i - 1][j][3] * dt,
                    detections[i - 1][j][3] + 0.1 * np.random.randn(),
                    detections[i - 1][j][4] + 1 * np.random.randn(),
                    detections[i - 1][j][5] + 1 * np.random.randn(),
                )
                for j in range(n_targs)
            ]
        )

    # make detection objects
    dets_2d_all = [
        DataContainer(
            frame=frame,
            timestamp=frame * dt,
            data=[
                BoxDetection(
                    data=Box2D(
                        box2d=[det[0], det[2], det[0] + det[4], det[2] + det[5]],
                        calibration=cam_calib,
                    ),
                    noise=np.array([10, 10, 5, 5]) ** 2,
                    source_identifier="detector-2d",
                    reference=GlobalOrigin3D,
                    obj_type="car",
                )
                for det in dets
            ],
            source_identifier=name_2d,
        )
        for frame, dets in enumerate(detections)
    ]
    return dets_2d_all, cam_calib


def make_kitti_2d_3d_tracking_data(dt=0.1, n_frames=10, n_targs=4):
    dets_3d_all = make_kitti_tracking_data(dt, n_frames, n_targs=n_targs)
    dets_2d_all = []
    reference = camera_calib.reference
    for i, dets_3d in enumerate(dets_3d_all):
        d_2d = [
            BoxDetection(
                data=d.box.project_to_2d_bbox(camera_calib),
                noise=np.array([10, 10, 5, 5]) ** 2,
                source_identifier="detector-2d",
                reference=reference,
                obj_type=d.obj_type,
            )
            for d in dets_3d
            if box_in_fov(d.box, camera_calib)
        ]
        dets_2d_all.append(
            DataContainer(i, dets_3d.timestamp, d_2d, source_identifier=name_2d)
        )
        assert isinstance(dets_3d, DataContainer)
    return dets_2d_all, dets_3d_all


def run_tracker(tracker, det_type, dt=0.25):
    platform = GlobalOrigin3D
    dets_all = make_kitti_tracking_data(
        dt=dt, n_frames=20, n_targs=4, det_type=det_type
    )
    for frame, dets in enumerate(dets_all):
        tracks = tracker(
            detections=dets,
            platform=platform,
        )
    assert len(tracks) == len(dets_all[-1])
    for i, trk in enumerate(tracks):
        for det in dets_all[-1]:
            if hasattr(det, "xyz"):
                if (trk.position - det.xyz).norm() < 4:
                    break
            elif hasattr(det, "xy"):
                if (trk.position - det.xy).norm() < 4:
                    break
            elif hasattr(det, "box"):
                if np.linalg.norm(trk.box.center.x - det.box.center.x) < 5:
                    break
        else:
            raise RuntimeError(trk)
