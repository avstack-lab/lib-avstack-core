# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-04
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-10-22
# @Description:
"""

"""

import numpy as np

from avstack import environment, geometry, modules


class VehicleEgoStack:
    def __init__(self, t_init, ego_init, *args, verbose=False, **kwargs):
        self.sensor_IDs = None
        self.control_mode = "autopilot"
        self.timestamp = None
        self.frame = None
        self.initialized = False
        self.environment = environment.EnvironmentState()
        self.is_passthrough = False
        self.verbose = verbose
        self._initialize_modules(t_init, ego_init, *args, **kwargs)

    def _initialize_modules(self, t_init, ego_init):
        raise NotImplementedError

    def tick(self, frame, timestamp, data_manager, ground_truth=None, **kwargs):
        # --- pre modules
        self.frame = frame
        self.timestamp = timestamp

        # -- modules
        control = self._tick_modules(
            frame=frame,
            timestamp=timestamp,
            data_manager=data_manager,
            ground_truth=ground_truth,
            **kwargs
        )

        # --- post-modules
        return control

    def _tick_modules(self, data_manager):
        raise NotImplementedError


# ==========================================================
# AUTOPILOT AV'S
# ==========================================================


class PassthroughAutopilotVehicle(VehicleEgoStack):
    """Vehicle should be set to autopilot externally"""

    def _initialize_modules(self, *args, **kwargs):
        self.is_passthrough = True

    def set_destination(self, *args, **kwargs):
        raise RuntimeError("Cannot set destination with autopilot AV")

    def _tick_modules(self, *args, **kwargs):
        return None


class GroundTruthBehaviorAgent(VehicleEgoStack):
    """Based on the provided carla example files"""

    def _initialize_modules(self, t_init, ego_init, map_data, *args, **kwargs):
        self.planning = modules.planning.vehicle.MapBasedPlanningAndControl(
            ego_init, map_data
        )

    def set_destination(self, destination, coordinates="avstack"):
        dest_true = self.planning.set_destination(destination, coordinates)
        return dest_true

    def _tick_modules(
        self, frame, timestamp, data_manager, ground_truth, *args, **kwargs
    ):
        ego_state = ground_truth.ego_state
        environment = ground_truth.environment
        objects_3d = ground_truth.objects
        objects_2d = []
        ctrl = self.planning(ego_state, environment, objects_3d, objects_2d)
        return ctrl


# ==========================================================
# TEST AV'S
# ==========================================================


class GoStraightEgo(VehicleEgoStack):
    """A silly ego vehicle that just goes straight"""

    def _initialize_modules(self, *args, **kwargs):
        self.plan = modules.planning.WaypointPlan()
        self.planning = modules.planning.vehicle.GoStraightPlanner()
        ctrl_lat = {"K_P": 1.2, "K_D": 0.1, "K_I": 0.02}
        ctrl_lon = {"K_P": 1.0, "K_D": 0.2, "K_I": 0.2}
        self.control = modules.control.vehicle.VehiclePIDController(ctrl_lat, ctrl_lon)

    def set_destination(self, *args, **kwargs):
        print("This ego does not take a destination")

    def _tick_modules(
        self, frame, timestamp, data_manager, ground_truth, *args, **kwargs
    ):
        ego_state = ground_truth.ego_state
        self.plan = self.planning(self.plan, ego_state)
        ctrl = self.control(ego_state, self.plan)
        return ctrl


class AutopilotWithCameraPerception(VehicleEgoStack):
    """An autopilot vehicle that runs camera perception for fun"""

    def _initialize_modules(
        self, *args, camera_perception="fasterrcnn", dataset="nuscenes", **kwargs
    ):
        self.is_passthrough = True
        self.perception = {
            "object_2d": modules.perception.object2dfv.MMDetObjectDetector2D(
                model=camera_perception, dataset=dataset, **kwargs
            ),
        }

    def set_destination(self, *args, **kwargs):
        raise RuntimeError("Cannot set destination with autopilot AV")

    def _tick_modules(
        self, frame, timestamp, data_manager, ground_truth, *args, **kwargs
    ):
        if data_manager.has_data("camera-0"):
            dets_2d = self.perception["object_2d"](
                data_manager.pop("camera-0"),
                frame=frame,
                identifier="camera_objects_2d",
            )
            if self.verbose:
                print("Detected {} objects with our camera     ".format(len(dets_2d)))
        return None


# ==========================================================
# LEVEL 2 AV'S
# ==========================================================


class Level2GtPerceptionGtLocalization(VehicleEgoStack):
    def _initialize_modules(self, t_init, ego_init, *args, **kwargs):
        self.localization = modules.localization.GroundTruthLocalizer(
            t_init=t_init, ego_init=ego_init, rate=10
        )
        self.perception = {
            "object_3d": modules.perception.object3d.GroundTruth3DObjectDetector(),
            "lane_lines": modules.perception.lanelines.GroundTruthLaneLineDetector(),
        }
        self.tracking = modules.tracking.tracker3d.Ab3dmotTracker(framerate=10)
        self.prediction = modules.prediction.KinematicPrediction(
            dt_pred=0.1, t_pred_forward=3
        )
        self.plan = modules.planning.WaypointPlan()
        self.planning = modules.planning.vehicle.AdaptiveCruiseControl()
        ctrl_lat = {"K_P": 1.2, "K_D": 0.2, "K_I": 0.05}
        ctrl_lon = {"K_P": 1.0, "K_D": 0.2, "K_I": 0.2}
        self.control = modules.control.vehicle.VehiclePIDController(ctrl_lat, ctrl_lon)

    def _tick_modules(
        self, frame, timestamp, data_manager, ground_truth, *args, **kwargs
    ):
        ego_state = self.localization(timestamp, ground_truth)
        objects_3d = self.perception["object_3d"](
            ground_truth, frame=frame, identifier="objects_3d"
        )
        objects_2d = []
        # filter to objects within 30 m
        objects_3d = [
            obj for obj in objects_3d if ego_state.position.distance(obj.box.t) < 30
        ]
        objects_3d = self.tracking(
            t=timestamp,
            frame=frame,
            detections=objects_3d,
            platform=ego_state,
            identifier="tracker-0",
        )
        lanes = self.perception["lane_lines"](
            ground_truth, frame=frame, identifier="lane_lines"
        )
        # preds_3d = self.prediction(objects_3d, frame=frame)
        self.plan = self.planning(
            self.plan, ego_state, self.environment, objects_3d, objects_2d, lanes
        )
        ctrl = self.control(ego_state, self.plan)
        return ctrl

    def set_destination(self, *args, **kwargs):
        print("Ignoring destination...this type of vehicle does not take a destination")


class Level2LidarBasedVehicle(VehicleEgoStack):
    def _initialize_modules(self, t_init, ego_init, *args, **kwargs):
        integrity = modules.localization.integrity.Chi2Integrity(p_thresh=0.95)
        self.localization = modules.localization.BasicGpsKinematicKalmanLocalizer(
            t_init, ego_init, rate=10, integrity=integrity
        )
        self.perception = {
            "object_3d": modules.perception.object3d.MMDetObjectDetector3D(
                model="pointpillars", dataset="nuscenes"
            ),
            "lane_lines": modules.perception.lanelines.GroundTruthLaneLineDetector(),
        }
        self.tracking = modules.tracking.tracker3d.Ab3dmotTracker(framerate=10)
        self.prediction = modules.prediction.KinematicPrediction(
            dt_pred=0.1, t_pred_forward=3
        )
        self.plan = modules.planning.WaypointPlan()
        self.planning = modules.planning.vehicle.AdaptiveCruiseControl()
        ctrl_lat = {"K_P": 1.2, "K_D": 0.2, "K_I": 0.05}
        ctrl_lon = {"K_P": 1.0, "K_D": 0.2, "K_I": 0.2}
        self.control = modules.control.vehicle.VehiclePIDController(ctrl_lat, ctrl_lon)

    def _tick_modules(
        self, frame, timestamp, data_manager, ground_truth, *args, **kwargs
    ):
        ego_state = self.localization(timestamp, data_manager.pop("gnss-0"))
        # TODO: handle the case when there isn't data yet -- should NOT be a "missed detection"
        objects_3d = self.perception["object_3d"](
            data_manager.pop("lidar-0"), frame=frame, identifier="objects_3d"
        )
        objects_2d = []
        objects_3d = self.tracking(
            t=timestamp,
            frame=frame,
            detections=objects_3d,
            platform=ego_state,
            identifier="tracker-0",
        )
        lanes = self.perception["lane_lines"](
            ground_truth, frame=frame, identifier="lane_lines"
        )
        # preds_3d = self.prediction(objects_3d, frame=frame)
        self.plan = self.planning(
            self.plan, ego_state, self.environment, objects_3d, objects_2d, lanes
        )
        ctrl = self.control(ego_state, self.plan)
        return ctrl

    def set_destination(self, *args, **kwargs):
        print("Ignoring destination...this type of vehicle does not take a destination")


# ==========================================================
# ANALYSIS AV'S --> perception, tracking, prediction
# ==========================================================


class LidarPerceptionAndTrackingVehicle(VehicleEgoStack):
    def _initialize_modules(
        self,
        *args,
        framerate=10,
        lidar_perception="pointpillars",
        tracking="basic-box-tracker",
        dataset="kitti",
        **kwargs
    ):
        """Initialize modules"""
        self.perception = {
            "object_3d": modules.perception.object3d.MMDetObjectDetector3D(
                model=lidar_perception, dataset=dataset, **kwargs
            )
        }
        self.tracking = init_tracking(tracking, framerate, **kwargs)
        self.prediction = modules.prediction.KinematicPrediction(
            dt_pred=1.0 / framerate, t_pred_forward=3, **kwargs
        )

    def _tick_modules(self, frame, timestamp, data_manager, *args, **kwargs):
        dets_3d = self.perception["object_3d"](
            data_manager.pop("lidar-0"), frame=frame, identifier="lidar_objects_3d"
        )
        tracks_3d = self.tracking(
            t=timestamp,
            frame=frame,
            detections=dets_3d,
            platform=geometry.GlobalOrigin3D,
            identifier="tracker-0",
        )  # TODO: check platform
        predictions = self.prediction(tracks_3d, frame=frame)
        return tracks_3d, {"object_3d": dets_3d, "predictions": predictions}


class LidarCollabPerceptionAndTrackingVehicle(VehicleEgoStack):
    def _initialize_modules(
        self,
        *args,
        framerate=10,
        lidar_perception="pointpillars",
        tracking="basic-box-tracker",
        dataset="kitti",
        **kwargs
    ):
        """Initialize modules"""
        self.perception = {
            "object_3d": modules.perception.object3d.MMDetObjectDetector3D(
                model=lidar_perception, dataset=dataset, **kwargs
            )
        }
        self.tracking = init_tracking(tracking, framerate, **kwargs)
        self.prediction = modules.prediction.KinematicPrediction(
            dt_pred=0.5, t_pred_forward=2, **kwargs
        )

    def _tick_modules(
        self, frame, timestamp, data_manager, d_thresh=90, *args, **kwargs
    ):
        dets_3d = self.perception["object_3d"](
            data_manager.pop("lidar-0"), frame=frame, identifier="lidar_objects_3d"
        )
        n_collab = 0
        dets_3d = {"lidar_3d": dets_3d}
        for name in data_manager.data_names():
            if "collaborative" in name:
                if data_manager.has_data(name):
                    dets_collab = data_manager.pop(name)
                    # -- filter out ones far away
                    dets_collab_keep = []
                    for det in dets_collab:
                        det.box.change_origin(geometry.GlobalOrigin3D)
                        if det.box.t.norm() <= d_thresh:
                            dets_collab_keep.append(det)
                    n_collab += len(dets_collab_keep)
                    if len(dets_collab_keep) > 0:
                        dets_3d[name] = dets_collab_keep
        if self.verbose:
            print("Added {} collab detections".format(n_collab))
        tracks_3d = self.tracking(
            t=timestamp,
            frame=frame,
            detections=dets_3d,
            platform=geometry.GlobalOrigin3D,
            identifier="tracker-0",
        )  # TODO: check platform
        predictions = self.prediction(tracks_3d, frame=frame)
        return tracks_3d, {"object_3d": dets_3d, "predictions": predictions}


class LidarCameraPerceptionAndTrackingVehicle(VehicleEgoStack):
    def _initialize_modules(
        self,
        *args,
        framerate=10,
        lidar_perception="pointpillars",
        camera_perception="fasterrcnn",
        tracking="basic-box-tracker-fusion-3stage",
        dataset="kitti",
        **kwargs
    ):
        """Initialize modules"""
        self.perception = {
            "object_3d": modules.perception.object3d.MMDetObjectDetector3D(
                model=lidar_perception, dataset=dataset, **kwargs
            ),
            "object_2d": modules.perception.object2dfv.MMDetObjectDetector2D(
                model=camera_perception, dataset=dataset, **kwargs
            ),
        }
        self.tracking = init_tracking(tracking, framerate, **kwargs)
        self.prediction = modules.prediction.KinematicPrediction(
            dt_pred=1.0 / framerate, t_pred_forward=3, **kwargs
        )

    def _tick_modules(self, frame, timestamp, data_manager, *args, **kwargs):
        try:
            img = data_manager.pop("image-0")
        except KeyError as e:
            img = data_manager.pop("image-2")
        dets_2d = self.perception["object_2d"](
            img, frame=frame, identifier="camera_objects_2d"
        )
        dets_3d = self.perception["object_3d"](
            data_manager.pop("lidar-0"), frame=frame, identifier="lidar_objects_3d"
        )
        tracks_3d = self.tracking(
            t=timestamp,
            frame=frame,
            detections={"2d": dets_2d, "3d": dets_3d},
            platform=geometry.GlobalOrigin3D,  # TODO: check this
            identifier="tracker-0",
        )
        predictions = self.prediction(tracks_3d, frame=frame)
        return tracks_3d, {
            "object_3d": dets_3d,
            "object_2d": dets_2d,
            "predictions": predictions,
        }


class LidarCamera3DFusionVehicle(VehicleEgoStack):
    def _initialize_modules(
        self,
        *args,
        framerate=10,
        lidar_perception="pointpillars",
        camera_perception="smoke",
        lidar_tracking="basic-box-tracker",
        camera_tracking="basic-box-tracker",
        fusion="boxtrack-to-boxtrack",
        dataset="kitti",
        **kwargs
    ):
        """Initialize modules"""
        self.perception = {
            "object_3d_lid": modules.perception.object3d.MMDetObjectDetector3D(
                model=lidar_perception, dataset=dataset, **kwargs
            ),
            "object_3d_cam": modules.perception.object3d.MMDetObjectDetector3D(
                model=camera_perception, dataset=dataset, **kwargs
            ),
        }
        self.tracking = {
            "lidar": init_tracking(lidar_tracking, framerate, **kwargs),
            "camera": init_tracking(camera_tracking, framerate, **kwargs),
        }
        if fusion == "boxtrack-to-boxtrack":
            self.fusion = modules.fusion.BoxTrackToBoxTrackFusion3D(
                association="IoU", assignment="gnn", algorithm="CI", **kwargs
            )
        else:
            raise NotImplementedError(fusion)
        self.prediction = modules.prediction.KinematicPrediction(
            dt_pred=1.0 / framerate, t_pred_forward=3, **kwargs
        )

    def _tick_modules(self, frame, timestamp, data_manager, *args, **kwargs):
        try:
            img = data_manager.pop("image-0")
        except KeyError as e:
            img = data_manager.pop("image-2")
        dets_3d_cam = self.perception["object_3d_cam"](
            img, frame=frame, identifier="camera_objects_3d"
        )
        dets_3d_lid = self.perception["object_3d_lid"](
            data_manager.pop("lidar-0"), frame=frame, identifier="lidar_objects_3d"
        )
        # -- put in nominal origin due to the fusion module
        for dets in [dets_3d_cam, dets_3d_lid]:
            for det in dets:
                det.change_origin(geometry.GlobalOrigin3D)
        tracks_camera = self.tracking["camera"](
            dets_3d_cam, frame=frame, identifier="tracker-0"
        )
        tracks_lidar = self.tracking["lidar"](
            timestamp=timestamp,
            frame=frame,
            detections=dets_3d_lid,
            platform=geometry.GlobalOrigin3D,
            identifier="tracker-1",
        )
        tracks_fused = self.fusion(tracks_camera, tracks_lidar, frame=frame)
        predictions = self.prediction(tracks_fused, frame=frame)

        return tracks_fused, {
            "object_3d": {"lidar": dets_3d_lid, "camera": dets_3d_cam},
            "tracks": {"lidar": tracks_lidar, "camera": tracks_camera},
            "predictions": predictions,
        }


# ==========================================================
# UTILITIES
# ==========================================================


def init_perception(model, dataset, **kwargs):
    if model.lower() in ["pointpillars", "second", "smoke", "pgd"]:
        M = modules.perception.object3d.MMDetObjectDetector3D(
            model=model, dataset=dataset, **kwargs
        )
    elif model.lower() in ["fasterrcnn", "yolo"]:
        M = modules.perception.object2d.MMDetObjectDetector2D(
            model=model, dataset=dataset, **kwargs
        )
    else:
        raise NotImplementedError(model)
    return M


def init_tracking(algorithm, framerate, **kwargs):
    if algorithm.lower() == "ab3dmot":
        T = modules.tracking.tracker3d.Ab3dmotTracker(framerate=framerate, **kwargs)
    elif algorithm.lower() == "basic-box-tracker":
        T = modules.tracking.tracker3d.BasicBoxTracker3D(framerate=framerate, **kwargs)
    elif algorithm.lower() == "basic-box-tracker-fusion-3stage":
        T = modules.tracking.tracker3d.BasicBoxTrackerFusion3Stage(
            framerate=framerate, **kwargs
        )
    elif "eagermot" in algorithm.lower():
        plus = "+" in algorithm
        n_box_confirmed = (
            int(algorithm.split("+")[1].split("_")[0])
            if plus and "_" in algorithm
            else 0
        )
        n_joint_coast = (
            int(algorithm.split("+")[1].split("_")[1])
            if plus and "_" in algorithm
            else np.inf
        )
        T = modules.tracking.tracker3d.EagermotTracker(
            framerate, plus, n_box_confirmed, n_joint_coast, **kwargs
        )
    else:
        raise NotImplementedError(algorithm)
    return T


def init_fusion(algorithm, **kwargs):
    if algorithm == "boxtrack-to-boxtrack":
        F = modules.fusion.BoxTrackToBoxTrackFusion3D(
            association="IoU", assignment="gnn", algorithm="CI", **kwargs
        )
    else:
        raise NotImplementedError(algorithm)
    return F
