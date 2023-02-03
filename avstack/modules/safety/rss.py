# -*- coding: utf-8 -*-
# @Author: spencer@primus
# @Date:   2022-06-15
# @Last Modified by:   spencer@primus
# @Last Modified time: 2022-09-16


import itertools
import math

import numpy as np

from avstack.geometry import NominalOriginStandard
from avstack.geometry import transformations as tforms


try:
    import ad_rss as ad
except ModuleNotFoundError as e:
    print("Cannot import rss library")
    use_rss = False
else:
    use_rss = True


def get_pedestrian_dynamics():
    pedestrian_dynamics = ad.rss.world.RssDynamics()
    pedestrian_dynamics.alphaLon.accelMax = 2.0
    pedestrian_dynamics.alphaLon.brakeMax = -2.0
    pedestrian_dynamics.alphaLon.brakeMin = -2.0
    pedestrian_dynamics.alphaLon.brakeMinCorrect = 0.01
    pedestrian_dynamics.alphaLat.accelMax = 0.001
    pedestrian_dynamics.alphaLat.brakeMin = -0.001
    pedestrian_dynamics.lateralFluctuationMargin = 0.1
    pedestrian_dynamics.responseTime = 0.8
    pedestrian_dynamics.maxSpeedOnAcceleration = 10
    pedestrian_dynamics.unstructuredSettings.pedestrianTurningRadius = 2.0
    pedestrian_dynamics.unstructuredSettings.driveAwayMaxAngle = 2.4
    pedestrian_dynamics.unstructuredSettings.pedestrianContinueForwardIntermediateHeadingChangeRatioSteps = (
        3
    )
    pedestrian_dynamics.unstructuredSettings.pedestrianContinueForwardIntermediateAccelerationSteps = (
        0
    )
    pedestrian_dynamics.unstructuredSettings.pedestrianBrakeIntermediateAccelerationSteps = (
        3
    )
    pedestrian_dynamics.unstructuredSettings.pedestrianFrontIntermediateHeadingChangeRatioSteps = (
        4
    )
    pedestrian_dynamics.unstructuredSettings.pedestrianBackIntermediateHeadingChangeRatioSteps = (
        0
    )

    # not used:
    pedestrian_dynamics.unstructuredSettings.vehicleYawRateChange = 1.3
    pedestrian_dynamics.unstructuredSettings.vehicleMinRadius = 3.5
    pedestrian_dynamics.unstructuredSettings.vehicleTrajectoryCalculationStep = 0.2
    pedestrian_dynamics.unstructuredSettings.vehicleFrontIntermediateYawRateChangeRatioSteps = (
        4
    )
    pedestrian_dynamics.unstructuredSettings.vehicleBackIntermediateYawRateChangeRatioSteps = (
        0
    )
    pedestrian_dynamics.unstructuredSettings.vehicleContinueForwardIntermediateAccelerationSteps = (
        3
    )
    pedestrian_dynamics.unstructuredSettings.vehicleBrakeIntermediateAccelerationSteps = (
        3
    )
    return pedestrian_dynamics


def get_vehicle_dynamics():
    ego_dynamics = ad.rss.world.RssDynamics()
    ego_dynamics.alphaLon.accelMax = 6
    ego_dynamics.alphaLon.brakeMax = -6
    ego_dynamics.alphaLon.brakeMin = -2
    ego_dynamics.alphaLon.brakeMinCorrect = -1
    ego_dynamics.alphaLat.accelMax = 0.5
    ego_dynamics.alphaLat.brakeMin = -0.8
    ego_dynamics.lateralFluctuationMargin = 0.1
    ego_dynamics.responseTime = 1
    ego_dynamics.maxSpeedOnAcceleration = 50
    ego_dynamics.unstructuredSettings.pedestrianTurningRadius = 2.0
    ego_dynamics.unstructuredSettings.driveAwayMaxAngle = 2.4
    ego_dynamics.unstructuredSettings.vehicleYawRateChange = 1.3
    ego_dynamics.unstructuredSettings.vehicleMinRadius = 3.5
    ego_dynamics.unstructuredSettings.vehicleTrajectoryCalculationStep = 0.2
    ego_dynamics.unstructuredSettings.vehicleFrontIntermediateYawRateChangeRatioSteps = (
        4
    )
    ego_dynamics.unstructuredSettings.vehicleBackIntermediateYawRateChangeRatioSteps = 0
    ego_dynamics.unstructuredSettings.vehicleContinueForwardIntermediateAccelerationSteps = (
        3
    )
    ego_dynamics.unstructuredSettings.vehicleBrakeIntermediateAccelerationSteps = 3
    ego_dynamics.unstructuredSettings.pedestrianTurningRadius = 2.0
    ego_dynamics.unstructuredSettings.pedestrianContinueForwardIntermediateHeadingChangeRatioSteps = (
        3
    )
    ego_dynamics.unstructuredSettings.pedestrianContinueForwardIntermediateAccelerationSteps = (
        0
    )
    ego_dynamics.unstructuredSettings.pedestrianBrakeIntermediateAccelerationSteps = 3
    ego_dynamics.unstructuredSettings.pedestrianFrontIntermediateHeadingChangeRatioSteps = (
        4
    )
    ego_dynamics.unstructuredSettings.pedestrianBackIntermediateHeadingChangeRatioSteps = (
        0
    )
    return ego_dynamics


class RoadSegment:
    def __init__(self, road_length, road_width):
        self.road_length = road_length
        self.road_width = road_width

    @property
    def model(self):
        road_segment = ad.rss.world.RoadSegment()
        lane_segment = ad.rss.world.LaneSegment()
        lane_segment.id = 0
        lane_segment.length.minimum = ad.physics.Distance(self.road_length)
        lane_segment.length.maximum = ad.physics.Distance(self.road_length)
        lane_segment.width.minimum = ad.physics.Distance(self.road_width)
        lane_segment.width.maximum = ad.physics.Distance(self.road_width)
        lane_segment.type = ad.rss.world.LaneSegmentType.Normal
        lane_segment.drivingDirection = ad.rss.world.LaneDrivingDirection.Positive
        road_segment.append(lane_segment)
        return road_segment


class RssSceneSafetyMetric:
    def __init__(self, metrics):
        self.metrics = metrics

    @property
    def safe(self):
        return np.all([m.safe for m in self.metrics])

    @property
    def dangerous_objects(self):
        d_objs_all = []
        for metric in self.metrics:
            d_objs_all.extend(metric.dangerous_objects)
        return d_objs_all

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        base_str = (
            f"Aggregate RSS Safety Metric over {len(self.metrics)} Metrics\n"
            f'    scene is {"safe" if self.safe else "unsafe"}\n'
        )
        if not self.safe:
            base_str += f"    objects {self.dangerous_objects} are dangerous"
        return base_str


class RssSafetyMetric:
    def __init__(self, situation, situation_type, state, response, safe):
        self.situation = situation
        self.situation_type = situation_type
        self.state = state
        self.response = response
        self.safe = safe

    @property
    def dangerous_objects(self):
        return self.response.dangerousObjects if self.response is not None else []

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        base_str = (
            f"Singular RSS Safety Metric:\n"
            + f'   {self.situation_type} situation is {"safe" if self.safe else "unsafe"}\n'
        )
        if not self.safe:
            n_dangerous = len(self.dangerous_objects)
            base_str += f'   there {"is" if n_dangerous==1 else "are"} {n_dangerous} dangerous object{"" if n_dangerous==1 else "s"}\n'
            base_str += f"   proposed_response is {self.response}"
        return base_str


BlankSafetyMetric = RssSafetyMetric(None, None, None, None, True)


class RssEvaluator:
    def __init__(self, ego_dynamics=None, car_dynamics=None, ped_dynamics=None):
        self.ego_dynamics = (
            ego_dynamics if ego_dynamics is not None else get_vehicle_dynamics()
        )
        self.car_dynamics = (
            car_dynamics if car_dynamics is not None else get_vehicle_dynamics()
        )
        self.ped_dynamics = (
            ped_dynamics if ped_dynamics is not None else get_pedestrian_dynamics()
        )
        self.road_segment = DEFAULT_ROAD_SEGMENT

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "RssEvaluator"

    def __call__(self, ego, objects, ego_relative=True, verbose=False):
        """Make a call to the RSS evaluator to evaluate the scene

        :ego_relative
                True: only evaluate ego + obj for all objects
                False: evaluate all pairwise combinations of [ego, obj1, obj2,...]

        either ego and objects are both global or both local
        """

        # -- ensure unique ID's
        IDs = [obj.ID for obj in objects] + [ego.ID]
        if not len(np.unique(IDs)) == len(IDs):
            raise ValueError(f"IDs cannot be the same, {IDs}")

        # -- get scene origin
        p_bl, p_tr = self._get_scene_bounds(ego, objects)

        # -- get the rss scene
        rss_metrics = []
        all_scenes = []
        all_worlds = []
        if ego_relative:
            rss_ego = self._rss_ego(ego)
            for obj in objects:
                rss_object = self._rss_object(ego, obj)
                if isinstance(rss_object, str) and (rss_object == "ignore"):
                    continue
                elif rss_object is None:
                    raise RuntimeError("Not supposed to be none...")

                # -- construct the scene
                rss_scene = ad.rss.world.Scene()
                rss_scene.egoVehicle = rss_ego
                rss_scene.egoVehicleRssDynamics = self.ego_dynamics
                rss_scene.egoVehicleRoad.append(self.road_segment.model)
                rss_scene.object = rss_object
                rss_scene.objectRssDynamics = self.car_dynamics

                # -- determine scene type based on longitudinal velocity
                vel_rel = obj.velocity - ego.velocity.vector
                vel_rel_global = (
                    vel_rel.vector_global
                )  # this should be in the ego frame.....
                if vel_rel_global[0] >= 0:
                    rss_scene.situationType = (
                        ad.rss.situation.SituationType.SameDirection
                    )
                else:
                    rss_scene.situationType = (
                        ad.rss.situation.SituationType.SameDirection
                    )
                #                     rss_scene.situationType = ad.rss.situation.SituationType.OppositeDirection
                situation_type = rss_scene.situationType

                # -- construct the world model
                world_model = ad.rss.world.WorldModel()
                world_model.defaultEgoVehicleRssDynamics = self.ego_dynamics
                world_model.timeIndex = 1
                world_model.scenes.append(rss_scene)
                all_scenes.append(rss_scene)
                all_worlds.append(world_model)
                rss_metric = self._evaluate(
                    world_model, situation_type, verbose=verbose
                )
                rss_metrics.append(rss_metric)

            self.all_scenes = all_scenes
            self.all_worlds = all_worlds
        else:
            raise NotImplementedError("Non-ego-relative is not implemented yet")
        return RssSceneSafetyMetric(rss_metrics)

    def _get_scene_bounds(self, ego, objects):
        p_bl = np.inf * np.ones((2,))
        p_tr = -np.inf * np.ones((2,))
        pos_ego_bev = ego.position.vector_global[:2]
        for obj in itertools.chain([ego], objects):
            pos_bev_bl = (
                obj.position.vector_global[:2]
                - pos_ego_bev
                - np.array([obj.box.l / 2, obj.box.w / 2])
            )
            pos_bev_tr = (
                obj.position.vector_global[:2]
                - pos_ego_bev
                + np.array([obj.box.l / 2, obj.box.w / 2])
            )
            p_bl = np.minimum(p_bl, pos_bev_bl)
            p_tr = np.maximum(p_tr, pos_bev_tr)
        return p_bl, p_tr

    def _evaluate(self, world_model, situation_type, verbose=False):
        # -- get functions
        rss_response_resolving = ad.rss.core.RssResponseResolving()
        rss_situation_checking = ad.rss.core.RssSituationChecking()
        rss_sitation_extraction = ad.rss.core.RssSituationExtraction()
        rss_situation_snapshot = ad.rss.situation.SituationSnapshot()
        rss_state_snapshot = ad.rss.state.RssStateSnapshot()
        rss_proper_response = ad.rss.state.ProperResponse()

        # -- run commands
        rss_sitation_extraction.extractSituations(world_model, rss_situation_snapshot)
        rss_situation_checking.checkSituations(
            rss_situation_snapshot, rss_state_snapshot
        )
        if len(rss_situation_snapshot.situations) > 0:
            longitudinal_distance = float(
                rss_situation_snapshot.situations[
                    0
                ].relativePosition.longitudinalDistance
            )
        else:
            longitudinal_distance = np.nan
        rss_response_resolving.provideProperResponse(
            rss_state_snapshot, rss_proper_response
        )

        # -- package outputs
        if verbose:
            print("== Situation Snaphost ==")
            print(rss_situation_snapshot)
            print("=== Relative Position ==")
            print(rss_situation_snapshot.situations[0].relativePosition)

        # -- get safety
        safe = rss_proper_response.isSafe
        res = RssSafetyMetric(
            rss_situation_snapshot,
            situation_type,
            rss_state_snapshot,
            rss_proper_response,
            safe,
        )
        return res

    def _rss_ego(self, ego):
        pos_2d = ego.position.vector_global[:2]
        vel_2d = ego.velocity.vector_global[:2]
        obj_type = ad.rss.world.ObjectType.EgoVehicle
        return self._get_rss_object(ego.ID, obj_type, ego.box3d, pos_2d, vel_2d, True)

    def _rss_object(self, ego, obj):
        pos_ego = ego.position.vector_global[:2]
        pos_oth = obj.position.vector_global[:2]
        vel_ego = ego.velocity.vector_global[:2]
        vel_oth = obj.velocity.vector_global[:2]
        pos_2d_rel = pos_oth - pos_ego
        vel_2d_rel = vel_oth - vel_ego
        if obj.obj_type.lower() in [
            "car",
            "construction_vehicle",
            "none",
            "cyclist",
            "bicycle",
            "bus",
            "truck",
            "motorcycle",
        ]:
            obj_type = ad.rss.world.ObjectType.OtherVehicle
        elif obj.obj_type.lower() in ["pedestrian", "person"]:
            obj_type = ad.rss.world.ObjectType.Pedestrian
        elif obj.obj_type.lower() in [
            "barrier",
            "rider",
            "traffic_cone",
            "cone",
            "train",
        ]:
            return "ignore"
        else:
            raise NotImplementedError(obj.obj_type.lower())
        return self._get_rss_object(
            obj.ID, obj_type, obj.box3d, pos_2d_rel, vel_2d_rel, False
        )

    def _get_rss_object(
        self, object_ID, object_type, box3d, pos_2d, vel_2d, is_ego, verbose=True
    ):
        box3d.change_origin(NominalOriginStandard)
        obj = ad.rss.world.Object()
        obj.objectId = object_ID
        obj.objectType = object_type
        obj.velocity.speedLonMin = ad.physics.Speed(abs(vel_2d[0]))
        obj.velocity.speedLonMax = ad.physics.Speed(abs(vel_2d[0]))
        obj.velocity.speedLatMin = ad.physics.Speed(vel_2d[1])
        obj.velocity.speedLatMax = ad.physics.Speed(vel_2d[1])
        obj.state.yaw = ad.physics.Angle(box3d.yaw)  # NOTE: yaw is in radians
        obj.state.dimension.length = ad.physics.Distance(box3d.l)
        obj.state.dimension.width = ad.physics.Distance(box3d.w)
        obj.state.yawRate = ad.physics.AngularVelocity(0.0)
        obj.state.centerPoint.x = ad.physics.Distance(float(box3d.t[0]))
        obj.state.centerPoint.y = ad.physics.Distance(float(box3d.t[1]))
        obj.state.speed = math.sqrt(vel_2d[0] ** 2 + vel_2d[1] ** 2)
        obj.state.steeringAngle = ad.physics.Angle(0.0)

        occupied_region = ad.rss.world.OccupiedRegion()
        occupied_region.segmentId = 0
        lons = [abs(pos_2d[0] - box3d.l / 2), abs(pos_2d[0]) + box3d.l / 2]
        lats = [abs(pos_2d[1] - box3d.w / 2), abs(pos_2d[1]) + box3d.w / 2]
        lon_min = np.clip(lons[0] / self.road_segment.road_length, 0, 0.99)
        lon_max = np.clip(lons[1] / self.road_segment.road_length, 0, 0.99)
        lat_min = np.clip(lats[0] / self.road_segment.road_width, 0, 0.99)
        lat_max = np.clip(lats[1] / self.road_segment.road_width, 0, 0.99)
        if False:  # (lon_min < 0) or (lon_max >= 1) or (lat_min < 0) or (lon_max >=1):
            if verbose:
                print(
                    f"Object outside of lane field -- {lon_min:.2f}, {lon_max:.2f}, {lat_min:.2f}, {lat_max:.2f}"
                )
            return None
        else:
            occupied_region.lonRange.minimum = ad.physics.ParametricValue(lon_min)
            occupied_region.lonRange.maximum = ad.physics.ParametricValue(lon_max)
            occupied_region.latRange.minimum = ad.physics.ParametricValue(lat_min)
            occupied_region.latRange.maximum = ad.physics.ParametricValue(lat_max)
            obj.occupiedRegions.append(occupied_region)
            return obj


if use_rss:
    DEFAULT_ROAD_SEGMENT = RoadSegment(road_length=100.0, road_width=50.0)
