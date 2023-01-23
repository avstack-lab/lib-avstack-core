# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-28
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-29
# @Description:
"""

"""

from cmath import sqrt
from math import cos, sin

import numpy as np

from avstack import transformations as tforms
from avstack.geometry import Rotation, StandardCoordinates, Transform, Translation

from .base import Waypoint, WaypointPlan


class CollisionDetection:
    def __init__(self, ego, tracks):
        self.ego = ego
        self.tracks = tracks

    def collision(self, ego_p, obj_p, ID):
        ego_l = self.ego.box.l
        ego_w = self.ego.box.w
        ego_h = self.ego.box.h
        ego_yaw = self.ego.box.yaw
        obj_l = self.tracks[ID].box3d.l
        obj_w = self.tracks[ID].box3d.w
        obj_h = self.tracks[ID].box3d.h
        obj_yaw = self.tracks[ID].box3d.yaw
        if ((ego_p[0] - obj_p[0]) ** 2 + (ego_p[1] - obj_p[1]) ** 2) ** 0.5 > (
            max(ego_l, ego_w, ego_h) + max(obj_l, obj_w, obj_h)
        ):
            return False
        else:
            if ego_yaw == obj_yaw:
                return not (
                    (ego_p[0] + ego_l / 2) < (obj_p[0] - obj_l / 2)
                    or (ego_p[0] - ego_l / 2) > (obj_p[0] + obj_l / 2)
                    or (ego_p[1] + ego_w / 2) < (obj_p[1] - obj_w / 2)
                    or (ego_p[1] - ego_w / 2) > (obj_p[1] + obj_w / 2)
                )
            else:
                dis_x = obj_p[0] - ego_p[0]
                dis_y = obj_p[1] - ego_p[1]
                ego_lx = ego_l / 2 * cos(ego_yaw)
                ego_wx = ego_w / 2 * sin(ego_yaw)
                ego_ly = ego_l / 2 * sin(ego_yaw)
                ego_wy = -ego_w / 2 * cos(ego_yaw)
                obj_lx = obj_l / 2 * cos(obj_yaw)
                obj_wx = obj_w / 2 * sin(obj_yaw)
                obj_ly = obj_l / 2 * sin(obj_yaw)
                obj_wy = -obj_w / 2 * cos(obj_yaw)
                return (
                    abs(dis_x * cos(ego_yaw) + dis_y * sin(ego_yaw))
                    <= abs(obj_lx * cos(ego_yaw) + obj_ly * sin(ego_yaw))
                    + abs(obj_wx * cos(ego_yaw) + obj_wy * sin(ego_yaw))
                    + ego_l / 2
                    and abs(dis_x * sin(ego_yaw) - dis_y * cos(ego_yaw))
                    <= abs(obj_lx * sin(ego_yaw) - obj_ly * cos(ego_yaw))
                    + abs(obj_wx * sin(ego_yaw) - obj_wy * cos(ego_yaw))
                    + ego_w / 2
                    and abs(dis_x * cos(obj_yaw) + dis_y * sin(obj_yaw))
                    <= abs(ego_lx * cos(obj_yaw) + ego_ly * sin(obj_yaw))
                    + abs(ego_wx * cos(obj_yaw) + ego_wy * sin(obj_yaw))
                    + obj_l / 2
                    and abs(dis_x * sin(obj_yaw) - dis_y * cos(obj_yaw))
                    <= abs(ego_lx * sin(obj_yaw) - ego_ly * cos(obj_yaw))
                    + abs(ego_wx * sin(obj_yaw) - ego_wy * cos(obj_yaw))
                    + obj_w / 2
                )

    def collision_monitor(self, preds_ego, preds_tracks):
        collision = {}
        for i, preds_trk in enumerate(preds_tracks.values()):
            for t in preds_ego:
                ego_p = preds_ego[t].position
                obj_p = preds_trk[t].position.vector
                if self.collision(ego_p, obj_p, i) == True:
                    collision[preds_trk[t].ID] = t
                    break
        return collision


def get_object_to_follow(ego_state, objects_3d, lane_lines):
    """Find an object to follow, if one exists

    objects are already in relative frame
    """
    # -- extract from main camera
    if len(lane_lines) == 1:
        lane_lines = lane_lines[0]
    if len(lane_lines) != 2:
        return None

    # -- find track to use
    d_min = np.inf
    obj_follow = None
    for obj in objects_3d:
        rel_pos = (
            obj.position
        )  # ego_state.attitude @ (obj.position - ego_state.position)
        d_next = rel_pos.norm()
        if lane_lines[0].object_between_lanes_projected(lane_lines[1], rel_pos) and (
            rel_pos.x - ego_state.box3d.l / 2 - obj.box3d.l / 2 > 0
        ):
            if (d_next < d_min) and (d_next < 45):
                d_min = d_next
                obj_follow = obj
    return obj_follow


def determine_direction(ego_state, lane_lines, following_id, lane_id, tracks):
    # For ground_true
    if len(lane_lines) == 2:
        if abs(lane_id) == 1 or abs(lane_id) == 2:
            tracks_around = []
            for i in tracks:
                if (
                    ego_state.position.distance(tracks[i].position) < 4.5
                    and i != following_id
                ):
                    tracks_around.append(tracks[i])

            if abs(lane_id) == 1:
                diretcion = "right"
                offset = -3.5
            else:
                diretcion = "left"
                offset = 3.5

            if tracks_around == []:
                return diretcion
            else:
                for i in range(len(lane_lines)):
                    lane_lines[0][i].y += offset
                    lane_lines[1][i].y += offset
                for track_around in tracks_around:
                    if lane_lines[0].object_between_lanes_projected(
                        lane_lines[1], track_around.position - ego_state.position
                    ):
                        return None
                return diretcion
        else:
            return None
    # Interface for sensors
    else:
        raise NotImplementedError


class ChangeLine:
    def __init__(self, ego):
        self.ego = ego

    def GetDist(self):
        return None

    def GetWpt(self, ego_state, lane_lines, speed_target, offset, d_forward=5):
        forward_vec = ego_state.attitude.forward_vector
        left_vec = ego_state.attitude.left_vector

        if (len(lane_lines) > 0) and (isinstance(lane_lines[0], list)):
            lane_lines = lane_lines[0]

        if len(lane_lines) == 2:
            _, lateral_offset, yaw_offset = lane_lines[
                0
            ].compute_center_lane_and_offset(lane_lines[1])
            lateral_offset += offset
            target_loc = (
                ego_state.position + lateral_offset * left_vec + d_forward * forward_vec
            )
            R_b2way = Rotation(
                target_loc.coordinates, tforms.get_rot_yaw_matrix(-yaw_offset, "+z")
            )
            R_world2b = ego_state.attitude
            target_rot = R_b2way @ R_world2b
            target_speed = speed_target
        else:
            raise NotImplementedError
        distance = ego_state.position.distance(target_loc)
        target_point = Transform(target_rot, target_loc)
        return distance, Waypoint(target_point, speed_target)


# class AdaptiveCruiseControl():
#     def __init__(self, goal_dt=3, min_dt=2, min_d=2):
#         self.goal_dt = goal_dt
#         self.min_dt = min_dt
#         self.min_d = min_d
#         self._steps_to_goal_dt = 5

#     def __call__(self, ego, objects, speed_limit):
#         obj_follow = self._get_following_object(ego, objects)
#         target_speed = self._following_control(ego, obj_follow, speed_limit)
#         return target_speed

#     def _following_control(self, ego, obj_follow, speed_limit):
#         if obj_follow is None:
#             target_speed = speed_limit
#         else:
#             distance = ego.distance_front_to_back(obj_follow)
#             time = distance / obj_follow.speed
#             if (time <= self.min_dt) or (distance <= self.min_d):
#                 target_speed = 0.0  # slow down dramatically
#             elif time <= 0.8*self.goal_dt:
#                 target_speed =  # slow a little
#             elif 0.8*self.goal_dt <= time <= 1.2*self.goal_dt:
#                 target_speed = min(speed_limit, obj_follow.speed)  # close enough
#             else:
#                 target_speed = min(speed_limit, obj_follow.speed) + 5  # far away
#         return target_speed

#     def _get_following_object(self, ego, objects):
#         """Get which vehicle we should be following"""
#         for obj in objects:
#             raise
#         return obj


# class TailgatingMonitor():
#     def __init__(self, goal_dt=3, min_dt=2, min_d=2, angle_consider_deg=15):
#         self.goal_dt = 3
#         self.min_dt = min_dt
#         self.min_d = min_d
#         self.cangle_min = np.cosd(angle_consider_deg)

#     def __call__(self, ego, objects, speed_limit):
#         """Determine if objects are tailgating us"""
#         obj_behind = self._get_tailgate_object(ego, objects, speed_limit)

#     def _tailgating_control(self, ego, obj_behind, speed_limit):
#         raise NotImplementedError
#         # left_turn = waypoint.left_lane_marking.lane_change
#         # right_turn = waypoint.right_lane_marking.lane_change
#         # left_wpt = waypoint.get_left_lane()
#         # right_wpt = waypoint.get_right_lane()
#         # if behind_vehicle_state and self.speed < get_speed(behind_vehicle):
#         #     if (right_turn == carla.LaneChange.Right or right_turn ==
#         #             carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
#         #         new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
#         #             self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=1)
#         #         if not new_vehicle_state:
#         #             print("Tailgating, moving to the right!")
#         #             self.behavior.tailgate_counter = 200
#         #             self.set_destination(right_wpt.transform.location,
#         #                                  self.end_waypoint.transform.location, clean=True)
#         #     elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
#         #         new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
#         #             self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=-1)
#         #         if not new_vehicle_state:
#         #             print("Tailgating, moving to the left!")
#         #             self.behavior.tailgate_counter = 200
#         #             self.set_destination(left_wpt.transform.location,
#         #                                  self.end_waypoint.transform.location, clean=True)

#     def _get_tailgate_object(self, ego, objects, speed_limit):
#         back_vector = -ego.get_forward_vector()

#         for obj in objects:
#             vector_to_object = None
#             cangle = np.dot(back_vector, vector_to_object)
#             if abs(cangle) <= self.cangle_min:
#                 tailgating = True
#             else:
#                 tailgating = False
#         return obj


# class OvertakingPlanner():
#     def __init__(self, goal_dt=3.0, min_dt=2, min_d=2):
#         self.goal_dt = goal_dt
#         self.min_dt = min_dt
#         self.min_d = min_d

#     def __call__(self, ego, objects, speed_limit):
#         slow_front_objects = self._get_slow_front_objects(ego, objects, speed_limit)
#         target_speed, waypoints = self._plan_overtake(ego, slow_front_objects, speed_limit)

#     def _get_slow_front_objects(self, ego, objects, speed_limit):
#         raise NotImplementedError

#     def _plan_overtake(self, ego, slow_front_objects, speed_limit):
#         raise NotImplementedError
# def _overtake(self, location, rotation, waypoint, vehicle_list):
#     """
#     This method is in charge of overtaking behaviors.

#         :param location: current location of the agent
#         :param waypoint: current waypoint of the agent
#         :param vehicle_list: list of all the nearby vehicles
#     """

#     left_turn = waypoint.left_lane_marking.lane_change
#     right_turn = waypoint.right_lane_marking.lane_change
#     v_f = rotation.get_forward_vector()
#     d_forward = 5

#     left_wpt = waypoint.get_left_lane()
#     right_wpt = waypoint.get_right_lane()
#     if (left_turn == carla.LaneChange.Left or left_turn ==
#             carla.LaneChange.Both) and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
#         new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
#             self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=180, lane_offset=-1)
#         if not new_vehicle_state:
#             print("Overtaking to the left!")
#             self.behavior.overtake_counter = 200
#             start = carla.Location(left_wpt.transform.location.x + d_forward*v_f.x,
#                                    left_wpt.transform.location.y + d_forward*v_f.y,
#                                    left_wpt.transform.location.z + d_forward*v_f.z)
#             self.set_destination(start,
#                                  self.end_waypoint.transform.location, clean=True)
#     elif right_turn == carla.LaneChange.Right and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
#         new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
#             self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=180, lane_offset=1)
#         if not new_vehicle_state:
#             print("Overtaking to the right!")
#             self.behavior.overtake_counter = 200
#             start = carla.Location(right_wpt.transform.location.x + d_forward*v_f.x,
#                                    right_wpt.transform.location.y + d_forward*v_f.y,
#                                    right_wpt.transform.location.z + d_forward*v_f.z)
#             self.set_destination(start,
#                                  self.end_waypoint.transform.location, clean=True)
#     control = None
#     if self.behavior.tailgate_counter > 0:
#         self.behavior.tailgate_counter -= 1
#     if self.behavior.overtake_counter > 0:
#         self.behavior.overtake_counter -= 1


# class TrafficLightPlanner():
#      """
#         This method is in charge of behaviors for red lights and stops.

#         WARNING: What follows is a proxy to avoid having a car brake after running a yellow light.
#         This happens because the car is still under the influence of the semaphore,
#         even after passing it. So, the semaphore id is temporarely saved to
#         ignore it and go around this issue, until the car is near a new one.

#             :param waypoint: current waypoint of the agent
#         """

#         light_id = self.vehicle.get_traffic_light().id if self.vehicle.get_traffic_light() is not None else -1

#         if self.light_state == "Red":
#             if not waypoint.is_junction and (self.light_id_to_ignore != light_id or light_id == -1):
#                 return 1
#             elif waypoint.is_junction and light_id != -1:
#                 self.light_id_to_ignore = light_id
#         if self.light_id_to_ignore != light_id:
#             self.light_id_to_ignore = -1
#         return 0
#         # 1: Red lights and stops behavior

#         if self.traffic_light_manager(ego_vehicle_wp) != 0:
#             return self.emergency_stop()

#  def _get_trafficlight_trigger_location(self, traffic_light):  # pylint: disable=no-self-use
#         """
#         Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
#         """
#         def rotate_point(point, radians):
#             """
#             rotate a given point by a given angle
#             """
#             rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
#             rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

#             return carla.Vector3D(rotated_x, rotated_y, point.z)

#         base_transform = traffic_light.get_transform()
#         base_rot = base_transform.rotation.yaw
#         area_loc = base_transform.transform(traffic_light.trigger_volume.location)
#         area_ext = traffic_light.trigger_volume.extent

#         point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
#         point_location = area_loc + carla.Location(x=point.x, y=point.y)

#         return carla.Location(point_location.x, point_location.y, point_location.z)


#     def _is_light_red(self, lights_list):
#         """
#         Method to check if there is a red light affecting us. This version of
#         the method is compatible with both European and US style traffic lights.

#         :param lights_list: list containing TrafficLight objects
#         :return: a tuple given by (bool_flag, traffic_light), where
#                  - bool_flag is True if there is a traffic light in RED
#                    affecting us and False otherwise
#                  - traffic_light is the object itself or None if there is no
#                    red traffic light affecting us
#         """
#         ego_vehicle_location = self._vehicle.get_location()
#         ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

#         for traffic_light in lights_list:
#             object_location = self._get_trafficlight_trigger_location(traffic_light)
#             object_waypoint = self._map.get_waypoint(object_location)

#             if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
#                 continue

#             ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
#             wp_dir = object_waypoint.transform.get_forward_vector()
#             dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

#             if dot_ve_wp < 0:
#                 continue

#             if is_within_distance_ahead(object_waypoint.transform,
#                                         self._vehicle.get_transform(),
#                                         self._proximity_tlight_threshold):
#                 if traffic_light.state == carla.TrafficLightState.Red:
#                     return (True, traffic_light)

#         return (False, None)


# class CollisionAvoidancePlanner():
#     def collision_and_car_avoid_manager(self, location, rotation, waypoint):
#     """
#     This module is in charge of warning in case of a collision
#     and managing possible overtaking or tailgating chances.

#         :param location: current location of the agent
#         :param waypoint: current waypoint of the agent
#         :return vehicle_state: True if there is a vehicle nearby, False if not
#         :return vehicle: nearby vehicle
#         :return distance: distance to nearby vehicle
#     """
#     if self.information['vehicles'] == 'truth':
#         vehicle_list = self._world.get_actors().filter("*vehicle*")
#     elif self.information['vehicles'] == 'from_objects':
#         vehicle_list = [obj[1] for obj in self.objects if obj[0].lower() == 'vehicle']
#     elif self.information['vehicles'] == 'from_tracks':
#         vehicle_list = [trk for trk in self.tracks if trk.obj_type.lower() == 'vehicle']
#     else:
#         raise NotImplementedError

#     def dist(v): return v.get_location().distance(waypoint.transform.location)
#     vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self.vehicle.id]

#     if self.direction == RoadOption.CHANGELANELEFT:
#         vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(
#             waypoint, location, vehicle_list, max(
#                 self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=-1)
#     elif self.direction == RoadOption.CHANGELANERIGHT:
#         vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(
#             waypoint, location, vehicle_list, max(
#                 self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=1)
#     else:
#         vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(
#             waypoint, location, vehicle_list, max(
#                 self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=30)

#         # Check for overtaking
#         if vehicle_state and self.direction == RoadOption.LANEFOLLOW and \
#                 not waypoint.is_junction and self.speed > 5 \
#                 and self.behavior.overtake_counter == 0 and self.speed > get_speed(vehicle):
#             self._overtake(location, rotation, waypoint, vehicle_list)

#         # Check for tailgating

#         elif not vehicle_state and self.direction == RoadOption.LANEFOLLOW \
#                 and not waypoint.is_junction and self.speed > 5 \
#                 and self.behavior.tailgate_counter == 0:
#             self._tailgating(location, waypoint, vehicle_list)

#     return vehicle_state, vehicle, distance


#     def pedestrian_avoid_manager(self, location, waypoint):
#         """
#         This module is in charge of warning in case of a collision
#         with any pedestrian.

#             :param location: current location of the agent
#             :param waypoint: current waypoint of the agent
#             :return vehicle_state: True if there is a walker nearby, False if not
#             :return vehicle: nearby walker
#             :return distance: distance to nearby walker
#         """
#         if self.information['pedestrians'] == 'truth':
#             walker_list = self._world.get_actors().filter("*walker.pedestrian*")
#         elif self.information['pedestrians'] == 'from_objects':
#             walker_list = [obj[1] for obj in self.objects if obj[0].lower() == 'pedestrian']
#         elif self.information['pedestrians'] == 'from_tracks':
#             walker_list = [trk for trk in self.tracks if trk.obj_type.lower() == 'pedestrian']
#         else:
#             raise NotImplementedError
#         def dist(w): return w.get_location().distance(waypoint.transform.location)
#         walker_list = [w for w in walker_list if dist(w) < 10]

#         if self.direction == RoadOption.CHANGELANELEFT:
#             walker_state, walker, distance = self._bh_is_vehicle_hazard(waypoint, location, walker_list, max(
#                 self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=90, lane_offset=-1)
#         elif self.direction == RoadOption.CHANGELANERIGHT:
#             walker_state, walker, distance = self._bh_is_vehicle_hazard(waypoint, location, walker_list, max(
#                 self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=90, lane_offset=1)
#         else:
#             walker_state, walker, distance = self._bh_is_vehicle_hazard(waypoint, location, walker_list, max(
#                 self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=60)

#         return walker_state, walker, distance


#         # 2.1: Pedestrian avoidancd behaviors

#         walker_state, walker, w_distance = self.pedestrian_avoid_manager(
#             ego_vehicle_loc, ego_vehicle_wp)

#         if walker_state:
#             # Distance is computed from the center of the two cars,
#             # we use bounding boxes to calculate the actual distance
#             distance = w_distance - max(
#                 walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
#                     self.vehicle.bounding_box.extent.y, self.vehicle.bounding_box.extent.x)

#             # Emergency brake if the car is very close.
#             if distance < self.behavior.braking_distance:
#                 return self.emergency_stop()
