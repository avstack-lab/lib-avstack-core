# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-28
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-28
# @Description:
"""

"""

import numpy as np
from copy import copy, deepcopy
from avstack import transformations as tforms
from avstack.geometry import Transform, Rotation

from . import components
from .base import WaypointPlan, Waypoint, _PlanningAlgorithm


class AdaptiveCruiseControl(_PlanningAlgorithm):
    """Follow an object in a lane within a suitable distance"""
    def __init__(self, dt_target=3, dt_max=20, object_ID=1, verbose=False):
        self.dt_target = dt_target
        self.dt_max = dt_max
        self.lane_keeping_planner = LaneKeepingPlanner()
        self.object_ID = object_ID
        self.verbose = verbose
        self.following = False

    def __call__(self, plan, ego_state, environment, objects_3d, objects_2d, lane_lines, **kwargs):
        ############################################
        # TODO: IMPROVE THIS LOGIC
        # --- USE PREDICTIONS
        # --- UPDATE ALL FUTURE TARGET SPEEDS
        # --- SQUEEZE TARGET WAYPOINTS TO WITHIN LANES
        # --- IMPROVE TARGET ROTATION USING SLERP/SQUAD
        # --- REMOVE IN LINE ASSUMPTIONS
        # --- MAKE MORE EFFICIENT
        ############################################
        speed_limit = environment.speed_limit
        plan.update(ego_state)

        # -- check if there is an object to follow
        obj_follow = components.get_object_to_follow(ego_state, objects_3d, lane_lines)
        if obj_follow is not None:
            if self.verbose:
                print('::Planning - found object to follow')
            pos_rel = obj_follow.position
            vel_rel = obj_follow.velocity
            range_rel = pos_rel.norm()
            speed_rel = vel_rel.norm()
            speed_ego = ego_state.velocity.norm()
            t_behind_track = range_rel / speed_ego if speed_ego > 0 else np.inf
            t_to_catch = np.inf if speed_rel <= 0 else range_rel/speed_rel

        # -- default to lane keeping, otherwise follow obj
        if (obj_follow is None) or ((t_behind_track >= self.dt_max) and speed_ego > 0):
            if self.following:
                plan.clear()
                self.following = False
            if self.verbose:
                print('::Planning - defaulting to lane keeping planner')
            plan = self.lane_keeping_planner(
                plan, ego_state, lane_lines, environment)
        else:
            self.following = True
            forward = ego_state.attitude.forward_vector
            vel_global = ego_state.attitude.T @ obj_follow.velocity + ego_state.velocity
            loc_rel = obj_follow.position - 4*forward
            loc = ego_state.attitude.T @ loc_rel + ego_state.position
            dist = (loc - ego_state.position).norm()
            low = 7
            upp = 30
            if low < dist < upp:
                target_speed = min(speed_limit, vel_global.norm())
            elif dist > upp:
                target_speed = speed_limit
            else:
                target_speed = 0
            wpt = Waypoint(Transform(ego_state.attitude, loc), target_speed)
            plan.push(dist, wpt)
        return plan


class LaneKeepingPlanner(_PlanningAlgorithm):
    """Keep the lane and follow traffic signals"""
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, plan, ego_state, lane_lines, environment, **kwargs):
        """Execute planning logic to get next set of waypoints""" 
        plan.update(ego_state)
        if plan.needs_waypoint():
            distance, waypoint = self._get_waypoint(ego_state, lane_lines, environment.speed_limit)
            plan.push(distance, waypoint)
        return plan

    def _get_waypoint(self, ego_state, lane_lines, speed_target, d_forward=2):
        forward_vec = ego_state.attitude.forward_vector
        left_vec = ego_state.attitude.left_vector

        # Extract from main camera
        if (len(lane_lines) > 0) and (isinstance(lane_lines[0], list)):
            lane_lines = lane_lines[0]
        # else:
        #     raise NotImplementedError(f'{len(lane_lines)}, {lane_lines}')

        # Process lane lines
        if len(lane_lines) < 2:
            target_speed = speed_target - 5
            target_loc = ego_state.position + d_forward*forward_vec
            target_rot = ego_state.attitude
            print('Lanes not found...going forward')
        elif len(lane_lines) == 2:
            _, lateral_offset, yaw_offset = lane_lines[0].compute_center_lane_and_offset(lane_lines[1])
            target_loc = ego_state.position + \
                         lateral_offset*left_vec + \
                         d_forward*forward_vec
            R_b2way = Rotation(tforms.get_rot_yaw_matrix(-yaw_offset, '+z'), target_loc.origin)
            R_world2b = ego_state.attitude
            target_rot = R_b2way @ R_world2b
            target_speed = speed_target
        else:
            raise NotImplementedError
        distance = ego_state.position.distance(target_loc)
        target_point = Transform(target_rot, target_loc)
        return distance, Waypoint(target_point, target_speed)


class RandomPlanner(_PlanningAlgorithm):
    """Finds random waypoints to go to"""
    def __init__(self, max_lateral_dist=2, min_forward_dist=5, max_forward_dist=10, max_speed=20, verbose=False):
        self.min_forward_dist = min_forward_dist
        self.max_forward_dist = max_forward_dist
        self.max_lateral_dist = max_lateral_dist
        self.max_speed = max_speed
        self.verbose = verbose

    def __call__(self, plan, ego_state, **kwargs):
        plan.update(ego_state)
        if plan.needs_waypoint():
            plan.push(*self._get_waypoint(ego_state))
        return plan

    def _get_waypoint(self, ego_state):
        forward_vec = tforms.get_rot_yaw_matrix(ego_state.attitude.yaw, '+z')[:,0]
        d1 = self.min_forward_dist
        d2 = self.max_forward_dist
        da = ((d2-d1) * np.random.rand() + d1) * forward_vec
        db = self.max_lateral_dist*(2*(np.random.rand(3)-1/2))
        d_pos = da + db
        target_loc = ego_state.position + d_pos
        target_rot = deepcopy(ego_state.attitude)
        target_point = Transform(target_rot, target_loc)
        target_speed = ((1-0.2)*np.random.rand() + 0.2) * self.max_speed
        dist_wpt = ego_state.position.distance(target_point)
        return dist_wpt, Waypoint(target_point, target_speed)


class StationaryPlanner(_PlanningAlgorithm):
    """Stays in the same spot"""
    def __init__(self, verbose=False, *args, **kwargs):
        self.init_state = None
        self.verbose = verbose

    def __call__(self, plan, ego_state, **kwargs):
        if self.init_state is None:
            plan.push(*self._get_waypoint(ego_state))
        return plan

    def _get_waypoint(self, ego_state):
        target_point = Transform(ego_state.attitude, ego_state.position)
        target_speed = 0
        dist_wpt = ego_state.position.distance(target_point)
        return dist_wpt, Waypoint(target_point, target_speed)


class GoStraightPlanner(_PlanningAlgorithm):
    """Moves forward"""
    def __init__(self, *args, d_forward=3, target_speed=20, verbose=False, **kwargs):
        self.d_forward = d_forward
        self.target_speed = target_speed
        self.verbose = verbose

    def __call__(self, plan, ego_state, **kwargs):
        plan.update(ego_state)
        if plan.needs_waypoint():
            plan.push(self._get_waypoint(ego_state))
        return plan

    def _get_waypoint(self, ego_state):
        forward_vec = tforms.get_rot_yaw_matrix(ego_state.attitude.yaw, '+z')[:,0]
        target_loc = ego_state.position + self.d_forward*forward_vec
        target_point = Transform(ego_state.attitude, target_loc)
        dist_wpt = ego_state.position.distance(target_point)
        return dist_wpt, Waypoint(target_point, self.target_speed)


# ===========================================================
# MAP-BASED PLANNER -- BASED ON THE CARLA EXAMPLES FILES
# ===========================================================

class MapBasedPlanningAndControl(_PlanningAlgorithm):
    """Uses map information to get to destination

    initially based on the CARLA examples by removing
    all ground truth information from the planner and using
    inputs instead

    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment,
    such as overtaking or tailgating avoidance. Adding to these are possible
    behaviors, the agent can also keep safety distance from a car in front of it
    by tracking the instantaneous time to collision and keeping it in a certain range.
    Finally, different sets of behaviors are encoded in the agent, from cautious
    to a more aggressive ones.
    """
    def __init__(self, ego_state, map_data, ignore_traffic_light=True, verbose=False):
        information = {'vehicles':'objects_3d',
                        'pedestrians':'objects_3d',
                        'traffic_lights':None}
        # deferred import to here for path reasons
        self.destination = None
        raise NotImplementedError('Have not implemented and validated behavior agent')
        self.agent = carla_components.behavior_agent.BehaviorAgent(ego_state=ego_state,
            map_data=map_data, information=information,
            ignore_traffic_light=ignore_traffic_light, behavior='normal')

    def set_destination(self, destination, coordinates='avstack', clean=True):
        """
        input in standard coordinates becomes carla coordinates
        """
        self.destination = destination
        if coordinates == 'avstack':
            dest = [destination[0], -destination[1], destination[2]]
        elif coordinates == 'carla':
            dest = destination
        else:
            raise NotImplementedError(coordinates)
        e_loc = self.agent.ego_state.get_location(format_as='carla')
        dest_true_avstack = self.agent.set_destination(e_loc, dest, clean=clean)        
        return dest_true_avstack

    def __call__(self, ego_state, environment, objects_3d, objects_2d):
        if self.destination is None:
            raise RuntimeError('Must set destination first')
        self.agent.update_information(ego_state, environment.speed_limit,
            objects_3d=objects_3d, objects_2d=objects_2d)
        ctrl = self.agent.run_step()
        return ctrl