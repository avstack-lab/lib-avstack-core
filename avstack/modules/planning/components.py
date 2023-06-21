# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-28
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-29
# @Description:
"""

"""

from math import cos, sin

import numpy as np


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
        try:
            rel_pos = obj.position
        except AttributeError:
            rel_pos = obj.box.position
        try:
            d_next = rel_pos.norm()
        except AttributeError:
            d_next = np.linalg.norm(rel_pos)
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
