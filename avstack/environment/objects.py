# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-28
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-27
# @Description:
"""

"""
import itertools
from copy import copy, deepcopy
from enum import IntEnum

import numpy as np
import quaternion

from avstack.geometry import (
    NominalOriginStandard,
    Rotation,
    Translation,
    VectorDirMag,
    bbox,
    get_origin_from_line,
)
from avstack.geometry import transformations as tforms
from avstack.maskfilters import box_in_fov, filter_points_in_box


class Occlusion(IntEnum):
    INVALID = -1
    NONE = 0
    PARTIAL = 1
    MOST = 2
    COMPLETE = 3
    UNKNOWN = 99


class ObjectState:
    _ids = itertools.count()

    def __init__(self, obj_type, ID=None):
        self.ID = ID if ID is not None else next(self._ids)
        self.obj_type = obj_type
        self.score = 1.0
        self.set(
            t=0,
            position=None,
            box=None,
            velocity=None,
            acceleration=None,
            attitude=None,
            angular_velocity=None,
            occlusion=Occlusion.UNKNOWN,
        )

    def __str__(self):
        return f"VehicleState {self.obj_type} at position {self.position}"

    def __repr__(self):
        return self.__str__()

    @property
    def occlusion(self):
        return self._occlusion

    @occlusion.setter
    def occlusion(self, occ):
        assert isinstance(occ, Occlusion), occ
        self._occlusion = occ

    @property
    def object_type(self):
        return self.obj_type

    @property
    def box3d(self):
        return self.box

    @property
    def speed(self):
        return self.velocity.norm()

    @property
    def yaw(self):
        return self.box.yaw

    def __getitem__(self, key):
        if key == "size":
            return self.box3d.size
        else:
            raise NotImplementedError(key)

    def deepcopy(self):
        return deepcopy(self)

    def set(
        self,
        t,
        position,
        box,
        velocity=None,
        acceleration=None,
        attitude=None,
        angular_velocity=None,
        occlusion=Occlusion.UNKNOWN,
        origin=NominalOriginStandard,
    ):
        self.t = t
        self.origin = origin
        self.occlusion = occlusion

        # -- position
        if isinstance(position, (list, np.ndarray)):
            position = Translation(position, origin=origin)
        self.position = position
        # -- bbox
        self.box = box
        # -- velocity
        if isinstance(velocity, (list, np.ndarray)):
            velocity = VectorDirMag(velocity, origin=origin)
        self.velocity = velocity
        # -- accel
        if isinstance(acceleration, (list, np.ndarray)):
            acceleration = VectorDirMag(acceleration, origin=origin)
        self.acceleration = acceleration
        # -- attitude
        if isinstance(attitude, (quaternion.quaternion)):
            attitude = Rotation(attitude, origin=origin)
        elif isinstance(attitude, (np.ndarray)):
            if attitude.shape == (3, 3):
                attitude = Rotation(attitude, origin=origin)
            else:
                raise NotImplementedError(attitude.shape)
        self.attitude = attitude  # world 2 body
        # -- angular vel
        if isinstance(angular_velocity, (list, np.ndarray)):
            angular_velocity = VectorDirMag(angular_velocity, origin=origin)
        self.angular_velocity = angular_velocity

    def predict(self, dt):
        assert dt >= 0
        pos = self.position + self.velocity * dt
        vel = self.velocity
        if (self.acceleration is not None) and self.acceleration.finite:
            pos += self.acceleration * (dt**2 / 2)
            vel += self.acceleration * dt
        box = deepcopy(self.box)
        acc = deepcopy(self.acceleration)
        att = deepcopy(self.attitude)
        ang = deepcopy(self.angular_velocity)
        VS = VehicleState(self.obj_type, ID=self.ID)
        VS.set(
            self.t + dt,
            pos,
            box,
            vel,
            acc,
            att,
            ang,
            occlusion=self.occlusion,
            origin=self.origin,
        )
        return VS

    def local_to_global(self, vehicle):
        """Transforms another vehicle's state from local to global

        assumes self is in global
        assumes self is the local origin of vehicle
        assumes vehicle is in the self frame
        """
        return local_to_global(self, vehicle)

    def global_to_local(self, vehicle):
        """Transforms another vehicle's state from global to local

        assumes self is in global
        assumes vehicle is in global
        """
        return global_to_local(self, vehicle)

    def set_occlusion_by_depth(self, depth_image, check_origin=True):
        """sets occlusion level using depth image

        Note that the depth image will capture the front-edge of the object
        Also take into account that an object doesn't take 100% of the bbox
        """
        if not box_in_fov(self.box, depth_image.calibration):
            occ = Occlusion.INVALID
        else:
            # box 2d is in x=along width, y=along height
            box_2d = self.box.project_to_2d_bbox(
                depth_image.calibration, check_origin=check_origin
            ).squeeze(depth_image.calibration.height, depth_image.calibration.width)
            depths = depth_image.depths[
                int(box_2d.ymin) : int(box_2d.ymax), int(box_2d.xmin) : int(box_2d.xmax)
            ]
            centered_depths = np.reshape(depths, -1) - (
                self.position.norm() - self.box.l / 2
            )

            if len(centered_depths) > 0:
                frac_viewable = sum(np.abs(centered_depths) < 5) / len(centered_depths)
                if frac_viewable > 0.5:
                    occ = Occlusion.NONE
                elif frac_viewable > 0.25:
                    occ = Occlusion.PARTIAL
                elif frac_viewable > 0.01:
                    occ = Occlusion.MOST
                else:
                    occ = Occlusion.COMPLETE
            else:
                occ = Occlusion.UNKNOWN
        self.occlusion = occ

    def set_occlusion_by_lidar(self, pc, check_origin=True):
        """Sets occlusion level using lidar captures"""
        box_self = self.box
        if check_origin:
            if not self.box.origin.allclose(pc.calibration.origin):
                box_self = deepcopy(self.box)
                box_self.change_origin(pc.calibration.origin)

        # then check the point cloud
        filter_pts_in_box = filter_points_in_box(pc.data, box_self.corners)
        pts_in_box = pc.data[filter_pts_in_box, :3]
        if len(pts_in_box) <= 1:  # to account for occasional errors
            occ = Occlusion.COMPLETE
        elif len(pts_in_box) <= 5:
            occ = Occlusion.MOST
        else:
            # -- find a rotation matrix to go from v_sensor to v_object
            R = np.eye(3)
            vs = np.array([1, 0, 0])
            vo = box_self.t.vector / np.linalg.norm(box_self.t.vector)
            if not np.allclose(vs, vo):
                w = np.cross(vs, vo)
                s = np.linalg.norm(w)
                c = np.dot(vs, vo)
                if np.isclose(c, -1):
                    R = tforms.rotz(np.pi)  # eek...
                else:
                    wx = np.array(
                        [[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]
                    )
                    R += wx + np.dot(wx, wx) * 1 / (1 + c)

            # -- rotate the object by this amount to put it forward
            box_self.rotate(R.T)

            # -- get corners and compute the viewable area straight-on
            corners_box = box_self.corners
            box_maxes = np.amax(corners_box, axis=0)
            box_mines = np.amin(corners_box, axis=0)
            view_height = box_maxes[2] - box_mines[2]
            view_width = box_maxes[1] - box_mines[1]
            area_view = view_width * view_height

            # -- get actual area covered by the points
            pts_straight = pts_in_box @ R
            pts_maxes = np.amax(pts_straight, axis=0)
            pts_mines = np.amin(pts_straight, axis=0)
            cover_height = pts_maxes[2] - pts_mines[2]
            cover_width = pts_maxes[1] - pts_mines[1]
            area_cover = cover_width * cover_height

            if area_view < 0.2:
                occ = Occlusion.UNKNOWN
            else:
                area_ratio = area_cover / area_view
                # -- results
                if self.obj_type.lower() in [
                    "motorcycle",
                    "pedestrian",
                    "bicycle",
                    "cyclist",
                ]:
                    ratios = [0.3, 0.10, 0.05]
                else:
                    ratios = [0.5, 0.25, 0.05]
                if area_ratio > ratios[0]:
                    occ = Occlusion.NONE
                elif area_ratio > ratios[1]:
                    occ = Occlusion.PARTIAL
                elif area_ratio > ratios[2]:
                    occ = Occlusion.MOST
                else:
                    occ = Occlusion.COMPLETE
        self.occlusion = occ

    def change_origin(self, origin_new):
        pos = self.position
        pos.change_origin(origin_new) if pos is not None else None
        box = self.box
        box.change_origin(origin_new) if box is not None else None
        vel = self.velocity
        vel.change_origin(origin_new) if vel is not None else None
        acc = self.acceleration
        acc.change_origin(origin_new) if acc is not None else None
        att = self.attitude
        att.change_origin(origin_new) if att is not None else None
        ang = self.angular_velocity
        ang.change_origin(origin_new) if ang is not None else None
        self.set(
            self.t,
            pos,
            box,
            vel,
            acc,
            att,
            ang,
            occlusion=self.occlusion,
            origin=origin_new,
        )

    def format_as(self, format_):
        try:
            box2d = self.box2d
        except AttributeError as e:
            box2d = bbox.Box2D([-1, -1, -1, -1], None)
        try:
            box3d = self.box3d
        except AttributeError as e:
            raise e
        accel = [np.nan] * 3 if self.acceleration is None else self.acceleration
        if format_.lower() == "kitti":
            orientation = 0
            str_to_write = (
                "kitti-v2 %f %i %s %f %f %f %f %f %f %f %f %f %f %f %f %f %s"
                % (
                    self.t,
                    self.ID,
                    self.obj_type,
                    orientation,
                    box2d.xmin,
                    box2d.ymin,
                    box2d.xmax,
                    box2d.ymax,
                    box3d.h,
                    box3d.w,
                    box3d.l,
                    box3d.t[0],
                    box3d.t[1],
                    box3d.t[2],
                    box3d.yaw,
                    self.score,
                    self.origin.format_as_string(),
                )
            )
        elif format_.lower() == "avstack":
            str_to_write = (
                "avstack object_3d %f %i %s %i %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %s %s"
                % (
                    self.t,
                    self.ID,
                    self.obj_type,
                    int(self.occlusion),
                    self.position[0],
                    self.position[1],
                    self.position[2],
                    self.velocity[0],
                    self.velocity[1],
                    self.velocity[2],
                    accel[0],
                    accel[1],
                    accel[2],
                    box3d.h,
                    box3d.w,
                    box3d.l,
                    box3d.q.w,
                    box3d.q.x,
                    box3d.q.y,
                    box3d.q.z,
                    box3d.where_is_t,
                    self.origin.format_as_string(),
                )
            )
        else:
            raise NotImplementedError(f"Cannot format track as {format_}")
        return str_to_write

    def read_from_line(self, line):
        idx = 2
        timestamp = float(line[idx])
        idx += 1
        ID = int(line[idx])
        idx += 1
        obj_type = line[idx]
        idx += 1
        occlusion = Occlusion(int(line[idx]))
        idx += 1
        position = np.array([float(d) for d in line[idx : idx + 3]])
        idx += 3
        velocity = np.array([float(d) for d in line[idx : idx + 3]])
        idx += 3
        accel = np.array([float(d) for d in line[idx : idx + 3]])
        idx += 3
        h, w, l, yaw = [float(d) for d in line[idx : idx + 4]]
        idx += 4
        where_is_t = line[idx]
        idx += 1
        origin = get_origin_from_line(line[idx])

        box = bbox.Box3D([h, w, l, np.zeros((3,)), yaw], origin, where_is_t=where_is_t)
        self.ID = ID
        attitude = tforms.rotz(yaw)
        angular_velocity = None
        self.set(
            timestamp,
            position,
            box,
            velocity,
            accel,
            attitude,
            angular_velocity,
            occlusion=occlusion,
            origin=origin,
        )

    def get_location(self, format_as="avstack"):
        if format_as == "avstack":
            return self.position.vector
        elif format_as == "carla":
            return np.array([self.position.x, -self.position.y, self.position.z])
        else:
            raise NotImplementedError(format_as)

    def get_rotation(self, format_as="avstack"):
        if format_as == "avstack":
            return self.attitude.q
        elif format_as == "carla":
            roll, pitch, yaw = self.attitude.euler
            return 180 / np.pi * np.array([-pitch, -yaw, roll])


VehicleState = ObjectState
PedestrianState = ObjectState
CyclistState = ObjectState


def global_to_local(v_self, v_other):
    """
    Transforms an object from the global frame to the frame of v_self

    Assumes they share the same global reference

    obj_1.position.x := x_OR1_to_obj1_in_OR1
    obj_1.attitude.q := q_OR1_to_obj1
    obj_2.position.x := x_OR1_to_obj2_in_OR1
    obj_2.position.x := q_OR1_to_obj2

    The calculation for position, attitude is as follows:
    # -- position
    x_obj1_to_obj2_in_OR1 = x_OR1_to_obj2_in_OR1 - x_OR1_to_obj1_in_OR1
    x_obj1_to_obj2_in_obj1 = q_mult_vec(q_OR1_to_obj1, x_obj1_to_obj2_in_OR1)

    # -- attitude
    q_obj1_to_obj2 = q_OR1_to_obj2 * q_OR1_to_obj1.conjugate()
    """
    if (v_self.attitude is None) or (v_other.attitude is None):
        raise RuntimeError("Attitudes must be set to run this")
    R_g_to_s = v_self.attitude
    R_g_to_v = v_other.attitude
    pos = R_g_to_s @ (v_other.position - v_self.position)
    if (v_other.velocity is None) or (v_self.velocity is None):
        vel = None
    else:
        vel = R_g_to_s @ (v_other.velocity - v_self.velocity)
    if (v_other.acceleration is None) or (v_self.acceleration is None):
        acc = None
    else:
        acc = R_g_to_s @ (v_other.acceleration - v_self.acceleration)
    att = R_g_to_v @ R_g_to_s.T
    if (v_other.angular_velocity is None) or (v_self.angular_velocity is None):
        ang = None
    else:
        ang = R_g_to_s @ (v_other.angular_velocity - v_self.angular_velocity)
    if (v_self.box.where_is_t == "bottom") and (v_other.box.where_is_t == "bottom"):
        where_is_t = "bottom"
    elif (v_self.box.where_is_t == "center") and (v_other.box.where_is_t == "center"):
        where_is_t = "center"
    else:
        raise RuntimeError(
            f"box1 and box2 must be both either bottom or center...instead ({v_self.box.where_is_t}, {v_other.box.where_is_t})"
        )
    box = bbox.Box3D(
        [v_other.box.h, v_other.box.w, v_other.box.l, pos, att.q], where_is_t=where_is_t
    )
    VS = VehicleState(v_other.obj_type, v_other.ID)
    VS.set(
        v_other.t,
        pos,
        box,
        vel,
        acc,
        att,
        ang,
        occlusion=v_self.occlusion,
        origin=v_other.origin,
    )
    return VS


def local_to_global(v_self, v_other):
    """
    Transforms an object from the reference frame of v_self to global

    Assumes they share the same global reference

    obj_1.position.x := x_OR1_to_obj1_in_OR1
    obj_1.attitude.q := q_OR1_to_obj1
    obj_2.position.x := x_obj1_to_obj2_in_obj1
    obj_2.attitude.x := q_obj1_to_obj2

    The calculation for position, attitude is as follows:
    # -- position
    x_obj1_to_obj2_in_OR1 = q_mult_vec(q_OR1_to_obj1.conjugate(), x_obj1_to_obj2_in_obj1)
    x_OR1_to_obj2_in_OR1 = x_obj1_to_obj2_in_OR1 + x_OR1_to_obj1_in_OR1

    # -- attitude
    q_OR1_to_obj2 = q_obj1_to_obj2.conjugate() * q_OR1_to_obj1
    """
    if (v_self.attitude is None) or (v_other.attitude is None):
        raise RuntimeError("Attitudes must be set to run this")

    R_g_to_s = v_self.attitude
    R_s_to_g = R_g_to_s.T
    R_s_to_v = v_other.attitude

    pos = R_s_to_g @ v_other.position + v_self.position
    vel = R_s_to_g @ v_other.velocity + v_self.velocity
    acc = R_s_to_g @ v_other.acceleration + v_self.acceleration
    att = v_other.attitude @ R_g_to_s
    if (v_other.angular_velocity is None) or (v_self.angular_velocity is None):
        ang = None
    else:
        ang = R_s_to_g @ v_other.angular_velocity + v_self.angular_velocity

    VS = VehicleState(v_other.obj_type, v_other.ID)
    VS.set(
        v_other.t,
        pos,
        v_other.box,
        vel,
        acc,
        att,
        ang,
        occlusion=v_self.occlusion,
        origin=v_self.origin,
    )
    return VS


def global_to_local_from_origin(o_self, v_other):
    if (o_self.q is None) or (v_other.attitude is None):
        raise RuntimeError("Attitudes must be set to run this")
    R_g_to_s = o_self.rotation
    R_g_to_v = v_other.attitude
    pos = R_g_to_s @ (v_other.position - o_self.x)
    vel = R_g_to_s @ (v_other.velocity - np.zeros((3,)))
    acc = R_g_to_s @ (v_other.acceleration - np.zeros((3,)))
    att = R_g_to_v @ R_g_to_s.T
    if v_other.angular_velocity is None:
        ang = None
    else:
        ang = R_g_to_s @ (v_other.angular_velocity - np.zeros((3,)))
    box = bbox.Box3D(
        [v_other.box.h, v_other.box.w, v_other.box.l, pos, att.q],
        where_is_t=v_other.box.where_is_t,
    )
    VS = VehicleState(v_other.obj_type, v_other.ID)
    VS.set(
        v_other.t,
        pos,
        box,
        vel,
        acc,
        att,
        ang,
        occlusion=v_other.occlusion,
        origin=v_other.origin,
    )
    return VS


def IOU_2d(corners1, corners2):
    """Compute the IoU 2D

    corners1, corners2 - an array of [xmin, ymin, xmax, ymax]
    """
    inter = bbox.box_intersection(corners1, corners2)
    union = bbox.box_union(corners1, corners2)
    return inter / union


def IOU_3d(corners1, corners2):
    """Compute the IoU 3D

    corners1: numpy array (8,3), assume up direction is negative Y
    """
    inter = bbox.box_intersection(corners1, corners2)
    union = bbox.box_union(corners1, corners2)
    return inter / union
