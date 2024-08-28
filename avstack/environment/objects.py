# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-28
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-27
# @Description:
"""

"""
from __future__ import annotations

import itertools
import json
from copy import deepcopy
from enum import IntEnum

import numpy as np

from avstack.geometry import (
    Acceleration,
    AngularVelocity,
    Attitude,
    Box2D,
    Box3D,
    GlobalOrigin3D,
    PassiveReferenceFrame,
    Position,
    ReferenceFrame,
    RotationDecoder,
    VectorDecoder,
    VectorHeadTail,
    Velocity,
    bbox,
)
from avstack.geometry import transformations as tforms
from avstack.maskfilters import box_in_fov, filter_points_in_box


NoneType = type(None)


class Occlusion(IntEnum):
    INVALID = -1
    NONE = 0
    PARTIAL = 1
    MOST = 2
    COMPLETE = 3
    UNKNOWN = 99


class ObjectStateEncoder(json.JSONEncoder):
    def default(self, o):
        o_dict = {
            "obj_type": o.obj_type,
            "ID": o.ID,
            "occlusion": o.occlusion,
            "t": o.t,
            "box": o.box.encode() if o.box is not None else None,
            "position": o.position.encode() if o.position is not None else None,
            "velocity": o.velocity.encode() if o.velocity is not None else None,
            "acceleration": (
                o.acceleration.encode() if o.acceleration is not None else None
            ),
            "attitude": o.attitude.encode() if o.attitude is not None else None,
            "angular_velocity": (
                o.angular_velocity.encode() if o.angular_velocity is not None else None
            ),
            "visible_fraction": o.visible_fraction,
        }
        return {"objectstate": o_dict}


class ObjectStateDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "objectstate" in json_object:
            json_object = json_object["objectstate"]
            obj = ObjectState(
                obj_type=json_object["obj_type"],
                ID=json_object["ID"],
            )
            if "visible_fraction" in json_object:
                vis_frac = (
                    None
                    if json_object["visible_fraction"] is None
                    else float(json_object["visible_fraction"])
                )
            else:
                vis_frac = None

            obj.set(
                t=json_object["t"],
                box=(
                    None
                    if json_object["box"] is None
                    else json.loads(json_object["box"], cls=bbox.BoxDecoder)
                ),
                position=(
                    None
                    if json_object["position"] is None
                    else json.loads(json_object["position"], cls=VectorDecoder)
                ),
                velocity=(
                    None
                    if json_object["velocity"] is None
                    else json.loads(json_object["velocity"], cls=VectorDecoder)
                ),
                acceleration=(
                    None
                    if json_object["acceleration"] is None
                    else json.loads(json_object["acceleration"], cls=VectorDecoder)
                ),
                attitude=(
                    None
                    if json_object["attitude"] is None
                    else json.loads(json_object["attitude"], cls=RotationDecoder)
                ),
                angular_velocity=(
                    None
                    if json_object["angular_velocity"] is None
                    else json.loads(
                        json_object["angular_velocity"], cls=RotationDecoder
                    )
                ),
                occlusion=Occlusion(json_object["occlusion"]),
                visible_fraction=vis_frac,
            )
        else:
            return json_object
        return obj


class ObjectState:
    _ids = itertools.count()

    def __init__(self, obj_type, ID=None, score: float = 1.0):
        self.ID = ID if ID is not None else next(self._ids)
        self.obj_type = obj_type
        self.score = score
        self.set(
            t=0,
            position=None,
            box=None,
            velocity=None,
            acceleration=None,
            attitude=None,
            angular_velocity=None,
            occlusion=Occlusion.UNKNOWN,
            visible_fraction=None,
        )

    def __str__(self):
        return f"VehicleState {self.obj_type} at position {self.position}"

    def __repr__(self):
        return self.__str__()

    @property
    def timestamp(self):
        return self.t

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

    @property
    def reference(self):
        if self.position is not None:
            return self.position.reference
        elif self.attitude is not None:
            return self.attitude.reference
        else:
            return None

    @property
    def velocity_head_tail(self):
        pos_g = self.position.in_global()
        vel_g = self.velocity.in_global()
        return VectorHeadTail(pos_g.x, pos_g.x + vel_g.x, GlobalOrigin3D)

    def __getitem__(self, key):
        if key == "size":
            return self.box3d.size
        else:
            raise NotImplementedError(key)

    def encode(self):
        return json.dumps(self, cls=ObjectStateEncoder)

    def deepcopy(self):
        return deepcopy(self)

    def allclose(self, other):
        return self.as_reference().allclose(other.as_reference())  # HACK

    def as_reference(self):
        pos = self.position.x if self.position else np.zeros((3,))
        vel = self.velocity.x if self.velocity else np.zeros((3,))
        acc = self.acceleration.x if self.acceleration else np.zeros((3,))
        att = self.attitude.q if self.attitude else np.quaternion(1)
        ang = self.angular_velocity.q if self.angular_velocity else np.quaternion(1)
        ref = self.reference
        return ReferenceFrame(
            x=pos, v=vel, acc=acc, q=att, ang=ang, reference=ref, timestamp=self.t
        )

    def set(
        self,
        t,
        position: Position,
        box: Box2D | Box3D,
        velocity: Velocity = None,
        acceleration: Acceleration = None,
        attitude: Attitude = None,
        angular_velocity: AngularVelocity = None,
        occlusion=Occlusion.UNKNOWN,
        visible_fraction: float = None,
    ):
        self.t = t
        self.occlusion = occlusion
        self.visible_fraction = visible_fraction

        # -- position
        assert isinstance(position, (Position, NoneType)), type(position)
        self.position = position

        # -- bbox
        assert isinstance(box, (Box2D, Box3D, NoneType)), type(box)
        self.box = box
        if self.box is not None:
            if self.box.obj_type is None:
                self.box.obj_type = self.obj_type
            if self.box.ID is None:
                self.box.ID = self.ID

        # -- velocity
        assert isinstance(velocity, (Velocity, NoneType)), type(velocity)
        self.velocity = velocity

        # -- accel
        assert isinstance(acceleration, (Acceleration, NoneType)), type(acceleration)
        self.acceleration = acceleration

        # -- attitude
        assert isinstance(attitude, (Attitude, NoneType)), type(attitude)
        self.attitude = attitude  # world 2 body

        # -- angular vel
        assert isinstance(angular_velocity, (AngularVelocity, NoneType)), type(
            angular_velocity
        )
        self.angular_velocity = angular_velocity

    def distance(self, other, check_reference: bool = True) -> float:
        return self.position.distance(other, check_reference=check_reference)

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
        )
        return VS

    def change_reference(self, reference: ReferenceFrame | ObjectState, inplace: bool):
        """Transform the reference frame of this object

        If other is a reference frame, assume it is static.
        If other is another object state, it may not be static.
        """

        # wrapping reference frame
        if isinstance(reference, (PassiveReferenceFrame, ReferenceFrame)):
            pass
        elif isinstance(reference, ObjectState):
            reference = reference.as_reference()
        else:
            raise NotImplementedError(type(reference))

        # perform transformation
        if self.reference.allclose(reference):
            if not inplace:
                return self
        else:
            # transforms
            if self.position is not None:
                position = self.position.change_reference(reference, inplace=inplace)
            else:
                position = None
            if self.velocity is not None:
                velocity = self.velocity.change_reference(reference, inplace=inplace)
            else:
                velocity = None
            if self.acceleration is not None:
                acceleration = self.acceleration.change_reference(
                    reference, inplace=inplace
                )
            else:
                acceleration = None
            if self.box is not None:
                box = self.box.change_reference(reference, inplace=inplace)
            else:
                box = None
            if self.attitude is not None:
                attitude = self.attitude.change_reference(reference, inplace=inplace)
            else:
                attitude = None
            if self.angular_velocity is not None:
                angular_velocity = self.angular_velocity.change_reference(
                    reference, inplace=inplace
                )
            else:
                angular_velocity = None

            if not inplace:
                obj_out = ObjectState(self.obj_type, self.ID)
                obj_out.set(
                    t=self.t,
                    position=position,
                    box=box,
                    velocity=velocity,
                    acceleration=acceleration,
                    attitude=attitude,
                    angular_velocity=angular_velocity,
                    occlusion=Occlusion.UNKNOWN,
                )
                return obj_out

    def set_occlusion_by_depth(self, depth_image, check_reference=True):
        """sets occlusion level using depth image

        Note that the depth image will capture the front-edge of the object
        Also take into account that an object doesn't take 100% of the bbox
        """
        if not box_in_fov(
            self.box, depth_image.calibration, check_reference=check_reference
        ):
            occ = Occlusion.INVALID
        else:
            # box 2d is in x=along width, y=along height
            box_2d = self.box.project_to_2d_bbox(
                depth_image.calibration, check_reference=check_reference
            ).squeeze(
                depth_image.calibration.height,
                depth_image.calibration.width,
                inplace=False,
            )
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

    def set_occlusion_by_lidar(self, pc, check_reference=True):
        """Sets occlusion level using lidar captures"""
        box_self = self.box
        if check_reference:
            box_self = box_self.change_reference(
                pc.calibration.reference, inplace=False
            )

        # then check the point cloud
        filter_pts_in_box = filter_points_in_box(pc.data, box_self.corners)
        pts_in_box = pc.data[filter_pts_in_box, :3]
        area_ratio = None
        if len(pts_in_box) <= 5:  # to account for occasional errors
            occ = Occlusion.COMPLETE
        else:
            # -- find a rotation matrix to go from v_sensor to v_object
            R = np.eye(3)
            vs = np.array([1, 0, 0])
            vo = box_self.t.x / np.linalg.norm(box_self.t.x)
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
            q_rot = tforms.transform_orientation(R.T, "dcm", "quat")
            box_self = box_self.rotate(q_rot, inplace=False)

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
                    ratios = [0.5, 0.25, 0.10]
                if area_ratio > ratios[0]:
                    occ = Occlusion.NONE
                elif area_ratio > ratios[1]:
                    occ = Occlusion.PARTIAL
                elif area_ratio > ratios[2]:
                    occ = Occlusion.MOST
                else:
                    occ = Occlusion.COMPLETE
        self.visible_fraction = area_ratio
        self.occlusion = occ

    def get_location(self, format_as="avstack"):
        if format_as == "avstack":
            return self.position.x
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
        origin=v_other.reference,
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
        origin=v_self.reference,
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
        origin=v_other.reference,
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
