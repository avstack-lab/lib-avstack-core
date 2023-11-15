# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-27
# @Last Modified by:   spencer@primus
# @Last Modified date: 2022-08-25
# @Description:
"""

"""
import json
from typing import List

import numpy as np
from scipy.interpolate import interp1d

from avstack.datastructs import DataContainerDecoder
from avstack.geometry import (
    Box2D,
    Box3D,
    ReferenceDecoder,
    ReferenceFrame,
    SegMask2D,
    Vector,
    bbox,
)
from avstack.geometry.transformations import (
    cartesian_to_spherical,
    spherical_to_cartesian,
)


class DetectionEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(
            o, (CentroidDetection, RazDetection, RazelDetection, RazelRrtDetection)
        ):
            data = o.data.tolist()
        elif isinstance(o, BoxDetection):
            data = o.box.encode()
        else:
            raise NotImplementedError(f"{type(o)}, {o}")
        d_dict = {
            "source_identifier": str(o.source_identifier),
            "obj_type": str(o.obj_type) if o.obj_type is not None else None,
            "score": float(o.score) if o.score is not None else None,
            "data": data,
            "reference": o.reference.encode(),
        }
        return {type(o).__name__.lower(): d_dict}


class DetectionDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        try:
            reference = json.loads(
                list(json_object.values())[0]["reference"], cls=ReferenceDecoder
            )
        except Exception:
            pass
        if "centroiddetection" in json_object:
            json_object = json_object["centroiddetection"]
            out = CentroidDetection(
                source_identifier=json_object["source_identifier"],
                centroid=np.array(json_object["data"]),
                obj_type=json_object["obj_type"],
                score=json_object["score"],
                reference=reference,
            )
        elif "razdetection" in json_object:
            json_object = json_object["razdetection"]
            out = RazDetection(
                source_identifier=json_object["source_identifier"],
                raz=np.array(json_object["data"]),
                obj_type=json_object["obj_type"],
                score=json_object["score"],
            )
        elif "razeldetection" in json_object:
            json_object = json_object["razeldetection"]
            out = RazelDetection(
                source_identifier=json_object["source_identifier"],
                razel=np.array(json_object["data"]),
                obj_type=json_object["obj_type"],
                score=json_object["score"],
                reference=reference,
            )
        elif "razelrrtdetection" in json_object:
            json_object = json_object["razelrrtdetection"]
            out = RazelRrtDetection(
                source_identifier=json_object["source_identifier"],
                razelrrt=np.array(json_object["data"]),
                obj_type=json_object["obj_type"],
                score=json_object["score"],
                reference=reference,
            )
        elif "boxdetection" in json_object:
            json_object = json_object["boxdetection"]
            out = BoxDetection(
                source_identifier=json_object["source_identifier"],
                box=json.loads(json_object["data"], cls=bbox.BoxDecoder),
                obj_type=json_object["obj_type"],
                score=json_object["score"],
                reference=reference,
            )
        else:
            return json_object
        return out


class DetectionContainerDecoder(DataContainerDecoder):
    data_decoder = DetectionDecoder


class Detection_:
    def __init__(self, source_identifier, reference, obj_type, score):
        self.reference = reference
        self.source_identifier = source_identifier
        self.obj_type = obj_type
        self.score = score
        self.ID = None

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, reference):
        assert isinstance(reference, ReferenceFrame)
        self._reference = reference

    @property
    def obj_type(self):
        return self._obj_type

    @obj_type.setter
    def obj_type(self, obj_type):
        if obj_type is not None:
            # if obj_type.lower() not in detection_map:
            #     raise RuntimeError(f'Object of type {obj_type} not accounted for in {detection_map}')
            # else:
            self._obj_type = obj_type  # detection_map[obj_type]
        else:
            self._obj_type = None

    @property
    def origin(self):
        return self.data.reference

    def IoU(self, *args, **kwargs):
        return self.data.IoU(*args, **kwargs)

    def encode(self):
        return json.dumps(self, cls=DetectionEncoder)

    def change_reference(self, reference, inplace: bool):
        """Change reference frame of a detection"""
        if inplace:
            self._change_reference(reference, inplace=inplace)
        else:
            data = self._change_reference(reference, inplace=inplace)
            return self.factory()(
                self.source_identifier,
                data,
                reference,
                obj_type=self.obj_type,
                score=self.score,
            )

    def _change_reference(self, reference, inplace: bool):
        raise NotImplementedError

    def __str__(self):
        return f"{self.obj_type} detection from sensor {self.source_identifier}\n{self.data}"

    def __repr__(self):
        return self.__str__()


class CentroidDetection(Detection_):
    def __init__(
        self, source_identifier, centroid, reference, obj_type=None, score=None
    ):
        super().__init__(source_identifier, reference, obj_type, score)
        self.centroid = centroid

    @property
    def data(self):
        return self.centroid

    @property
    def centroid(self):
        return self._centroid

    @property
    def z(self):
        return self.centroid

    @property
    def xyz(self):
        return self.centroid[:3]

    @property
    def xy(self):
        return self.centroid[:2]

    @centroid.setter
    def centroid(self, centroid):
        if not isinstance(centroid, (np.ndarray)):
            raise TypeError(
                f"Input centroid of type {type(centroid)} is not of an acceptable type"
            )
        self._centroid = centroid

    @staticmethod
    def factory():
        return CentroidDetection

    def _change_reference(self, reference, inplace: bool):
        if len(self.centroid) == 3:
            vec = Vector(self.centroid, self.reference)
        elif len(self.centroid) == 2:
            vec = Vector([self.centroid[0], self.centroid[1], 0], self.reference)
        else:
            raise NotImplementedError(len(self.centroid))
        vec.change_reference(reference, inplace=True)
        if inplace:
            self.centroid = vec.x[: len(self.centroid)]
        else:
            return vec.x[: len(self.centroid)]


class RazDetection(Detection_):
    def __init__(self, source_identifier, raz, reference, obj_type=None, score=None):
        super().__init__(source_identifier, reference, obj_type, score)
        self.raz = raz

    @property
    def data(self):
        return self.raz

    @property
    def raz(self):
        return self._raz

    @property
    def z(self):
        return self.raz

    @raz.setter
    def raz(self, raz):
        if not isinstance(raz, np.ndarray):
            raise TypeError(
                f"Input raz of type {type(raz)} is not of an acceptable type"
            )
        self._raz = raz

    @property
    def xy(self):
        x, y = self.raz[0] * np.cos(self.raz[1]), self.raz[0] * np.sin(self.raz[1])
        return np.array([x, y])

    @xy.setter
    def xy(self, xy):
        self.raz = cartesian_to_spherical(np.array([xy[0], xy[1], 0]))[:2]

    @staticmethod
    def factory():
        return RazDetection

    def _change_reference(self, reference, inplace: bool):
        """Very hacky right now..."""
        vec = Vector(np.array([self.xy[0], self.xy[1], 0]), self.reference)
        vec = vec.change_reference(reference, inplace=inplace)
        if inplace:
            self.xy = vec.x[:2]
        else:
            return cartesian_to_spherical(np.array([vec.x[0], vec.x[1], 0]))[:2]


class RazelDetection(Detection_):
    def __init__(self, source_identifier, razel, reference, obj_type=None, score=None):
        super().__init__(source_identifier, reference, obj_type, score)
        self.razel = razel

    @property
    def data(self):
        return self.razel

    @property
    def razel(self):
        return self._razel

    @property
    def z(self):
        return self.razel

    @razel.setter
    def razel(self, razel):
        if not isinstance(razel, np.ndarray):
            raise TypeError(
                f"Input razel of type {type(razel)} is not of an acceptable type"
            )
        self._razel = razel

    @property
    def x(self):
        return self.xyz

    @property
    def xyz(self):
        x, y, z = spherical_to_cartesian(self.razel)
        return np.array([x, y, z])

    @xyz.setter
    def xyz(self, xyz):
        self.razel = cartesian_to_spherical(xyz)

    @staticmethod
    def factory():
        return RazelDetection

    def _change_reference(self, reference, inplace: bool):
        vec = Vector(self.xyz, self.reference)
        vec = vec.change_reference(reference, inplace=inplace)
        if inplace:
            self.xyz = vec.x
        else:
            return cartesian_to_spherical(vec.x)


class RazelRrtDetection(Detection_):
    """NOTE: range rate is defined as positive away from sensor"""

    def __init__(
        self, source_identifier, razelrrt, reference, obj_type=None, score=None
    ):
        super().__init__(source_identifier, reference, obj_type, score)
        self.razelrrt = razelrrt

    @property
    def data(self):
        return self.razelrrt

    @property
    def razelrrt(self):
        return self._razelrrt

    @razelrrt.setter
    def razelrrt(self, razelrrt):
        if not isinstance(razelrrt, np.ndarray):
            raise TypeError(
                f"Input razelrrt of type {type(razelrrt)} is not of an acceptable type"
            )
        self._razelrrt = razelrrt

    @property
    def z(self):
        return self.razelrrt

    @property
    def xyzrrt(self):
        x, y, z = spherical_to_cartesian(self.razelrrt[:3])
        return np.array([x, y, z, self.razelrrt[3]])

    @xyzrrt.setter
    def xyzrrt(self, xyzrrt):
        rng, az, el = cartesian_to_spherical(xyzrrt[:3])
        self.razelrrt = np.array([rng, az, el, xyzrrt[3]])

    @property
    def xyz(self):
        x, y, z = spherical_to_cartesian(self.razelrrt[:3])
        return np.array([x, y, z])

    @staticmethod
    def factory():
        return RazelRrtDetection

    def as_razel(self):
        return RazelDetection(
            source_identifier=self.source_identifier,
            razel=self.razelrrt[:3],
            reference=self.reference,
            obj_type=self.obj_type,
            score=self.score,
        )

    def _change_reference(self, reference, inplace: bool):
        uv_before = self.xyz / np.linalg.norm(self.xyz)
        x, y, z = (
            Vector(self.xyz, self.reference)
            .change_reference(reference, inplace=False)
            .x
        )
        uv_after = np.array([x, y, z]) / np.linalg.norm(np.array([x, y, z]))
        # NOTE: can't really change a range rate measurement reference...
        rrt = self.razelrrt[3] * np.dot(uv_before, uv_after)
        if inplace:
            self.xyzrrt = np.array([x, y, z, rrt])
            self.reference = reference
        else:
            rng, az, el = cartesian_to_spherical([x, y, z])
            return np.array([rng, az, el, rrt])


class BoxDetection(Detection_):
    def __init__(self, source_identifier, box, reference, obj_type=None, score=None):
        super().__init__(source_identifier, reference, obj_type, score)
        self.box = box

    @property
    def data(self):
        return self.box

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, box):
        if not (isinstance(box, Box2D) or isinstance(box, Box3D)):
            raise TypeError(
                f"Input box of type {type(box)} is not of an acceptable type"
            )
        self._box = box

    @property
    def box3d(self):
        return self.box

    @property
    def box2d(self):
        return self.box

    @property
    def z(self):
        return self.box

    @property
    def position(self):
        return self.box.position

    @staticmethod
    def factory():
        return BoxDetection

    def _change_reference(self, reference, inplace: bool):
        return self.data.change_reference(reference, inplace=inplace)


class JointBoxDetection(Detection_):
    def __init__(
        self,
        source_identifier,
        box2d,
        box3d,
        reference,
        obj_type=None,
        score=None,
    ):
        super().__init__(source_identifier, reference, obj_type, score)
        self.box2d = box2d
        self.box3d = box3d

    def __iter__(self):
        return iter((self.box2d, self.box3d))

    @property
    def data(self):
        return self.box2d, self.box3d

    @property
    def box_1(self):
        return self.box_2d

    @property
    def box_2(self):
        return self.box_3d

    @property
    def box_2d(self):
        return self._box2d

    @box_2d.setter
    def box_2d(self, box2d):
        if not isinstance(box2d, Box2D):
            raise TypeError(
                f"Input box of type {type(box2d)} is not of an acceptable type"
            )
        self._box2d = box2d

    @property
    def box_3d(self):
        return self._box3d

    @box_3d.setter
    def box_3d(self, box3d):
        if not isinstance(box3d, Box3D):
            raise TypeError(
                f"Input box of type {type(box3d)} is not of an acceptable type"
            )
        self._box_3d = box3d

    @staticmethod
    def factory():
        return JointBoxDetection

    def _change_reference(self, reference, inplace: bool):
        return super()._change_reference(reference, inplace)


class JointBoxDetectionAndOther(JointBoxDetection):
    pass


class MaskDetection(Detection_):
    def __init__(
        self, source_identifier, box, mask, reference, obj_type=None, score=None
    ):
        super().__init__(source_identifier, reference, obj_type, score)
        self.box = box
        self.mask = mask

    @property
    def data(self):
        return (self.box, self.mask)

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, box):
        if not (isinstance(box, Box2D)):
            raise TypeError(
                f"Input box of type {type(box)} is not of an acceptable type"
            )
        self._box = box

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if not (isinstance(mask, SegMask2D)):
            raise TypeError(
                f"Input mask of type {type(mask)} is not of an acceptable type"
            )
        self._mask = mask

    @property
    def box2d(self):
        return self.box

    @staticmethod
    def factory():
        return MaskDetection

    def _change_reference(self, reference, inplace: bool):
        return super()._change_reference(reference, inplace)


class OtherDetection(Detection_):
    def __init__(self, source_identifier, data, reference, obj_type=None, score=None):
        super().__init__(source_identifier, reference, obj_type, score)
        self.data = data


class LaneLineInSpace:
    """Lane line in terms of cartesian space"""

    def __init__(self, points: List[Vector]):
        self._points = points

    def __len__(self):
        return len(self._points)

    def __getitem__(self, index):
        return self._points[index]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Lane line with x:{self.x}, y:{self.y}"

    @property
    def x(self):
        return np.array([p.x[0] for p in self._points])

    @property
    def y(self):
        return np.array([p.x[1] for p in self._points])

    @staticmethod
    def factory():
        return LaneLineInSpace

    def distance_closest(self, obj):
        ### TODO: IMPROVE THIS WITH INTERPOLATION/CURVE FITTING/DOT PRODUCT
        return min([obj.distance(p) for p in self._points])

    def object_between_lanes(self, other, obj):
        center_lane, lane_width = self.compute_center_lane(other)
        if (center_lane) and (center_lane.distance_closest(obj) < lane_width / 2):
            return True
        else:
            return False

    def object_between_lanes_projected(self, other, obj):
        """Assumes forward, left, up coordinates"""
        # left = obj.y
        # if np.any([left <= p.y for p in self._points]) and \
        #     np.any([left >= p.y for p in other._points]):
        #     return True
        # # elif np.any([left >= p.y for p in self._points]) and \
        # #     np.any([left <= p.y for p in other._points]):
        # #     return True
        # else:
        #     return False
        for i in range(0, len(self._points)):
            # print(obj.y <= self._points[i].y, obj.y >= other._points[i].y, obj.x >= self._points[i].x - 2)
            if (
                obj.x[1] <= self._points[i].x[1]
                and obj.x[1] >= other._points[i].x[1]
                and obj.x[0] >= self._points[i].x[0] - 2
                and obj.x[0] <= self._points[i].x[0] + 2
            ):
                return True
            elif (
                obj.x[1] >= self._points[i].x[1]
                and obj.x[1] <= other._points[i].x[1]
                and obj.x[0] >= self._points[i].x[0] - 2
                and obj.x[0] <= self._points[i].x[0] + 2
            ):
                return True
        return False

    def compute_center_lane(self, other):
        # determine which is right and left lanes
        lane_left = self if np.mean(self.y) >= np.mean(other.y) else other
        lane_right = self if lane_left == other else other
        # compute center lane
        min_fwd, max_fwd = max(lane_left[0].x[0], lane_right[0].x[0]), min(
            lane_left[-1].x[0], lane_right[-1].x[0]
        )
        if not (max_fwd > min_fwd >= 0):
            # try to cut lane in half...
            min_fwd = max(lane_left[0].x[0], lane_right[0].x[0])
            max_fwd = min(
                lane_left[len(lane_left) // 2].x, lane_right[len(lane_right) // 2].x
            )
            if not (max_fwd > min_fwd >= 0):
                print("Invalid lane detection...")
                return None, None
        delta = 0.5
        x_pts = np.linspace(min_fwd, max_fwd, int((max_fwd - min_fwd) / delta))
        left_y = interp1d(lane_left.x, lane_left.y)(x_pts)
        right_y = interp1d(lane_right.x, lane_right.y)(x_pts)
        lane_width = np.mean(left_y - right_y)
        center_y = (left_y + right_y) / 2
        center_points = [
            Vector([x, y, 0], self._points[0].reference)
            for x, y in zip(x_pts, center_y)
        ]
        return LaneLineInSpace(center_points), lane_width

    def compute_center_lane_and_offset(self, other):
        center_lane, lane_width = self.compute_center_lane(other)
        if center_lane is not None:
            n_pts_offset = 5
            n_pts_yaw = 10
            yaw_offset = 0  # TODO
            lateral_offset = np.mean(center_lane.y[:n_pts_offset])
            return center_lane, lateral_offset, yaw_offset
        else:
            return None, 0, 0


class LaneLineInPixels:
    """Lane line in terms of pixels"""

    def __init__(self, lane_points, image_size):
        """Coordinates are in (row, col) pairs"""
        self._points = np.array(sorted(lane_points))
        assert (
            len(np.unique(self._points[:, 0])) == self._points.shape[0]
        ), "Must have unique rows"
        self._points_map = {r: c for r, c in self._points}
        self.image_size = image_size

    def __len__(self):
        return len(self._points)

    def __getitem__(self, index):
        return self.coordinate_by_index(index)

    @property
    def x(self):
        return np.array([p[0] for p in self._points])

    @property
    def y(self):
        return np.array([p[1] for p in self._points])

    @staticmethod
    def factory():
        return LaneLineInPixels

    def coordinate_by_index(self, index):
        return self._points[index, :]

    def col_by_row(self, row_index):
        return self._points[row_index]

    def compute_center_lane(self, other, lane_width):
        # determine which is right and left lanes
        lane_left = self if np.mean(self.y) <= np.mean(other.y) else other
        lane_right = self if lane_left == other else other

        # compute center lane
        r_l_dict = dict((k[0], i) for i, k in enumerate(lane_left))
        r_r_dict = dict((k[0], i) for i, k in enumerate(lane_right))
        inter = set(r_l_dict).intersection(set(r_r_dict))
        idx_left = [r_l_dict[x] for x in inter]
        idx_right = [r_r_dict[x] for x in inter]
        center_pairs = [
            (idx, (lane_left[r_l_dict[idx]][1] + lane_right[r_r_dict[idx]][1]) / 2)
            for idx in inter
        ]
        # meters/pixel at each row
        center_scaling = np.array(
            [
                lane_width
                / (lane_right[r_l_dict[idx]][1] - lane_left[r_r_dict[idx]][1])
                for idx in inter
            ]
        )
        return LaneLineInPixels(center_pairs, lane_left.image_size), center_scaling

    def compute_center_lane_and_offset(self, other, lane_width=3.7):
        """
        Estimate both the yaw error and the vehicle offset via the lanes

        Assumes lanes are locally straight (but could be globally curved)
        """
        center_lane, center_scaling = self.compute_center_lane(other, lane_width)
        offset = np.zeros_like(center_lane._points)
        offset[:, 1] = self.image_size[1] / 2
        cc = center_lane._points - offset
        cs = center_scaling
        nrows_offset = 5
        nrows_yaw = 10
        yaw_offset = 0  # TODO
        lateral_offset = np.mean(cc[:nrows_offset, 1] * cs[:nrows_offset])
        return center_lane, lateral_offset, yaw_offset
