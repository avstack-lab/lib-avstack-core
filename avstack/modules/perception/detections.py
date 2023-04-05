# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-27
# @Last Modified by:   spencer@primus
# @Last Modified date: 2022-08-25
# @Description:
"""

"""
from typing import List

import numpy as np
from scipy.interpolate import interp1d

from avstack.datastructs import DataContainer
from avstack.geometry import (
    Box2D,
    Box3D,
    NominalOriginStandard,
    SegMask2D,
    Translation,
    bbox,
)
from avstack.geometry.transformations import spherical_to_cartesian, cartesian_to_spherical


# detection_map = {'vehicle':'car', 'car':'car', 'pedestrian':'pedestrian', 'cyclist':'cyclist',
#                  'bicyclist':'cyclist', 'truck':'truck', 'bus':'bus', 'train':'train',
#                  'motorcycle':'motorcycle', 'rider':'pedestrian', 'bicycle':'cyclist',
#                  'person':'pedestrian', 'trailer':'trailer', 'construction_vehicle':'truck'}


def get_detections_from_file(det_file_path):
    with open(det_file_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    dets = []
    for line in lines:
        dets.append(get_detection_from_line(line))
    return dets


def get_detection_from_line(line):
    items = line.split()
    det_type = items[0]
    sID, obj_type, score = items[1:4]
    if det_type == "box-detection":
        box = bbox.get_box_from_line(" ".join(items[4:]))
        det = BoxDetection(sID, box, obj_type, score)
    elif det_type == "mask-detection":
        box = bbox.get_box_from_line(" ".join(items[4:34]))
        mask = bbox.get_segmask_from_line(" ".join(items[34:]))
        det = MaskDetection(sID, box, mask, obj_type, score)
    elif det_type == "centroid-detection":
        n_dims = int(items[4])
        centroid = np.array([float(d) for d in items[5 : 5 + n_dims]])
        det = CentroidDetection(sID, centroid, obj_type, score)
    elif det_type == "razel-detection":
        razel = np.array([float(d) for d in items[4 : 7]])
        det = RazelDetection(sID, razel, obj_type, score)
    elif det_type == "razelrrt-detection":
        razelrrt = np.array([float(d) for d in items[4 : 8]])
        det = RazelRrtDetection(sID, razelrrt, obj_type, score)
    else:
        raise NotImplementedError(det_type)
    return det


def format_data_container_as_string(DC):
    dets_strings = " ".join(["DETECTION " + det.format_as_string() for det in DC.data])
    return (
        f"datacontainer {DC.frame} {DC.timestamp} {DC.source_identifier} "
        f"{dets_strings}"
    )


def get_data_container_from_line(line):
    items = line.split()
    assert items[0] == "datacontainer"
    frame = int(items[1])
    timestamp = float(items[2])
    source_identifier = items[3]
    detections = [get_detection_from_line(det) for det in line.split("DETECTION")[1:]]
    return DataContainer(frame, timestamp, detections, source_identifier)


class Detection_:
    def __init__(self, source_identifier, obj_type, score, check_type):
        self.source_identifier = source_identifier
        self.obj_type = obj_type
        self.score = score
        self.check_type = check_type

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
        return self.data.origin

    def change_origin(self, origin):
        self.data.change_origin(origin)

    def __str__(self):
        return f"{self.obj_type} detection from sensor {self.source_identifier}\n{self.data}"

    def __repr__(self):
        return self.__str__()


class CentroidDetection(Detection_):
    def __init__(
        self, source_identifier, centroid, obj_type=None, score=None, check_type=False
    ):
        super().__init__(source_identifier, obj_type, score, check_type)
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

    @centroid.setter
    def centroid(self, centroid):
        if self.check_type:
            if not isinstance(centroid, np.ndarray):
                raise TypeError(
                    f"Input centroid of type {type(centroid)} is not of an acceptable type"
                )
        self._centroid = centroid

    def format_as_string(self):
        """Format data elements"""
        return f"centroid-detection {self.source_identifier} {self.obj_type} {self.score} {len(self.centroid)} {' '.join([str(d) for d in self.centroid])}"


class RazelDetection(Detection_):
    def __init__(
        self, source_identifier, razel, obj_type=None, score=None, check_type=False
    ):
        super().__init__(source_identifier, obj_type, score, check_type)
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
        if self.check_type:
            if not isinstance(razel, np.ndarray):
                raise TypeError(
                    f"Input razel of type {type(razel)} is not of an acceptable type"
                )
        self._razel = razel
    
    @property
    def xyz(self):
        x, y, z = spherical_to_cartesian(self.razel)
        return np.array([x, y, z])
        
    def format_as_string(self):
        """Format data elements"""
        return f"razel-detection {self.source_identifier} {self.obj_type} {self.score} {' '.join([str(d) for d in self.razel])}"


class RazelRrtDetection(Detection_):
    def __init__(
        self, source_identifier, razelrrt, obj_type=None, score=None, check_type=False
    ):
        super().__init__(source_identifier, obj_type, score, check_type)
        self.razelrrt = razelrrt

    @property
    def data(self):
        return self.razelrrt

    @property
    def razelrrt(self):
        return self._razelrrt
    
    @property
    def z(self):
        return self.razelrrt

    @razelrrt.setter
    def razelrrt(self, razelrrt):
        if self.check_type:
            if not isinstance(razelrrt, np.ndarray):
                raise TypeError(
                    f"Input razelrrt of type {type(razelrrt)} is not of an acceptable type"
                )
        self._razelrrt = razelrrt

    @property
    def xyzrrt(self):
        x, y, z = spherical_to_cartesian(self.razelrrt[:3])
        return np.array([x, y, z, self.razelrrt[3]])
    
    @property
    def xyz(self):
        x, y, z = spherical_to_cartesian(self.razelrrt[:3])
        return np.array([x, y, z])
        
    def format_as_string(self):
        """Format data elements"""
        return f"razelrrt-detection {self.source_identifier} {self.obj_type} {self.score} {' '.join([str(d) for d in self.razelrrt])}"


class JointBoxDetection(Detection_):
    def __init__(
        self,
        source_identifier,
        box2d,
        box3d,
        obj_type=None,
        score=None,
        check_type=False,
    ):
        super().__init__(source_identifier, obj_type, score, check_type)
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
        if self.check_type:
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
        if self.check_type:
            if not isinstance(box3d, Box3D):
                raise TypeError(
                    f"Input box of type {type(box3d)} is not of an acceptable type"
                )
        self._box_3d = box3d


class JointBoxDetectionAndOther(JointBoxDetection):
    pass


class BoxDetection(Detection_):
    def __init__(
        self, source_identifier, box, obj_type=None, score=None, check_type=False
    ):
        super().__init__(source_identifier, obj_type, score, check_type)
        self.box = box

    @property
    def data(self):
        return self.box

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, box):
        if self.check_type:
            if not (isinstance(box, Box2D) or isinstance(box, Box3D)):
                raise TypeError(
                    f"Input box of type {type(box)} is not of an acceptable type"
                )
        self._box = box

    @property
    def box3d(self):
        if self.check_type:
            if not isinstance(self.box, Box3D):
                raise ValueError
        return self.box

    @property
    def box2d(self):
        if self.check_type:
            if not isinstance(self.box, Box2D):
                raise ValueError
        return self.box

    @property
    def z(self):
        return self.box
    
    def format_as_string(self):
        """Convert to vehicle state and format"""
        return f"box-detection {self.source_identifier} {self.obj_type} {self.score} {self.box.format_as_string()}"


class MaskDetection(Detection_):
    def __init__(
        self, source_identifier, box, mask, obj_type=None, score=None, check_type=False
    ):
        super().__init__(source_identifier, obj_type, score, check_type)
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
        if self.check_type:
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
        if self.check_type:
            if not (isinstance(mask, SegMask2D)):
                raise TypeError(
                    f"Input mask of type {type(mask)} is not of an acceptable type"
                )
        self._mask = mask

    @property
    def box2d(self):
        if self.check_type:
            if not isinstance(self.box, Box2D):
                raise ValueError
        return self.box

    def format_as_string(self):
        """Convert to vehicle state and format"""
        return (
            f"mask-detection {self.source_identifier} {self.obj_type} {self.score} "
            + f"{self.box.format_as_string()} {self.mask.format_as_string()}"
        )


class OtherDetection(Detection_):
    def __init__(self, source_identifier, data, obj_type=None, score=None):
        super().__init__(source_identifier, obj_type, score)
        self.data = data


class LaneLineInSpace:
    """Lane line in terms of cartesian space"""

    def __init__(self, points: List[Translation]):
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
        return np.array([p.x for p in self._points])

    @property
    def y(self):
        return np.array([p.y for p in self._points])

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
                obj.y <= self._points[i].y
                and obj.y >= other._points[i].y
                and obj.x >= self._points[i].x - 2
                and obj.x <= self._points[i].x + 2
            ):
                return True
            elif (
                obj.y >= self._points[i].y
                and obj.y <= other._points[i].y
                and obj.x >= self._points[i].x - 2
                and obj.x <= self._points[i].x + 2
            ):
                return True
        return False

    def compute_center_lane(self, other):
        # determine which is right and left lanes
        lane_left = self if np.mean(self.y) >= np.mean(other.y) else other
        lane_right = self if lane_left == other else other
        # compute center lane
        min_fwd, max_fwd = max(lane_left[0].x, lane_right[0].x), min(
            lane_left[-1].x, lane_right[-1].x
        )
        if not (max_fwd > min_fwd >= 0):
            # try to cut lane in half...
            min_fwd = max(lane_left[0].x, lane_right[0].x)
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
            Translation([x, y, 0], origin=NominalOriginStandard)
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
