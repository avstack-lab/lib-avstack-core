# @Author: Spencer Hallyburton <spencer>
# @Date:   2021-02-04
# @Filename: bbox_util.py
# @Last modified by:   spencer
# @Last modified time: 2021-08-11


import logging
from copy import copy, deepcopy

import numpy as np
import quaternion
from scipy import sparse
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from avstack import calibration, exceptions
from avstack.geometry import transformations as tforms

from ..calibration import read_calibration_from_line
from .base import q_mult_vec
from .coordinates import CameraCoordinates, LidarCoordinates, StandardCoordinates
from .primitives import (
    Origin,
    Rotation,
    Transform,
    Translation,
    Vector,
    get_origin_from_line,
)


R_stan_to_cam = StandardCoordinates.get_conversion_matrix(CameraCoordinates)
q_stan_to_cam = quaternion.from_rotation_matrix(R_stan_to_cam)
R_cam_to_stan = R_stan_to_cam.T
q_cam_to_stan = q_stan_to_cam.conjugate()

NominalOriginStandard = Origin(np.zeros((3,)), np.quaternion(1))
NominalOriginCamera = Origin(np.zeros((3,)), q_stan_to_cam)

NominalRotation = Rotation(np.quaternion(1), origin=NominalOriginStandard)
NominalTranslation = Translation([0, 0, 0], origin=NominalOriginStandard)
NominalTransform = Transform(NominalRotation, NominalTranslation)


# ==============================================================================
# Define bounding boxes
# ==============================================================================


def get_boxes_from_file(box_file_path):
    with open(box_file_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    boxes = []
    for line in lines:
        boxes.append(get_box_from_line(line))
    return boxes


def get_box_from_line(line):
    items = line.split()
    try:
        box_type = items[0]
        if box_type == "box2d":
            xmin, ymin, xmax, ymax = [float(i) for i in items[1:5]]
            calib = read_calibration_from_line(" ".join(items[5:]))
            box = Box2D([xmin, ymin, xmax, ymax], calib)
        elif box_type == "box3d":
            h, w, l, x, y, z, qw, qx, qy, qz = [float(i) for i in items[1:11]]
            q = np.quaternion(qw, qx, qy, qz)
            origin = get_origin_from_line(" ".join(items[11:]))
            box = Box3D([h, w, l, [x, y, z], q], origin)
        else:
            raise NotImplementedError(box_type)
    except Exception as e:
        print(items)
        raise e
    return box


def get_segmask_from_line(line):
    items = line.split(" ")
    assert items[0] == "segmask-2d", items[0]
    mat_type = items[1]
    if mat_type == "coo-matrix":
        shape = (int(items[2].split("-")[0]), int(items[2].split("-")[1]))
        rows = np.array([float(v) for v in items[3].split("-")])
        cols = np.array([float(v) for v in items[4].split("-")])
        data = np.array([float(v) for v in items[5].split("-")])
        mask = sparse.coo_matrix((data, (rows, cols)), shape=shape).toarray()
    else:
        raise NotImplementedError(mat_type)
    calib = read_calibration_from_line(" ".join(items[6:]))
    return SegMask2D(mask, calib)


def wrap_minus_pi_to_pi(phases):
    phases = (phases + np.pi) % (2 * np.pi) - np.pi
    return phases


class SegMask2D:
    def __init__(self, mask, calibration) -> None:
        self.calibration = calibration
        self.mask = mask
        self.sparse_mask = sparse.coo_matrix(self.mask)
        self.img_shape = calibration.img_shape

    @property
    def data(self):
        return self.mask

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "SegMask"

    def __eq__(self, other: object) -> bool:
        if self.img_shape == other.img_shape:
            return (self.sparse_mask != other.sparse_mask).nnz == 0
        else:
            return False

    def sparse_mask_as_string(self):
        # TODO: convert this to a CSC matrix to be even more compressed
        shape_str = "-".join([str(v) for v in self.sparse_mask.shape])
        row_str = "-".join([str(v) for v in self.sparse_mask.row])
        col_str = "-".join([str(v) for v in self.sparse_mask.col])
        data_str = "-".join([str(v) for v in self.sparse_mask.data.astype(int)])
        sm_str = f"coo-matrix {shape_str} {row_str} {col_str} {data_str}"
        return sm_str

    def format_as_string(self) -> str:
        return f"segmask-2d {self.sparse_mask_as_string()} {self.calibration.format_as_string()}"


class Box2D:
    def __init__(self, box2d, calibration):
        self.calibration = calibration
        if (box2d[2] < box2d[0]) or (box2d[3] < box2d[1]):
            raise exceptions.BoundingBoxError(box2d)
        self.xmin = box2d[0]
        self.ymin = box2d[1]
        self.xmax = box2d[2]
        self.ymax = box2d[3]
        self.box2d = np.asarray(box2d)
        self.center = np.asarray(
            [(self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2]
        )
        self.corners = np.array(
            [
                [self.xmin, self.ymin],
                [self.xmin, self.ymax],
                [self.xmax, self.ymax],
                [self.xmax, self.ymin],
            ]
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Box2D: [%.2f, %.2f, %.2f, %.2f]" % (
            self.xmin,
            self.ymin,
            self.xmax,
            self.ymax,
        )

    def __eq__(self, other):
        return np.allclose(self.box2d, other.box2d)

    def deepcopy(self):
        return deepcopy(self)

    def check_valid(self, im_h, im_w):
        return self._x_valid(im_w) and self._y_valid(im_h)

    def _x_valid(self, im_w):
        return (
            (0 <= self.xmin < im_w)
            and (0 <= self.xmax < im_w)
            and (self.xmin < self.xmax)
        )

    def _y_valid(self, im_h):
        return (
            (0 <= self.ymin < im_h)
            and (0 <= self.ymax < im_h)
            and (self.ymin < self.ymax)
        )

    @property
    def angles(self):
        """use the calibration matrix to get az, el angles

        Procedure:
        1. Get center of bounding box (TODO: allow for centroid of mask)
        2. Change origin to principal point
        3. Use pixel size (rad/pixel) to get angles off center
        """
        return self.calibration.pixel_to_angle(self.center)

    @property
    def w(self):
        return self.xmax - self.xmin

    @property
    def h(self):
        return self.ymax - self.ymin

    @property
    def box2d_xywh(self):
        return [self.xmin, self.ymin, self.w, self.h]

    def IoU(self, other):
        inter = box_intersection(self.box2d, other.box2d)
        union = box_union(self.box2d, other.box2d)
        return inter / union

    def squeeze(self, im_h, im_w):
        x_min_s = int(max(0, min(self.xmin, im_w - 1)))
        y_min_s = int(max(0, min(self.ymin, im_h - 1)))
        x_max_s = int(max(0, min(self.xmax, im_w - 1)))
        y_max_s = int(max(0, min(self.ymax, im_h - 1)))
        return Box2D([x_min_s, y_min_s, x_max_s, y_max_s], self.calibration)

    def format_as_string(self):
        return f"box2d {self.xmin} {self.ymin} {self.xmax} {self.ymax} {self.calibration.format_as_string()}"


class Box3D:
    def __init__(
        self,
        box3d,
        origin=NominalOriginStandard,
        where_is_t="center",
        enforce_mins=True,
    ):
        """
        q: q_XXX_to_box, normally XXX := sensor origin
        """
        self.origin = origin
        self.where_is_t = where_is_t
        if len(box3d) == 5:
            h, w, l, t, q = box3d
            if not isinstance(t, list):
                if isinstance(t, np.ndarray):
                    t = t.flatten()
                else:
                    t = t.vector.flatten()
        elif len(box3d) == 7:
            h, w, l = box3d[0:3]
            t = box3d[3:6]
            q = box3d[6]
        else:
            raise NotImplementedError
        if enforce_mins:
            h = max(0.5, h)
            w = max(0.5, w)
            l = max(0.5, l)
        self.h, self.w, self.l = h, w, l
        self.t = Translation(np.array(t), origin=origin)
        assert isinstance(q, np.quaternion) or (len(q) == 4)
        self.rot = Rotation(q, origin=origin)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            "Box3D=[h: %.2f, w: %.2f, l: %.2f] x (x: %.2f y: %.2f, z: %.2f)\n  q: %s with origin: %s"
            % (
                self.h,
                self.w,
                self.l,
                self.t[0],
                self.t[1],
                self.t[2],
                str(self.q),
                str(self.origin),
            )
        )

    def __eq__(self, other):
        c1 = (self.h == other.h) and (self.w == other.w) and (self.l == other.l)
        c2 = self.t == other.t
        c3 = self.q == other.q
        return c1 and c2 and c3

    def deepcopy(self):
        return deepcopy(self)

    def allclose(self, other):
        c1 = (
            np.isclose(self.h, other.h)
            and np.isclose(self.w, other.w)
            and np.isclose(self.l, other.l)
        )
        c2 = self.t.allclose(other.t)
        c3 = self.rot.allclose(other.rot)
        return c1 and c2 and c3

    @property
    def size(self):
        return np.array([self.h, self.w, self.l])

    @property
    def volume(self):
        return self.h * self.w * self.l

    @property
    def center(self):
        return self.t

    @property
    def center_global(self):
        t_glob = deepcopy(self.t)
        t_glob.change_origin(NominalOriginStandard)
        if self.where_is_t == "bottom":
            t_glob.z += self.h / 2  # to convert to center
        return t_glob

    @property
    def attitude(self):
        return self.q

    @property
    def q(self):
        return self.rot.q

    @property
    def R(self):
        return tforms.transform_orientation(self.q, "quat", "dcm")

    @property
    def yaw(self):
        """Here, yaw is 0 forward which is NOT the KITTI standard

        NOTE: will we ever truly want yaw in the global frame?

        yaw           := q_OR1_to_obj
        self.q        := q_O_to_obj
        self.origin.q := q_OR1_to_O
        """
        q_OR1_to_O = self.origin.q
        q_O_to_obj = self.q
        q_OR1_to_obj = q_O_to_obj * q_OR1_to_O
        yaw = tforms.transform_orientation(q_OR1_to_obj, "quat", "euler")[2]
        return yaw

    @property
    def yaw_this_frame(self):
        """Get yaw in the current reference frame"""
        q_O_to_obj = self.q
        yaw = tforms.transform_orientation(q_O_to_obj, "quat", "euler")[2]
        return yaw

    @property
    def corners(self):
        return compute_box_3d_corners(box3d=self)

    @property
    def corners_global(self):
        """Compute global corners by inverting the origin"""
        return (
            self.origin.inv() @ self.corners
        )  # invert them to global coordinate frame

    @property
    def corners_global_without_pitch_roll(self):
        raise NotImplementedError

    def add_noise(self, noise_variance):
        """Add noise to each component

        NOTE: no noise can be added to angle yet
        NOTE: also assumes no cross correlations for now

        noise must be Gaussian of the form: [h, w, l, x, y, z]
        """
        if noise_variance is None:
            return
        assert len(noise_variance) == 6
        noise_samples = np.random.randn(len(noise_variance))
        noisy_vals = [np.sqrt(nv) * ns for nv, ns in zip(noise_variance, noise_samples)]
        self.h += noisy_vals[0]
        self.w += noisy_vals[1]
        self.l += noisy_vals[2]
        self.t += np.array(noisy_vals[3:6])

    def center_box(self, inplace=True):
        assert inplace, "Only doing inplace for now"
        if self.where_is_t == "center":
            pass
        else:
            O1 = deepcopy(self.origin)
            self.t.change_origin(NominalOriginStandard)
            self.t += np.array([0, 0, self.h / 2])
            self.t.change_origin(O1)
            self.where_is_t = "center"

    def change_origin(self, origin_new):
        self.t.change_origin(origin_new)
        self.rot.change_origin(origin_new)
        self.origin = origin_new

    def IoU(self, other, metric="3D", run_angle_check=True, error_on_angle_check=False):
        """
        IMPORTANT NOTE: THIS METRIC ONLY WORKS WITH A YAW ANGLE
        (YAW AS DEFINED IN THE STANDARD FRAME) IT DOES NOT WORK
        WHEN THERE ARE PITCH AND ROLL ANGLES

        Check those angles in the global frame as follows:
        self.q        := q_O_2_V
        self.origin.q := q_OG_2_O

        q_OG_2_V = q_O_2_V * q_OG_2_O

        Which simplifies to:
        q_OG_2_V = self.q * self.origin.q

        """
        if self.center_global.distance(other.center_global) > (
            max(self.size) / 2 + max(other.size) / 2
        ):
            return 0.0

        if run_angle_check:
            eps = 0.05
            q_stand_to_box = self.q * self.origin.q
            if (abs(q_stand_to_box.x) > eps) or (abs(q_stand_to_box.y) > eps):
                msg = (
                    f"Not a good idea to run this IoU equation"
                    f" when there are more than just yaw angles in the"
                    f" global frame...{q_stand_to_box}"
                )
                if error_on_angle_check:
                    raise ValueError(msg)
                else:
                    logging.warning(msg)

        if metric == "3D":
            # c1 = self.corners_global_without_pitch_roll
            # c2 = other.corners_global_without_pitch_roll
            c1 = self.corners_global
            c2 = other.corners_global
            inter = box_intersection(c1, c2, up="+z")
            union = box_union(c1, c2, up="+z")
            iou = inter / union
            if (iou < 0) or (iou > 1.5):
                iou = max(
                    1.0, iou
                )  # BUG: IoU can be >1.0 if roll and pitch angles are non-zero since the formula used approximates intersection equation as only yaw angle
                import ipdb

                ipdb.set_trace()
                inter = box_intersection(c1, c2, up="+z")
                union = box_union(c1, c2, up="+z")
                raise RuntimeError("Invalid iou output")
        else:
            raise NotImplementedError(metric)
        return max(0.0, min(1.0, iou))

    def rotate(self, q):
        """Rotates the attitude AND the translation of the box"""
        if isinstance(q, np.quaternion):
            pass
        elif isinstance(q, np.ndarray) and q.shape == (3, 3):
            q = quaternion.from_rotation_matrix(q)
        elif isinstance(q, Rotation):
            q = q.q
        else:
            raise NotImplementedError(type(q))
        t = q_mult_vec(q, self.t.vector)
        return Box3D([self.h, self.w, self.l, t, q * self.q], self.origin)

    def rotate_attitude(self, q):
        """Rotates the attitude of the box only"""
        raise NotImplementedError

    def translate(self, L):
        """Translates the position of the box"""
        if isinstance(L, np.ndarray):
            T = L
        else:
            T = L.vector
        return Box3D([self.h, self.w, self.l, self.t + T, self.q], self.origin)

    def project_to_2d_bbox(self, calib, squeeze=True, check_origin=True):
        """Project 3D bounding box into a 2D bounding box"""
        return proj_3d_bbox_to_2d_bbox(self, calib, check_origin).squeeze(
            calib.height, calib.width
        )

    def project_corners_to_2d_image_plane(self, calib, squeeze=True):
        """Project 3D bounding box corners only image plane"""
        return proj_3d_bbox_to_image_plane(self, calib, squeeze=squeeze)

    def format_as_string(self):
        return (
            f"box3d {self.h} {self.w} {self.l} {self.t.vector[0]} "
            f"{self.t.vector[1]} {self.t.vector[2]} {self.q.w} "
            f"{self.q.x} {self.q.y} {self.q.z} {self.origin.format_as_string()}"
        )


# ==============================================================================
# Other Utilities
# ==============================================================================


def polygon_clip(subjectPolygon, clipPolygon):
    """Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**
    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]
    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList


def poly_area(x, y):
    """Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates"""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convex_hull_intersection(p1, p2):
    """Compute area of two convex hull's intersection area.
    p1,p2 are a list of (x,y) tuples of hull vertices.
    return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def box_area(corners):
    """Compute 2D box area
    corners - an array of [xmin, ymin, xmax, ymax]
    """
    return (corners[2] - corners[0] + 1) * (corners[3] - corners[1] + 1)


def box_volume(corners):
    """Compute 3D box volume
    corners1: numpy array (8,3), assume up direction is +Z
    X - forward, Y - left, Z - up
    Corners start in upper top left and go counterclockwise on top plane
    then lower top left and go counterclockwise on bottom plane


          0-----3
         -     --
        -     - -
       -     -  -
      -     -   -
     -     -    -
    1-----2     7
    -     -    -
    -     -   -
    -     -  -
    -     - -
    -     --
    5-----6
    """
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def _box_intersection_2d(corners1, corners2):
    """Compute intersection of 2D boxes"""
    xA = max(corners1[0], corners2[0])
    yA = max(corners1[1], corners2[1])
    xB = min(corners1[2], corners2[2])
    yB = min(corners1[3], corners2[3])
    return max(0, xB - xA + 1) * max(0, yB - yA + 1)


def sort_corners_invariant(mat):
    """Sort matrices to be permutation-invariant

    Sort according to the proper bounding box corner ordering
    Need to flip 2 <--> 3 and  6 <--> 7 to preserve order

    Negatives to allow sorting descending
    """
    # get ordering over the columns manually...
    mat *= -1
    mat = -np.sort(mat.view("f8,f8,f8"), order=["f2", "f1", "f0"], axis=0).view(
        np.float64
    )
    mat[[2, 3], :] = mat[[3, 2], :]
    mat[[6, 7], :] = mat[[6, 7], :]
    return mat


def _box_intersection_3d(corners1, corners2, up="+z"):
    """Compute 3D bounding box IoU.

    From: https://github.com/AlienCat-K/3D-IoU-Python/blob/master/3D-IoU-Python.py
    Input:
        corners1: numpy array (8,3), assume up direction is +Z
        corners2: numpy array (8,3), assume up direction is +Z
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

          0-----3
         -     --
        -     - -
       -     -  -
      -     -   -
     -     -    -
    1-----2     7
    -     -    -
    -     -   -
    -     -  -
    -     - -    z    x
    -     --     |   -
    5-----6      |  -
                 | -
        y <------|-

    Can enforce the order by sorting in the starndard coordinates as follows:

    z, descending -- first
    y, descending -- second
    x, descending -- third
    """
    assert corners1.shape == corners2.shape, f"{corners1}, {corners2}"
    if np.any(np.isnan(corners1)) or np.any(np.isnan(corners2)):
        print(corners1)
        print(corners2)
        raise RuntimeError("NaN somewhere!")

    # Add some very small noise
    noise = 1e-12 * np.linalg.norm(corners2[0, :] - corners2[1, :])
    corners2 = copy(corners2) + noise

    # Put in the right coordinate frame
    if up == "-y":
        R = CameraCoordinates.get_conversion_matrix(LidarCoordinates)
        corners1 = (R @ corners1.T).T
        corners2 = (R @ corners2.T).T
    elif up == "+z":
        pass
    else:
        raise NotImplementedError(up)

    # -- Get the top-down intersection (BEV)
    rect1 = [(-corners1[i, 1], corners1[i, 0]) for i in range(4)]
    rect2 = [(-corners2[i, 1], corners2[i, 0]) for i in range(4)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    if np.all(rect1 == rect2):
        inter = None
        inter_area = box_volume(corners1)
    else:
        try:
            inter_bev, inter_area_bev = convex_hull_intersection(rect1, rect2)
        except QhullError as e:
            inter = None
            inter_area = 0

    # -- Get the vertical intersection ratio
    top1, bot1 = max(corners1[:, 2]), min(corners1[:, 2])
    top2, bot2 = max(corners2[:, 2]), min(corners2[:, 2])
    z_overlap = max(0.0, min(top1, top2) - max(bot1, bot2))

    return inter_area_bev * z_overlap


def box_intersection(corners1, corners2, up="+z"):
    if len(corners1) != len(corners2):
        raise RuntimeError(f"{len(corners1)} vs {len(corners2)} corners")
    if len(corners1) == 4:
        return _box_intersection_2d(corners1, corners2)
    elif len(corners1) == 8:
        return _box_intersection_3d(corners1, corners2, up=up)
    else:
        raise NotImplementedError(f"{len(corners1)} not implemented")


def _box_union_2d(corners1, corners2):
    return (
        box_area(corners1) + box_area(corners2) - box_intersection(corners1, corners2)
    )


def _box_union_3d(corners1, corners2, up):
    return (
        box_volume(corners1)
        + box_volume(corners2)
        - box_intersection(corners1, corners2, up=up)
    )


def box_union(corners1, corners2, up="+z"):
    if len(corners1) != len(corners2):
        raise RuntimeError(f"{len(corners1)} vs {len(corners2)} corners")
    if len(corners1) == 4:
        return _box_union_2d(corners1, corners2)
    elif len(corners1) == 8:
        return _box_union_3d(corners1, corners2, up=up)
    else:
        raise NotImplementedError(f"{len(corners1)} not implemented")


def compute_box_size(corners_3d, heading_angle):
    """
    Takes an object's set of corners and gets h, w, l

    Corners are specified in the image rectangular coordinates
    Assumes corner convention in order to specify which is which
    """
    R = tforms.roty(heading_angle)
    box_oriented = corners_3d @ R  # invert the rotation
    l = max(box_oriented[:, 0]) - min(box_oriented[:, 0])
    h = max(box_oriented[:, 1]) - min(box_oriented[:, 1])
    w = max(box_oriented[:, 2]) - min(box_oriented[:, 2])
    return (l, w, h)


def compute_box_3d_corners(box3d, yaw=None):
    """Computes the 3D bounding box in the box's coordinate frame

    X - forward, Y - left, Z - up
    Corners start in upper top left and go counterclockwise on top plane
    then lower top left and go counterclockwise on bottom plane


          0-----3
         -     --
        -     - -
       -     -  -
      -     -   -
     -     -    -
    1-----2     7
    -     -    -
    -     -   -
    -     -  -
    -     - -
    -     --
    5-----6

    Starts with box in standard frame, then converts to origin

    Conversion process is as follows:

    box3d.q         := q_O_2_V --> origin to vehicle body frame
    box3d.t         := x_O_2_V_in_O -- > origin to vehicle body frame in origin frame
    corners_3d_base := x_V_2_pts_in_V --> vehicle body frame to corner points in body frame

    To get corners in the origin frame, do the following:
    x_V_2_pts_in_O = q_O_2_V.conjugate() * x_V_2_pts_in_V
    x_O_2_pts_in_O = x_V_2_pts_in_O + x_O_2_V_in_O

    This expands to:
    corners_out =  box3d.q.conjugate() * corners_3d_base + box3d.t

    """
    # -- get 3d coordinates of a nominal box in standard coordinates
    l = box3d.l
    w = box3d.w
    h = box3d.h
    x_corners = np.array(
        [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2], dtype=np.float64
    )
    y_corners = np.array(
        [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2], dtype=np.float64
    )
    z_corners = np.array([h, h, h, h, 0, 0, 0, 0], dtype=np.float64)
    if box3d.where_is_t == "bottom":
        pass
    elif box3d.where_is_t == "center":
        z_corners -= h / 2
    else:
        raise NotImplementedError(box3d.where_is_t)

    # -- convert to the desired coordinates in the box's frame
    corners_3d_base = np.concatenate(
        (x_corners[:, None], y_corners[:, None], z_corners[:, None]), axis=1
    )
    # NOTE: no need to do anything with the new origin here...corners will assume
    # the origin of the vehicle

    # -- apply box rotation quaternion, then translation
    corners_3d = (
        q_mult_vec(box3d.q.conjugate(), corners_3d_base)
        + np.asarray(box3d.t)[:, None].T
    )

    return corners_3d


def compute_orientation_3d(box3d, P):
    """Takes an object and a projection matrix (P) and projects the 3d
    object orientation vector into the image plane.
    Returns:
        orientation_2d: (2,2) array in left image coord.
        orientation_3d: (2,3) array in in rect camera coord.
    """

    # compute rotational matrix around yaw axis
    R = tforms.roty(box3d.yaw)

    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, box3d.l], [0, 0], [0, 0]])

    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0, :] = orientation_3d[0, :] + box3d.t[0]
    orientation_3d[1, :] = orientation_3d[1, :] + box3d.t[1]
    orientation_3d[2, :] = orientation_3d[2, :] + box3d.t[2]

    # vector behind image plane?
    if np.any(orientation_3d[2, :] < 0.1):
        orientation_2d = None
        return orientation_2d, np.transpose(orientation_3d)

    # project orientation into the image plane
    orientation_2d = tforms.project_to_image(np.transpose(orientation_3d), P)
    return orientation_2d, np.transpose(orientation_3d)


def in_hull(p, hull):
    from scipy.spatial import Delaunay

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def proj_3d_bbox_to_image_plane(
    box3d, calib_img, check_origin=True, squeeze=False, squeeze_method="lines"
):
    """
    Squeeze:
    Using the squeeze via-line method, the following combinations of points
    must be checked

          0-----3
         -     --
        -     - -
       -     -  -
      -     -   -
     -     -    -
    1-----2     7
    -     -    -
    -     -   -
    -     -  -
    -     - -
    -     --
    5-----6

    All pairs:
      0 -- 1
      0 -- 3
      0 -- 4
      1 -- 2
      1 -- 5
      2 -- 3
      2 -- 6
      3 -- 7
      4 -- 5
      4 -- 7
      5 -- 6
      6 -- 7
    """
    # Get 3D box points
    if check_origin:
        box3d = deepcopy(box3d)
        box3d.change_origin(calib_img.origin)
    box3d_pts_3d = box3d.corners

    # Project into image plane
    corners_3d_in_image = tforms.project_to_image(box3d_pts_3d, calib_img.P)
    if False:
        cix = corners_3d_in_image[:, 0].copy()
        ciy = corners_3d_in_image[:, 1].copy()
        if squeeze_method == "lines":
            # Squeeze assuming lines between points
            pt_is_off = ((cix < 0) | (cix >= calib_img.width)) | (
                (ciy < 0) | (ciy >= calib_img.height)
            )
            pt_pairs = [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (2, 3),
                (2, 6),
                (3, 7),
                (4, 5),
                (4, 7),
                (5, 6),
                (6, 7),
            ]
            for p1, p2 in pt_pairs:
                if pt_is_off[p1] and pt_is_off[p2]:
                    continue  # don't update both -- only 1 at a time
                elif pt_is_off[p1]:
                    poff = p1
                    pon = p2
                elif pt_is_off[p2]:
                    poff = p2
                    pon = p1
                else:
                    continue
                # The off point must move along the line to the on point
                dx_off_to_on = cix[pon] - cix[poff]
                dy_off_to_on = ciy[pon] - ciy[poff]
                m_y_x = dy_off_to_on / dx_off_to_on  # m is (delta y) / (delta x)
                m_x_y = dx_off_to_on / dy_off_to_on  # m is (delta x) / (delta y)
                # Find the point on this line where all conditions satisfied using iterative update (slow)
                ds_total = 0
                ds_max = 1000
                ds = 0.01
                cix_new = cix[poff].copy()
                ciy_new = ciy[poff].copy()
                while ds_total < ds_max:
                    # update point moving along line
                    if cix_new < 0:  # x needs to get more positive
                        cix_new += ds
                        ciy_new += ds * m_y_x
                    elif cix_new >= calib_img.width:  # x needs to get more negative
                        cix_new -= ds
                        ciy_new -= ds * m_y_x
                    elif ciy_new < 0:  # y needs to get more positive
                        cix_new += ds * m_x_y
                        ciy_new += ds
                    elif ciy_new >= calib_img.height:  # y needs to get more negative
                        cix_new -= ds * m_x_y
                        ciy_new -= ds
                    else:
                        break  # were done!
                    ds_total += ds
                else:
                    import ipdb

                    ipdb.set_trace()
                    raise RuntimeError("Could not find suitable point")
                corners_3d_in_image[poff, 0] = cix_new
                corners_3d_in_image[poff, 1] = ciy_new
        elif squeeze_method == "points":
            # Squeeze by just pushing points to edge
            corners_3d_in_image[:, 0] = np.maximum(
                0, np.minimum(cix, calib_img.width - 1)
            )
            corners_3d_in_image[:, 1] = np.maximum(
                0, np.minimum(ciy, calib_img.height - 1)
            )
        else:
            raise NotImplementedError(squeeze_method)
    return corners_3d_in_image


def proj_3d_bbox_to_2d_bbox(box3d, calib_img, check_origin=True):
    # Get 3D box points
    corners_3d_in_image = proj_3d_bbox_to_image_plane(box3d, calib_img, check_origin)

    # Get mins and maxes to make 2D bbox
    xmin, xmax = np.min(corners_3d_in_image[:, 0]), np.max(corners_3d_in_image[:, 0])
    ymin, ymax = np.min(corners_3d_in_image[:, 1]), np.max(corners_3d_in_image[:, 1])
    box2d = np.array([xmin, ymin, xmax, ymax])
    return Box2D(box2d, calib_img)
