# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-08-07
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-29
# @Description:
"""

"""
from copy import deepcopy

import numpy as np

import avstack.transformations as tforms
from avstack.geometry import NominalOriginStandard, get_origin_from_line


def read_calibration_from_line(line):
    elems = line.split(" ")
    assert elems[0] == "calibration", elems[0]
    origin = get_origin_from_line(" ".join(elems[1:9]))
    if len(elems) > 9:
        assert elems[9] == "intrinsics", elems[9]
        P_cam = np.reshape(np.array([float(e) for e in elems[10:22]]), (3, 4))
        assert elems[22] == "img_shape"
        img_shape = tuple([int(e) for e in elems[23:25]])
        return CameraCalibration(origin, P_cam, img_shape)
    else:
        return Calibration(origin)


class Calibration:
    def __init__(self, origin):
        """
        T - extrinsic matrix (rotation + translation)

        extrinsic matrix is transformation from global to local
        """
        self.origin = origin

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Calibration with origin: {self.origin}"

    def format_as_string(self):
        return "calibration " + self.origin.format_as_string()

    def save_to_file(self, filename):
        if not filename.endswith(".txt"):
            filename = filename + ".txt"
        with open(filename, "w") as f:
            f.write(self.format_as_string())

    def change_origin(self, origin_new):
        self.origin = origin_new

    def transform_3d_to_3d(self, pts_3d, origin):
        """Transform points from self frame to frame specified by origin

        puts pts_3d from "local self" into global, then from global into "local other"

        self represents the transformtion for pts_3d glbal to local
        """
        try:
            pts_3d = deepcopy(pts_3d)
            pts_3d.change_origin(origin)
        except AttributeError as e:
            pts_3d = self.origin.change_points_origin(pts_3d, origin)
        return pts_3d

    def transform_3d_to_3d_inv(self, pts_3d, origin):
        """Opposite transformation of transform_3d_to_3d

        Origin represents the transformation for pts_3d global to local
        """

        try:
            pts_3d = deepcopy(pts_3d)
            pts_3d.change_origin(self.origin)
        except AttributeError as e:
            pts_3d = origin.change_points_origin(pts_3d, self.origin)
        return pts_3d


class CameraCalibration(Calibration):
    def __init__(
        self,
        origin,
        P,
        img_shape,
        fov_horizontal=None,
        fov_vertical=None,
        square_pixels=False,
    ):
        """
        P - intrinsic matrix
        T - extrinsic matrix (rotation + translation)
        img_shape - (height, width, channels) of image
        """
        self.img_shape = img_shape
        if len(img_shape) == 3:
            self.height, self.width, self.n_channels = img_shape
        else:
            self.height, self.width = img_shape
            self.n_channels = 1
        self.P = P
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)
        self.principal_point = np.array([self.c_u, self.c_v])  # in (x, y)
        # -- pixel size using linear approximation
        if not square_pixels:
            self.pixel_size_u = (
                fov_horizontal / (2 * self.f_u) if fov_horizontal is not None else None
            )
            self.pixel_size_v = (
                fov_vertical / (2 * self.f_v) if fov_vertical is not None else None
            )
        else:
            if (fov_horizontal is None) and (fov_vertical is None):
                self.pixel_size_u = None
                self.pixel_size_v = None
            elif (fov_horizontal is not None) and (fov_vertical is not None):
                pix_size = fov_horizontal / (2 * self.f_u)
                pix_size_2 = fov_vertical / (2 * self.f_v)
                if not np.isclose(pix_size, pix_size_2):
                    raise RuntimeError(
                        "Pixel size does not check out if square pixels assumed"
                    )
            elif fov_horizontal is not None:
                pix_size = fov_horizontal / (2 * self.f_u)
            else:
                pix_size = fov_vertical / (2 * self.f_v)
            self.pixel_size_u = self.pixel_size_v = pix_size
        super().__init__(origin)

    def __str__(self):
        return f"Camera Calibration with origin: {self.origin}; P:{self.P}"

    def format_as_string(self):
        P_as_str = " ".join([str(v) for v in np.ravel(self.P)])
        c_str = (
            "calibration "
            + self.origin.format_as_string()
            + " intrinsics "
            + P_as_str
            + f" img_shape {self.img_shape[0]} {self.img_shape[1]}"
        )
        return c_str

    def pixel_to_angle(self, x_pixels):
        """Takes pixel coordinates and get angles

        assumes x_pixels has (0,0) in the top left

        Returns the (az, el) angles
        """
        x_centered = self.principal_point - x_pixels
        # azel = x_centered * np.array([self.pixel_size_u, self.pixel_size_v])
        if len(x_centered.shape) == 1:
            azel = np.array(
                [
                    np.arctan2(x_centered[0], self.f_u),
                    np.arctan2(x_centered[1], self.f_v),
                ]
            )
        else:
            azel = np.zeros_like(x_centered)[:, :2]
            azel[:, 0] = np.arctan2(x_centered[:, 0], self.f_u)
            azel[:, 1] = np.arctan2(x_centered[:, 1], self.f_v)
        return azel

    def project_3d_points(self, pts_3d, origin_pts):
        """
        :T -- extrinsic matrix of the reference frame of the 3d points

        if T is none, pts_3d must be in the 3D reference frame centered on the camera
        """
        if isinstance(pts_3d, np.ndarray) and (len(pts_3d.shape) == 1):
            pts_3d = pts_3d[:, None].T
            squeeze_out = True
        else:
            squeeze_out = False
        pts_3d_img = self.transform_3d_to_3d_inv(pts_3d[:, :3], origin=origin_pts)
        pts_3d_hom = tforms.cart2hom(pts_3d_img)
        pts_2d = np.dot(pts_3d_hom, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        pts_2d_cam = pts_2d[:, 0:2]
        if pts_3d.shape[1] > 3:
            pts_2d_cam = np.hstack([pts_2d_cam, pts_3d[:, 3:]])
        if squeeze_out:
            pts_2d_cam = np.squeeze(pts_2d_cam)
        return pts_2d_cam


NominalCalibration = Calibration(NominalOriginStandard)
