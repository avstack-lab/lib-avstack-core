# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-08-07
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-29
# @Description:
"""

"""
from __future__ import annotations

import numpy as np

from avstack.geometry.refchoc import get_reference_from_line


def read_calibration_from_line(line):
    elems = line.split(" ")
    assert elems[0] == "calibration", elems[0]
    if elems[1] == "intrinsics":
        P_cam = np.reshape(np.array([float(e) for e in elems[2:14]]), (3, 4))
        assert elems[14] == "img_shape"
        img_shape = tuple([int(e) for e in elems[15:17]])
        try:
            channel_order = elems[18]
        except IndexError:
            channel_order = "bgr"
        reference = get_reference_from_line(" ".join(elems[19:]))
        return CameraCalibration(
            reference, P_cam, img_shape, channel_order=channel_order
        )
    else:
        reference = get_reference_from_line(" ".join(elems[1:]))
        return Calibration(reference)


class Calibration:
    def __init__(self, reference):
        """
        T - extrinsic matrix (rotation + translation)

        extrinsic matrix is transformation from global to local
        """
        self.reference = reference

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Calibration with reference: {self.reference}"

    def format_as_string(self):
        return "calibration " + self.reference.format_as_string()

    def save_to_file(self, filename):
        if not filename.endswith(".txt"):
            filename = filename + ".txt"
        with open(filename, "w") as f:
            f.write(self.format_as_string())


class CameraCalibration(Calibration):
    def __init__(
        self,
        reference,
        P: np.ndarray,
        img_shape: tuple,
        fov_horizontal=None,
        fov_vertical=None,
        square_pixels=False,
        channel_order="bgr",
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
        self.channel_order = channel_order
        super().__init__(reference)

    def __str__(self):
        return f"Camera Calibration with reference: {self.reference}; P:{self.P}"

    def format_as_string(self):
        P_as_str = " ".join([str(v) for v in np.ravel(self.P)])
        c_str = (
            "calibration "
            + "intrinsics "
            + P_as_str
            + f" img_shape {self.img_shape[0]} {self.img_shape[1]}"
            + f" channel_order {self.channel_order} "
            + self.reference.format_as_string()
        )
        return c_str
