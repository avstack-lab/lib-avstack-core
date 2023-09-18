# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-09
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-10-22
# @Description:
"""
Custom sensor data structures to standardize interfaces to data
and provide maximal forward, backward compatibility.
"""
from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from avstack import datastructs, maskfilters, messages
from avstack.calibration import CameraCalibration, LidarCalibration
from avstack.geometry import GlobalOrigin3D, PointMatrix3D, ReferenceFrame
from avstack.geometry import transformations as tforms


class SensorData:
    """Base class for sensor data structure"""

    def __init__(
        self, timestamp, frame, data, calibration, source_ID, source_name, **kwargs
    ):
        self.timestamp = float(timestamp)
        self.frame = int(frame)
        self.source_ID = str(source_ID)
        self.source_name = str(source_name)
        self.source_identifier = self.source_name + "-" + self.source_ID
        self.data = data
        self.calibration = calibration
        allowed_keys = {"depth", "in_front"}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

    @property
    def shape(self):
        """list: Returns the shape of the data."""
        return self.data.shape

    @property
    def reference(self):
        """avstack.calibration.reference: Returns origin of the sensor when data captured."""
        return self.calibration.reference

    @property
    def _default_subfolder(self):
        return self.source_identifier

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index):
        """Indexes into sensor data list/array"""
        s2 = self.duplicate()
        s2.data = self.data[index]
        return s2

    def __matmul__(self, other):
        """Called when doing self @ other
        creates a transformation object that transforms
        """
        raise NotImplementedError

    def duplicate(self):
        return self.__class__(
            timestamp=self.timestamp,
            frame=self.frame,
            data=self.data,
            calibration=self.calibration,
            source_ID=self.source_ID,
            source_name=self.source_name,
        )

    def view(self):
        raise NotImplementedError

    def save_to_folder(self, folder, filename=None, add_subfolder=True, **kwargs):
        if add_subfolder:
            folder = os.path.join(folder, self._default_subfolder)
        os.makedirs(folder, exist_ok=True)
        if filename is None:
            filename = "timestamp_%08.2f-frame_%06d" % (self.timestamp, self.frame)
        self.calibration.save_to_file(os.path.join(folder, "calib-" + filename))
        self.save_to_file(os.path.join(folder, "data-" + filename), **kwargs)

    def save_to_file(self, filename):
        """Each derived class saves data"""
        raise NotImplementedError


class ImuData(SensorData):
    """IMU datastructure

    Attributes:
        timestamp (float):
            The time of the sensor data capture
        frame (int):
            The discrete frame when data were captured
        source_ID (int):
            Unique identifier for this sensor
        source_name (str):
            Name characterizing the sensor
        source_identifier (str):
            Concatenation of the source name and ID
        data (dict):
            Dictionary containing: dt, dv, dth, R
        calibration (str):
            A calibration class describing the sensor's state
    """

    def __init__(self, *args, source_name="imu", **kwargs):
        super().__init__(*args, **kwargs, source_name=source_name)


class GpsData(SensorData):
    """GPS datastructure

    Attributes:
        timestamp (float):
            The time of the sensor data capture
        frame (int):
            The discrete frame when data were captured
        source_ID (int):
            Unique identifier for this sensor
        source_name (str):
            Name characterizing the sensor
        source_identifier (str):
            Concatenation of the source name and ID
        data (np.ndarray):
            Data for this sensor
        calibration (str):
            A calibration class describing the sensor's state
    """

    def __init__(
        self,
        *args,
        levar,
        source_name="gps",
        **kwargs,
    ):
        super().__init__(*args, **kwargs, source_name=source_name)
        self.levar = levar


class ImageData(SensorData):
    """Image datastructure

    Attributes:
        timestamp (float):
            The time of the sensor data capture
        frame (int):
            The discrete frame when data were captured
        source_ID (int):
            Unique identifier for this sensor
        source_name (str):
            Name characterizing the sensor
        data (np.ndarray):
            Data for this sensor. Monocular would be [N x M x 1], color [N x M x 3]
        calibration (str):
            A calibration class describing the sensor's state
    """

    def __init__(self, *args, source_name="image", **kwargs):
        super().__init__(*args, **kwargs, source_name=source_name)

    def save_to_file(self, filepath):
        save_image_file(self.rgb_image, filepath, self.source_name)

    @property
    def rgb_image(self):
        return (
            self.data
            if self.calibration.channel_order == "rgb"
            else self.data[:, :, ::-1]
        )

    def view(self, axis=False, extent=None, objects=None, view="box2d"):
        pil_im = Image.fromarray(self.rgb_image)
        plt.figure(figsize=[2 * x for x in plt.rcParams["figure.figsize"]])
        plt.imshow(pil_im, extent=extent)
        if not axis:
            plt.axis("off")
        plt.show()


class SemanticSegmentationImageData(ImageData):
    def __init__(self, *args, source_name="semsegimage", **kwargs):
        super().__init__(*args, source_name=source_name, **kwargs)
        self._rgb_image = None

    def save_to_file(self, filepath):
        save_image_file(self.data, filepath, self.source_name)

    @property
    def rgb_image(self):
        """Defer this calculation to save cost at runtime"""
        if self._rgb_image is None:
            # each tag gets its own color
            colors = self.calibration.colors
            self._rgb_image = np.zeros(
                (self.data.shape[0], self.data.shape[1], 3), dtype=np.uint8
            )
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    self._rgb_image[i, j, :] = colors[self.data[i, j, 2]]
        return self._rgb_image

    @rgb_image.setter
    def rgb_image(self, image):
        self._rgb_image = image


class DepthImageData(SensorData):
    """Depth image datastructure

    Attributes:
        timestamp (float):
            The time of the sensor data capture
        frame (int):
            The discrete frame when data were captured
        source_ID (int):
            Unique identifier for this sensor
        source_name (str):
            Name characterizing the sensor
        source_identifier (str):
            Concatenation of the source name and ID
        data (np.ndarray):
            Depth image in format of [N x M x 3] in a coding
        calibration (str):
            A calibration class describing the sensor's state
    """

    def __init__(self, *args, encoding="carla", source_name="depthimage", **kwargs):
        super().__init__(*args, **kwargs, source_name=source_name)
        self.depth_in_meters = None
        self.encoding = encoding

    @property
    def grayscale_image(self):
        return self.depths

    @property
    def depths(self):
        """Defer this calculation to save cost at runtime"""
        if self.depth_in_meters is None:
            if self.encoding == "carla":
                d = np.asarray(self.data).astype(np.float32)
                R, G, B = d[:, :, 2], d[:, :, 1], d[:, :, 0]
                normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
                self.depth_in_meters = 1000 * normalized
            elif self.encoding == "midas":
                bits = 1
                depth_min = self.data.min()
                depth_max = self.data.max()
                max_val = (2**(8*bits))-1
                self.depth_in_meters = max_val * (self.data - depth_min) / (depth_max - depth_min)
                self.depth_in_meters = self.depth_in_meters.astype("uint8" if bits == 1 else "uint16")
            else:
                raise NotImplementedError
        return self.depth_in_meters

    def save_to_file(self, filepath):
        save_image_file(self.data, filepath, self.source_name)

    def view(self, axis=False, extent=None):
        """View image using matplotlib"""
        pil_im = Image.fromarray(self.depths)
        plt.figure(figsize=[2 * x for x in plt.rcParams["figure.figsize"]])
        plt.imshow(pil_im, extent=extent, cmap='gray')
        if not axis:
            plt.axis("off")
        plt.show()


class LidarData(SensorData):
    """LiDAR point cloud datastructure

    Attributes:
        timestamp (float):
            The time of the sensor data capture
        frame (int):
            The discrete frame when data were captured
        source_ID (int):
            Unique identifier for this sensor
        source_name (str):
            Name characterizing the sensor
        source_identifier (str):
            Concatenation of the source name and ID
        data (np.ndarray):
            Point cloud matrix of size [# points, # features]
        calibration (str):
            A calibration class describing the sensor's state
        n_features (int):
            Number of features for the LiDAR data (e.g., (X, Y, Z, Int))
    """

    def __init__(self, *args, source_name="lidar", n_features=4, **kwargs):
        super().__init__(*args, **kwargs, source_name=source_name)
        self.n_features = n_features

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, key: tuple | int):
        if isinstance(key, tuple):
            return self.data[key[0], key[1]]
        elif isinstance(key, int):
            return self.data[key, :]
        else:
            raise NotImplementedError(key)

    def filter_by_range(self, min_range: float, max_range: float, inplace=True):
        if (min_range is not None) or (max_range is not None):
            min_range = 0 if min_range is None else min_range
            max_range = np.inf if max_range is None else max_range
            mask = maskfilters.filter_points_range(self, min_range, max_range)
            return self.filter(mask, inplace=inplace)
        else:
            if not inplace:
                return self.duplicate()

    def filter(self, mask, inplace: bool = True):
        if inplace:
            self.data = self.data.filter(mask)
        else:
            data = self.data.filter(mask)
            return LidarData(
                self.timestamp, self.frame, data, self.calibration, self.source_ID
            )

    def project(self, calib_other):
        if isinstance(calib_other, CameraCalibration):
            depth = np.linalg.norm(self.data[:, :3], axis=1)
            data = self.data.project_to_2d(calib_other)
            return ProjectedLidarData(
                self.timestamp,
                self.frame,
                data,
                calib_other,
                self.source_ID,
                depth=depth,
            )
        else:
            if not isinstance(calib_other, LidarCalibration):
                raise NotImplementedError(type(calib_other))
            data = self.data.change_calibration(calib_other, inplace=False)
            return LidarData(
                self.timestamp, self.frame, data, calib_other, self.source_ID
            )

    def transform_to_ground(self):
        """Only possible with reference's reference to global origin for now"""
        if self.reference.reference != GlobalOrigin3D:
            raise NotImplementedError(
                "Cannot yet project if calibration not level-1 only"
            )
        x_new = np.array(
            [self.reference.x[0], self.reference.x[1], 0]
        )  # flat on ground
        yaw_old = tforms.transform_orientation(self.reference.q, "quat", "euler")[2]
        q_new = tforms.transform_orientation(
            [0, 0, yaw_old], "euler", "quat"
        )  # keep old yaw
        ref_new = ReferenceFrame(x_new, q_new, reference=self.reference.reference)
        calib_new = LidarCalibration(reference=ref_new)
        return self.project(calib_new)

    def save_to_file(self, filepath: str, flipy: bool = False, as_ply: bool = False):
        if isinstance(self.data, (PointMatrix3D, np.ndarray)):
            data = self.data if isinstance(self.data, np.ndarray) else self.data.x
            data = data.astype(np.float32)
            if as_ply:
                if not filepath.endswith(".ply"):
                    filepath = filepath + ".ply"
                import open3d as o3d

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data[:, :3])
                o3d.io.write_point_cloud(filepath, pcd)
            else:
                if not filepath.endswith(".bin"):
                    filepath = filepath + ".bin"
                with open(filepath, "wb") as f:
                    data.tofile(f, sep="")
        else:
            try:
                if isinstance(self.data.raw_data, memoryview):
                    if as_ply:
                        raise NotImplementedError
                    #     if filepath.endswith(".bin"):
                    #         filepath = filepath.replace(".bin", ".ply")
                    #     elif not filepath.endswith(".ply"):
                    #         filepath = filepath + ".ply"
                    # filepath = filepath.replace(".ply", ".bin")
                    if not filepath.endswith(".bin"):
                        filepath += ".bin"
                    data = np.frombuffer(self.data.raw_data, dtype=np.float32).reshape(
                        [-1, self.n_features]
                    )
                    if flipy:
                        try:
                            data[:, 1] *= -1
                        except ValueError as e:
                            data = np.copy(data)
                            data[:, 1] *= -1
                    with open(filepath, "wb") as f:
                        data.tofile(filepath)
                    # else:
                    # self.data.save_to_disk(filepath) # later we'll figure out this way
                    return
            except AttributeError as e:
                pass
            raise NotImplementedError(self.data)

    def as_spherical_matrix(self, rate: float, sensor: str):
        """Converts cartesian point cloud data to spherical matrix"""
        if sensor.lower() == "kitti":
            sensor = "hdl-64e"
        elif sensor.lower() == "nuscenes":
            sensor = "hdl-32e"

        # convert to spherical (and degrees)
        R = np.linalg.norm(self.data[:, :3], axis=1)
        A = 180 / np.pi * np.arctan2(self.data[:, 1], self.data[:, 0])
        A[A < 0] += 360
        E = 180 / np.pi * np.arcsin(self.data[:, 2] / R)
        E[E > 90] -= 360

        # how many discrete elevations do we have?
        elevations = messages.get_velodyne_elevation_table(sensor)

        # get sensor characteristics
        if sensor is None:
            raise NotImplementedError
        elif sensor.lower() == "vlp-16":
            vendor = "velodyne"
            firing_time = 55.296e-6  # for all 16 lasers
            packet_height = 32
            packet_width = 12
        elif sensor.lower() == "hdl-32e":
            vendor = "velodyne"
            firing_time = 46.080e-6  # for all 32 lasers
            packet_height = 32
            packet_width = 12
        elif sensor.lower() == "hdl-64e":
            vendor = "velodyne"
            firing_time = 50.0e-6
            packet_height = 32
            packet_width = 12  # first 6 are "upper", second 6 are "lower"
        elif sensor.lower() in ["hdl-64e-s2", "hdl-64e-s2.1"]:
            vendor = "velodyne"
            firing_time = 48.0e-6  # 32 of upper synchedwith 32 of lower
            packet_height = 32
            packet_width = 12  # first 6 are "upper", second 6 are "lower"
        else:
            raise NotImplementedError(
                "Do not have a routine for {} sensor".format(sensor)
            )

        # rate implies azimuth resolution
        if vendor == "velodyne":
            # RPM = rate * 60
            az_res = rate * 360 * firing_time
            n_azimuths = int(1.0 // (rate * firing_time))
            azimuths = {i: az_res * i for i in range(0, n_azimuths, 1)}
        else:
            raise NotImplementedError(vendor)

        # bin points into discrete values (make monotonic, then unsort)
        idx_az = np.argsort(list(azimuths.values()))  # should already be sorted...
        idx_el = np.argsort(list(elevations.values()))
        idx_az_to_bin_az = {i: idx for i, idx in enumerate(idx_az)}
        idx_el_to_bin_el = {i: idx for i, idx in enumerate(idx_el)}
        A_idxs = np.digitize(A, [azimuths[idx] for idx in idx_az], right=False) - 1
        E_idxs = np.digitize(E, [elevations[idx] for idx in idx_el], right=False) - 1
        A_bins = np.array([int(idx_az_to_bin_az[a_idx]) for a_idx in A_idxs])
        E_bins = np.array([int(idx_el_to_bin_el[e_idx]) for e_idx in E_idxs])

        # now we have indices in the matrix for each point
        # **all combinations of (a_bin, e_bin) should be unique**
        AE_M = np.empty((len(azimuths), len(elevations)))
        AE_M[:] = np.nan
        n_conflicts = 0
        conflict_coords = []

        # NOTE: this is slow...make faster later
        for rng, a_bin, e_bin in zip(R, A_bins, E_bins):
            if np.isnan(AE_M[a_bin, e_bin]):
                AE_M[a_bin, e_bin] = rng
            else:
                conflict_coords.append([a_bin, e_bin, rng])
                n_conflicts += 1

        # Assign conflicts one-by-one greedily
        delta_orders = [
            [i, j, i**2 + j**2] for i in range(-12, 13) for j in range(-5, 6)
        ]
        delta_orders.sort(key=lambda x: x[2])
        still_conflict = []
        # miss_coords = [[a_m, e_m] for a_m, e_m in zip(*np.where(np.isnan(AE_M)))]
        for i, conflict in enumerate(conflict_coords):
            for d_az, d_el, cost in delta_orders:
                i_az = (conflict[0] + d_az) % AE_M.shape[0]  # wrap azimuth angles
                j_el = conflict[1] + d_el
                if (j_el < 0) or (j_el >= AE_M.shape[1]):  # don't wrap elevation angles
                    continue
                if np.isnan(AE_M[i_az, j_el]):
                    AE_M[i_az, j_el] = conflict[2]
                    break
            else:
                still_conflict.append(conflict)
        return AE_M, azimuths, elevations


class ProjectedLidarData(SensorData):
    """LiDAR point cloud projected into a 2D view

    Attributes:
        timestamp (float):
            The time of the sensor data capture
        frame (int):
            The discrete frame when data were captured
        source_ID (int):
            Unique identifier for this sensor
        source_name (str):
            Name characterizing the sensor
        source_identifier (str):
            Concatenation of the source name and ID
        data (np.ndarray):
            Point cloud matrix of size [# points, # features]
        calibration (str):
            A calibration class describing the sensor's state
        n_features (int):
            Number of features for the LiDAR data (e.g., (X, Y, Z, Int))
    """

    def __init__(self, *args, source_name="projected-lidar", **kwargs):
        super().__init__(*args, **kwargs, source_name=source_name)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.data[key[0], key[1]]
        elif isinstance(key, int):
            return self.data[key[0], :]
        else:
            raise NotImplementedError(key)


class _RadarData(SensorData):
    """Classic RADAR datastructure

    Attributes:
        timestamp (float):
            The time of the sensor data capture
        frame (int):
            The discrete frame when data were captured
        source_ID (int):
            Unique identifier for this sensor
        source_name (str):
            Name characterizing the sensor
        source_identifier (str):
            Concatenation of the source name and ID
        data (np.ndarray):
            TODO
        calibration (str):
            A calibration class describing the sensor's state
        n_features (int):
            Number of features for the LiDAR data (e.g., (X, Y, Z, Int))
    """

    def __init__(self, *args, source_name="radar", **kwargs):
        super().__init__(*args, **kwargs, source_name=source_name)

    def save_to_file(self, filepath: str):
        """Saves radar detection information in a numpy array"""
        if isinstance(self.data, np.ndarray):
            if not filepath.endswith(".txt"):
                filepath = filepath + ".txt"
            np.savetxt(filepath, self.data, fmt="%.18e", delimiter=", ")
        else:
            if not filepath.endswith(".bin"):
                filepath = filepath + ".bin"
            data = np.frombuffer(self.data.raw_data, dtype=np.float32).reshape([-1, 4])
            data = data[:, [3, 1, 2, 0]]
            with open(filepath, "wb") as f:
                data.tofile(filepath)

    def filter(self, mask, inplace: bool = True):
        if inplace:
            self.data = self.data.filter(mask)
        else:
            data = self.data.filter(mask)
            return self.__class__(
                self.timestamp, self.frame, data, self.calibration, self.source_ID
            )


class ProjectedRadarData(ProjectedLidarData):
    pass


class RadarDataRazelRRT(_RadarData):
    def convert_to_cartesian(self):
        data = self.data.copy()
        data.x[:, :3] = tforms.matrix_spherical_to_cartesian(data.x[:, :3])
        return RadarDataXYZRRT(
            self.timestamp, self.frame, data, self.calibration, self.source_ID
        )

    def project(self, calib_other):
        return self.convert_to_cartesian().project(calib_other)

    def filter_by_range(self, min_range: float, max_range: float, inplace=True):
        raise NotImplementedError
        if (min_range is not None) or (max_range is not None):
            min_range = 0 if min_range is None else min_range
            max_range = np.inf if max_range is None else max_range
            mask = (self.data > min_range) & (self.data < max_range)
            return self.filter(mask, inplace=inplace)
        else:
            if not inplace:
                return self.duplicate()


class RadarDataXYZRRT(_RadarData):
    def convert_to_spherical(self):
        data = self.data.copy()
        data.x[:, :3] = tforms.matrix_cartesian_to_spherical(data.x[:, :3])
        return RadarDataRazelRRT(
            self.timestamp, self.frame, data, self.calibration, self.source_ID
        )

    def project(self, calib_other):
        if isinstance(calib_other, CameraCalibration):
            depth = np.linalg.norm(self.data[:, :3], axis=1)
            data = self.data.project_to_2d(calib_other)
            return ProjectedRadarData(
                self.timestamp,
                self.frame,
                data,
                calib_other,
                self.source_ID,
                depth=depth,
            )
        else:
            data = self.data.change_calibration(calib_other)
            return RadarDataXYZRRT(
                self.timestamp, self.frame, data, calib_other, self.source_ID
            )

    def filter_by_range(self, min_range: float, max_range: float, inplace=True):
        raise NotImplementedError
        if (min_range is not None) or (max_range is not None):
            min_range = 0 if min_range is None else min_range
            max_range = np.inf if max_range is None else max_range
            mask = maskfilters.filter_points_range(self, min_range, max_range)
            return self.filter(mask, inplace=inplace)
        else:
            if not inplace:
                return self.duplicate()


def save_image_file(
    data: np.ndarray, filepath: str, is_depth: bool = False, ext: str = ".png"
):
    """Saves image to a file with opencv

    Args:
        data (np.ndarray): Image matrix
        filepath (str): File path to save image
        is_depth (bool): True if image is depth image, false otherwise
        ext (str): Image format extension
    """
    ext_check = [".png", ".jpg", ".jpeg", ".tiff", ".tif"]
    for ext_c in ext_check:
        if filepath.endswith(ext_c):
            filepath = filepath.replace(ext_c, "")
            break
    filepath = filepath + ext
    if isinstance(data, np.ndarray):
        cv2.imwrite(filepath, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
    else:
        try:
            if isinstance(data.raw_data, memoryview):
                if is_depth:
                    data.save_to_disk(filepath)  # possibly convert to carla.Depth (?)
                else:
                    data.save_to_disk(filepath)
                return
        except AttributeError as e:
            pass
        raise NotImplementedError(data)


# ==============================================
# DATA BUFFERS
# ==============================================

DataBuffer = datastructs.PriorityQueue


class ImuBuffer(datastructs.PriorityQueue):
    """IMU data buffer

    Can perform accurate integration of IMU data up to some time.
    Instantiated as a min-heap priority queue
    """

    TYPE = "ImuBuffer"

    def __init__(self):
        super().__init__(max_size=None, max_heap=False)  # pop "earliest" first

    def integrate_up_to(self, t_up_to: float):
        """Integrate the IMU data in the buffer

        Args:
            t_up_to: time to integrate up to (exclusive)

        Returns:
            ImuData object with the integrated data.
        """
        imu_elements = self.pop_all_below(t_up_to)
        dt_total = 0.0
        dv_total = np.zeros((3,))
        dth_total = np.zeros((3,))
        frame = None
        source_ID = None
        imu_calib = None
        R = np.zeros((6, 6))
        for imu_data in imu_elements:
            dt_total += imu_data.data["dt"]
            dv_total += imu_data.data["dv"]
            dth_total += imu_data.data["dth"]
            R = imu_data.data["R"]
            source_ID = imu_data.source_ID
            imu_calib = imu_data.calibration
            frame = imu_data.frame
        return ImuData(
            t_up_to,
            frame,
            {"dt": dt_total, "dv": dv_total, "dth": dth_total, "R": R},
            imu_calib,
            source_ID,
        )
