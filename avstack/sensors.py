# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-09
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-10-22
# @Description:
"""

"""
from copy import copy, deepcopy
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from operator import itemgetter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

import matplotlib.pyplot as plt
from avstack import datastructs, messages
from avstack.geometry import Origin, q_mult_vec
from avstack.calibration import CameraCalibration
from avstack.utils import maskfilters
from avstack import transformations as tforms


class SensorData():
    def __init__(self, timestamp, frame, data, calibration, source_ID, source_name, **kwargs):
        self.timestamp = timestamp
        self.frame = frame
        self.source_ID = source_ID
        self.source_name = source_name
        self.source_identifier = source_name + '-' + str(source_ID)
        self.data = data
        self.calibration = calibration
        allowed_keys = {'depth', 'in_front'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

    @property
    def shape(self):
        return self.data.shape

    @property
    def origin(self):
        return self.calibration.origin

    @property
    def _default_subfolder(self):
        return self.source_identifier

    def __getitem__(self, index):
        s = deepcopy(self)
        s.data = s.data[index]
        return s

    def __matmul__(self, other):
        """Called when doing self @ other
        creates a transformation object that transforms
        """
        raise NotImplementedError

    def change_origin(self, origin_new):
        raise NotImplementedError

    def project(self, calib_other):
        raise NotImplementedError(f'{type(self)} has not implemented project')

    def view(self):
        raise NotImplementedError

    def save_to_folder(self, folder, filename=None, add_subfolder=True, **kwargs):
        if add_subfolder:
            folder = os.path.join(folder, self._default_subfolder)
        os.makedirs(folder, exist_ok=True)
        if filename is None:
            filename = 'timestamp_%08.2f-frame_%06d' % (self.timestamp, self.frame)
        self.calibration.save_to_file(os.path.join(folder, 'calib-'+filename))
        self.save_to_file(os.path.join(folder, 'data-'+filename), **kwargs)

    def save_to_file(self, filename):
        """Each derived class saves data"""
        raise NotImplementedError


class ImuData(SensorData):
    def __init__(self, *args, source_name='imu', **kwargs):
        super().__init__(*args, **kwargs, source_name=source_name)


class GpsData(SensorData):
    def __init__(self, *args, levar, source_name='gps', **kwargs,):
        super().__init__(*args, **kwargs, source_name=source_name)
        self.levar = levar


class ImageData(SensorData):
    def __init__(self, *args, source_name='image', **kwargs):
        super().__init__(*args, **kwargs, source_name=source_name)

    def save_to_file(self, filepath):
        save_image_file(self.data, filepath, self.source_name)

    def view(self, axis=False, extent=None):
        pil_im = Image.fromarray(self.data)
        plt.figure(figsize=[2*x for x in plt.rcParams["figure.figsize"]])
        plt.imshow(pil_im, extent=extent)
        if not axis:
            plt.axis('off')
        plt.show()


class DepthImageData(SensorData):
    def __init__(self, *args, source_name='depthimage', **kwargs):
        super().__init__(*args, **kwargs, source_name=source_name)
        self.depth_in_meters = None

    @property
    def depths(self):
        """Defer this calculation to save cost at runtime"""
        if self.depth_in_meters is None:
            d = np.asarray(self.data).astype(np.float32)
            R, G, B = d[:,:,0], d[:,:,1], d[:,:,2]
            normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
            self.depth_in_meters = 1000 * normalized
        return self.depth_in_meters

    def save_to_file(self, filepath):
        save_image_file(self.data, filepath, self.source_name)

    def view(self, axis=False, extent=None):
        pil_im = Image.fromarray(self.depths)
        plt.figure(figsize=[2*x for x in plt.rcParams["figure.figsize"]])
        plt.imshow(pil_im, extent=extent)
        if not axis:
            plt.axis('off')
        plt.show()


class LidarData(SensorData):
    def __init__(self, *args, source_name='lidar', n_features=4, **kwargs):
        super().__init__(*args, **kwargs, source_name=source_name)
        self.n_features = n_features

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.data[key[0], key[1]]
        elif isinstance(key, int):
            return self.data[key[0],:]
        else:
            raise NotImplementedError(key)

    def filter_by_range(self, min_range, max_range, inplace=True):
        if (min_range is not None) or (max_range is not None):
            min_range = 0 if min_range is None else min_range
            max_range = np.inf if max_range is None else max_range
            mask = maskfilters.filter_points_range(self, min_range, max_range)
            return self.filter(mask, inplace=inplace)
        else:
            if not inplace:
                return deepcopy(self)

    def filter(self, mask, inplace=True):
        if inplace:
            self.data = self.data[mask,:]
        else:
            data = self.data[mask,:]
            return LidarData(self.timestamp, self.frame, data, self.calibration, self.source_ID)

    def project(self, calib_other):
        if isinstance(calib_other, CameraCalibration):
            data = calib_other.project_3d_points(self.data, origin_pts=self.calibration.origin)
            depth = np.linalg.norm(self.data[:,:3], axis=1)
            return ProjectedLidarData(self.timestamp, self.frame, data, calib_other,
                self.source_ID, depth=depth)
        else:
            data = calib_other.project_3d_points(self.data, origin_pts=self.calibration.origin)
            return LidarData(self.timestamp, self.frame, data, calib_other, self.source_ID)

    def transform_to_ground(self):
        x_old = self.origin.x
        x_new = np.array([x_old[0], x_old[1], 0])  # flat to the ground
        eul_old = self.origin.euler
        q_new = tforms.transform_orientation([0, 0, eul_old[2]], 'euler', 'quat')  # only yaw still applied
        O_new = Origin(x_new, q_new)
        self.change_origin(O_new)

    def change_origin(self, origin_new):
        self.data[:,:3] = self.origin.change_points_origin(self.data[:,:3], origin_new)
        self.calibration.change_origin(origin_new)

    def save_to_file(self, filepath, flipy=False, as_ply=False):
        if isinstance(self.data, np.ndarray):
            if as_ply:
                if not filepath.endswith('.ply'):
                    filepath = filepath + '.ply'
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self.data[:,:3])
                o3d.io.write_point_cloud(filepath, pcd)
            else:
                if not filepath.endswith('.bin'):
                    filepath = filepath + '.bin'
                with open(filepath, 'wb') as f:
                    self.data.tofile(f, sep='')
        else:
            try:
                if isinstance(self.data.raw_data, memoryview):
                    if not filepath.endswith('.ply'):
                        filepath = filepath + '.ply'
                    if flipy:
                        data = np.frombuffer(self.data.raw_data, dtype=np.float32).reshape([-1, self.n_features])
                        try:
                            data[:,1] *= -1
                        except ValueError as e:
                            data = np.copy(data)
                            data[:,1] *= -1
                        filepath = filepath.replace('.ply', '.bin')
                        with open(filepath, 'wb') as f:
                            data.tofile(filepath)
                    else:
                        self.data.save_to_disk(filepath)
                    return
            except AttributeError as e:
                pass
            raise NotImplementedError(self.data)

    def as_spherical_matrix(self, rate, sensor):
        """Converts data to spherical matrix"""
        if sensor.lower() == 'kitti':
            sensor = 'hdl-64e'
        elif sensor.lower() == 'nuscenes':
            sensor = 'hdl-32e'
                
        # convert to spherical (and degrees)
        R = np.linalg.norm(self.data[:,:3], axis=1)
        A = 180/np.pi * np.arctan2(self.data[:,1], self.data[:,0])
        A[A<0] += 360
        E = 180/np.pi * np.arcsin(self.data[:,2] / R)
        E[E>90] -= 360        

        # how many discrete elevations do we have?
        elevations = messages.get_velodyne_elevation_table(sensor)

        # get sensor characteristics
        if sensor is None:
            raise NotImplementedError
        elif sensor.lower() == 'vlp-16':
            vendor = 'velodyne'
            firing_time = 55.296e-6  # for all 16 lasers
            packet_height = 32
            packet_width = 12
        elif sensor.lower() == 'hdl-32e':
            vendor = 'velodyne'
            firing_time = 46.080e-6 # for all 32 lasers
            packet_height = 32
            packet_width = 12
        elif sensor.lower() == 'hdl-64e':
            vendor = 'velodyne'
            firing_time = 50.0e-6
            packet_height = 32
            packet_width = 12  # first 6 are "upper", second 6 are "lower"
        elif sensor.lower() in ['hdl-64e-s2', 'hdl-64e-s2.1']:
            vendor = 'velodyne'
            firing_time = 48.0e-6  # 32 of upper synchedwith 32 of lower
            packet_height = 32
            packet_width = 12  # first 6 are "upper", second 6 are "lower"
        else:
            raise NotImplementedError(
                'Do not have a routine for {} sensor'.format(sensor))

        # rate implies azimuth resolution
        if vendor == 'velodyne':
            # RPM = rate * 60
            az_res = rate * 360 * firing_time
            n_azimuths = int( 1.//(rate * firing_time) )
            azimuths = {i:az_res*i for i in range(0, n_azimuths, 1)}
        else:
            raise NotImplementedError(vendor)

        # bin points into discrete values (make monotonic, then unsort)
        idx_az = np.argsort(list(azimuths.values()))  # should already be sorted...
        idx_el = np.argsort(list(elevations.values()))
        idx_az_to_bin_az = {i:idx for i, idx in enumerate(idx_az)}
        idx_el_to_bin_el = {i:idx for i, idx in enumerate(idx_el)}
        A_idxs = np.digitize(A, [azimuths[idx] for idx in idx_az], right=False) - 1
        E_idxs = np.digitize(E, [elevations[idx] for idx in idx_el], right=False) - 1
        A_bins = np.array([int(idx_az_to_bin_az[a_idx]) for a_idx in A_idxs])
        E_bins = np.array([int(idx_el_to_bin_el[e_idx]) for e_idx in E_idxs])

        # now we have indices in the matrix for each point
        # **all combinations of (a_bin, e_bin) should be unique**
        AE_M = np.empty((len(azimuths),len(elevations)))
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
        delta_orders = [[i,j,i**2+j**2] for i in range(-12,13) for j in range(-5,6)]
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

    def as_pseudo_packets(self, rate, sensor=None):
        """Converts lidar data to pseudo-packets in velodyne format
        
        KITTI uses:
            - Velodyne HDL-64E
            - 64 lines
            - 10 Hz capture
            - single return mode

        nuScenes uses:
            - Velodyne 32
            - 32-line lidar
            - vertical FOV from -30 to 10 degrees
            - range of 70m
            - 20 Hz capture rate
            - single return mode

        WaymoOpenDataset uses 
            - range of 75m
            - 10 Hz capture rate
            - vertical FOV from -17.6 to 2.4 degrees
            - dual return mode
        
        Steps:
        1. convert cartesian to spherical
        2. bin along elevation
        3. bin along azimuth
        """

        # get number of things that will be computed
        n_el_per_block = min(len(elevations), packet_height)
        n_az_per_block = packet_height / n_el_per_block
        n_az_per_packet = (packet_height*packet_width) / len(elevations)
        n_packets_per_sweep = int(len(azimuths) / n_az_per_packet)

        # make pseudo packets incrementally
        pseudo_packets = []
        t0 = self.timestamp
        for i in range(n_packets_per_sweep):
            header = messages.VelodyneHeader()
            data_blocks = []
            timestamp = t0 + firing_time*n_az_per_block*packet_width*i
            for j in range(0, len(elevations), n_el_per_block):
                el_this = np.array(list(range(j*n_el_per_block, (j+1)*n_el_per_block)))
                for k in range(0, n_az_per_packet):
                    az_this = n_az_per_packet + k*n_az_per_block  # a single entry
                    data = np.reshape(AE_M[az_this, el_this], -1)
                    assert len(data) == packet_height, 'Number of data elements must be packet height'
                    data_blocks.append(messages.VelodyneDataBlock(
                        azimuths[az_this], data))
            factory = messages.VelodyneFactory('StrongestReturn', sensor)
            pseudo_packets.append(messages.VelodynePseudoPacket(
                header, data_blocks, timestamp, factory))
        return pseudo_packets

    def as_packets(self, sensor=''):
        raise NotImplementedError


class ProjectedLidarData(SensorData):
    def __init__(self, *args, source_name='projected-lidar', **kwargs):
        super().__init__(*args, **kwargs, source_name=source_name)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.data[key[0], key[1]]
        elif isinstance(key, int):
            return self.data[key[0],:]
        else:
            raise NotImplementedError(key)


class RadarDataRazelRRT(SensorData):
    def __init__(self, *args, source_name='radar', **kwargs):
        super().__init__(*args, **kwargs, source_name=source_name)


def save_image_file(data, filepath, source_name, ext='.png'):
    ext_check = ['.png', '.jpg', '.jpeg', '.tiff', '.tif']
    for ext_c in ext_check:
        if filepath.endswith(ext_c):
            filepath = filepath.replace(ext_c, '')
            break
    filepath = filepath + ext
    if isinstance(data, np.ndarray):
        cv2.imwrite(filepath, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
    else:
        try:
            if isinstance(data.raw_data, memoryview):
                if source_name == 'depthimage':
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

class DataBuffer(datastructs.PriorityQueue):
    TYPE = 'DataBuffer'


class ImuBuffer(DataBuffer):
    TYPE = 'ImuBuffer'
    def __init__(self):
        super().__init__(max_size=None, max_heap=False)  # pop "earliest" first

    def integrate_up_to(self, t_up_to):
        imu_elements = self.pop_all_below(t_up_to)
        dt_total = 0.0
        dv_total = np.zeros((3,))
        dth_total = np.zeros((3,))
        frame = None
        source_ID = None
        imu_calib = None
        R = np.zeros((6,6))
        for timestamp, imu_data in imu_elements:
            dt_total += imu_data.data['dt']
            dv_total += imu_data.data['dv']
            dth_total += imu_data.data['dth']
            R = imu_data.data['R']
            source_ID = imu_data.source_ID
            imu_calib = imu_data.calibration
            frame = imu_data.frame
        return ImuData(t_up_to, frame, {'dt':dt_total, 'dv':dv_total, 'dth':dth_total, 'R':R},
            imu_calib, source_ID)
