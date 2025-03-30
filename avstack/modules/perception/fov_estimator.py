from typing import TYPE_CHECKING, Any, Tuple, Union


if TYPE_CHECKING:
    from avstack.geometry import ReferenceFrame
    from avstack.sensors import LidarData

import numpy as np
from concave_hull import concave_hull
from scipy.interpolate import make_smoothing_spline

from avstack.config import MODELS
from avstack.geometry import GlobalOrigin3D, Polygon
from avstack.modules import BaseModule
from avstack.sensors import ProjectedLidarData
from avstack.utils.decorators import apply_hooks
from collections import defaultdict



class _LidarFovEstimator(BaseModule):
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def _execute_on_array(
        self, data: np.ndarray, reference: "ReferenceFrame"
    ) -> "Polygon":
        raise NotImplementedError

    def _execute_on_bev_array(
        self, data: np.ndarray, reference: "ReferenceFrame"
    ) -> "Polygon":
        raise NotImplementedError


@MODELS.register_module()
class ConcaveHullLidarFOVEstimator(_LidarFovEstimator):
    def __init__(
        self,
        concavity: int = 1,
        length_threshold: float = 4,
        max_height: float = np.inf,
        name: str = "fov_estimator",
        *args,
        **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)
        self.concavity = concavity
        self.length_threshold = length_threshold
        self.max_height = max_height

    @apply_hooks
    def __call__(
        self, pc: "LidarData", in_global: bool = False, *args: Any, **kwds: Any
    ) -> "Polygon":
        fov = pc.concave_hull_bev(
            concavity=self.concavity,
            length_threshold=self.length_threshold,
            in_global=in_global,
            max_height=self.max_height,
        )
        return fov

    def call_on_array(self, pc: np.ndarray):
        boundary = concave_hull(
            pc[:, :2], concavity=self.concavity, length_threshold=self.length_threshold
        )
        return boundary


class _RayTraceFovEstimator(_LidarFovEstimator):
    def __init__(
        self,
        z_min: float = -3.0,
        z_max: float = 3.0,
        n_azimuth_bins: int = 100,
        n_range_bins: int = 1000,
        range_max: float = 100,
        az_tol: float = 0.05,
        smoothing: float = None,
        name: str = "fov_estimator",
        *args,
        **kwargs
    ):
        """Uses ray tracing to estimate a field of view from lidar data

        Args:
            z_min: minimum z value in ground frame for bev filtering
            z_min: maximum z value in ground frame for bev filtering
            az_tolerance: azimuth tolerance to consider for max range
            smoothing: a parameter that controls smooth vs. fit tradeoff
                larger = smoother, smaller = more fit to data

        Projects the data into the ground plane then filters into the birds
        eye view based on vertical axis thresholds.

        Builds a semi-smooth non-parametric model of range = f(azimuth)
        with B-splines.
        """
        super().__init__(name=name, *args, **kwargs)
        self.z_min = z_min
        self.z_max = z_max
        self.n_azimuth_bins = n_azimuth_bins
        self.n_range_bins = n_range_bins
        self.range_max = range_max
        self.az_tol = az_tol
        self.smoothing = smoothing

    @apply_hooks
    def __call__(
        self,
        pc: Union[ProjectedLidarData, "LidarData"],
        in_global: bool = False,
        centering: bool = True,
        *args: Any,
        **kwds: Any
    ) -> "Polygon":
        """Common call method for ray trace fov estimators"""
        # project into BEV
        if not isinstance(pc, ProjectedLidarData):
            pc_bev = pc.project_to_2d_bev(
                z_min=self.z_min,
                z_max=self.z_max,
            )
        else:
            pc_bev = pc

        # get the boundary
        boundary = self._estimate_fov_from_cartesian_lidar(
            pc_bev=pc_bev.data.x[:, :2],
            n_range_bins=self.n_range_bins,
            n_azimuth_bins=self.n_azimuth_bins,
            range_max=self.range_max,
            centering=centering,
        )
        
        self._eliminate_isolated_pts(pc_bev, 10, 30)

        # transform to global if desired
        fov = Polygon(
            boundary=boundary,
            reference=pc_bev.reference,
            frame=pc.frame,
            timestamp=pc.timestamp,
        )
        if in_global:
            fov.change_reference(reference=GlobalOrigin3D, inplace=True)

        return fov

    def call_on_array(self, pc: np.ndarray):
        boundary = self._estimate_fov_from_cartesian_lidar(
            pc_bev=pc,
            n_range_bins=self.n_range_bins,
            n_azimuth_bins=self.n_azimuth_bins,
            range_max=self.range_max,
            centering=True,
        )
        return boundary

    @staticmethod
    def n_bins_to_edges(
        pc_bev_azimuth: np.ndarray,
        pc_bev_range: np.ndarray,
        n_range_bins: int,
        n_azimuth_bins: int,
        range_max: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        min_az = min(pc_bev_azimuth)
        max_az = max(pc_bev_azimuth)
        range_upper = min(range_max, max(pc_bev_range))
        rng_edges = np.linspace(0, range_upper, num=n_range_bins)
        az_edges = np.linspace(min_az, max_az, num=n_azimuth_bins)
        return rng_edges, az_edges

    def _estimate_fov_from_cartesian_lidar(
        self,
        pc_bev: np.ndarray[float],
        n_range_bins: int,
        n_azimuth_bins: int,
        range_max: float,
        centering: bool,
    ) -> np.ndarray:
        """Wrapper to feed the polar estimator"""

        # center the lidar data
        if centering:
            centroid = np.mean(pc_bev[:, :2], axis=0)
            pc_bev[:, :2] -= centroid

        # convert to polar coordinates
        pc_bev_azimuth = np.arctan2(pc_bev[:, 1], pc_bev[:, 0])
        pc_bev_range = np.linalg.norm(pc_bev[:, :2], axis=1)

        # estimate the boundary
        boundary = self._estimate_fov_from_polar_lidar(
            pc_bev_range=pc_bev_range,
            pc_bev_azimuth=pc_bev_azimuth,
            n_range_bins=n_range_bins,
            n_azimuth_bins=n_azimuth_bins,
            range_max=range_max,
        )

        # uncenter the boundary
        if centering:
            pc_bev[:, :2] += centroid
            boundary += centroid

        return boundary

    def _estimate_fov_from_polar_lidar(
        self,
        pc_bev_range: np.ndarray[float],
        pc_bev_azimuth: np.ndarray[float],
        n_range_bins: int,
        n_azimuth_bins: int,
        range_max: float,
    ) -> np.ndarray:
        """To be implemented in subclass"""
        raise NotImplementedError
    
    def _eliminate_isolated_pts(self, pc_bev, m_away, num_pts):
        ptMap = defaultdict(int)
        usable_pts = []
        for p1 in pc_bev.data.x:
            p1x, p1y = p1[0], p1[1]
            for p2 in pc_bev.data.x:
                p2x, p2y = p2[0], p2[1]
                if p1x == p2x and p1y == p2y: 
                    continue
                dis = np.linalg.norm([p1x - p2x, p1y - p2y])    
                if (dis < m_away):
                    ptMap[(p1x, p1y)] += 1
                if (ptMap[(p1x, p1y)] == num_pts):
                    usable_pts.append([p1x, p1y])
                    break
        usable_pts = np.array(usable_pts)
        pc_bev.data.x = usable_pts 


@MODELS.register_module()
class FastRayTraceBevLidarFovEstimator(_RayTraceFovEstimator):
    @staticmethod
    def _estimate_fov_from_polar_lidar(
        pc_bev_azimuth: np.ndarray[float],
        pc_bev_range: np.ndarray[float],
        n_range_bins: int,
        n_azimuth_bins: int,
        range_max: float,
    ) -> np.ndarray:

        # map bins to edges
        rng_edges, az_edges = _RayTraceFovEstimator.n_bins_to_edges(
            pc_bev_azimuth=pc_bev_azimuth,
            pc_bev_range=pc_bev_range,
            n_range_bins=n_range_bins,
            n_azimuth_bins=n_azimuth_bins,
            range_max=range_max,
        )

        # populate a grid
        col_indices = np.array([list(range(len(rng_edges) - 1))] * (len(az_edges) - 1))
        values, _, _ = np.histogram2d(
            x=pc_bev_azimuth,
            y=pc_bev_range,
            bins=[az_edges, rng_edges],
        )
        v_bool = values > 0
        idx_max = np.max(col_indices * v_bool, axis=1)
        az_out = az_edges[:-1]
        rng_out = rng_edges[idx_max]

        # Construct the fov model
        x_out = rng_out * np.cos(az_out)
        y_out = rng_out * np.sin(az_out)
        boundary = np.concatenate((x_out[:, None], y_out[:, None]), axis=1)

        return boundary


@MODELS.register_module()
class SlowRayTraceBevLidarFovEstimator(_RayTraceFovEstimator):
    @staticmethod
    def _estimate_fov_from_polar_lidar(
        pc_bev_azimuth: np.ndarray[float],
        pc_bev_range: np.ndarray[float],
        n_range_bins: int,
        n_azimuth_bins: int,
        range_max: float,
        az_tolerance: float = 0.05,
        smoothing: Union[None, float] = None,
    ) -> np.ndarray:

        # NOTE: arctan2 output is on [-pi, pi] so ensure az queries are the same
        # make edges for the bins
        rng_edges, az_edges = _RayTraceFovEstimator.n_bins_to_edges(
            pc_bev_azimuth=pc_bev_azimuth,
            pc_bev_range=pc_bev_range,
            range_max=range_max,
            n_range_bins=n_range_bins,
            n_azimuth_bins=n_azimuth_bins,
        )
        az_out = []
        rng_out = []
        for az_q in az_edges:
            # NOTE: this currently doesn't handle the wrap around condition
            # look for azimuths within the tolerance
            idx_az_valid = (pc_bev_azimuth >= az_q - az_tolerance) & (
                pc_bev_azimuth <= az_q + az_tolerance
            )
            if sum(idx_az_valid) == 0:
                continue
            # take the max range over these azimuths
            rng_out.append(max(pc_bev_range[idx_az_valid]))
            az_out.append(az_q)
        az_out = np.asarray(az_out)
        rng_out = np.asarray(rng_out)

        # Construct a polygon model
        # add small amount of noise to prevent for duplicate x
        x_out = rng_out * np.cos(az_out) + 1e-6 * np.random.randn(len(az_out))
        y_out = rng_out * np.sin(az_out)

        # If we have a smoothing parameter, run a B-spline on the output
        if smoothing is not None:
            idx_sort = x_out.argsort()
            spline = make_smoothing_spline(
                x_out[idx_sort], y_out[idx_sort], lam=smoothing
            )
            y_out = spline(x_out)

        # boundary is x and y vertices
        boundary = np.concatenate((x_out[:, None], y_out[:, None]), axis=1)

        return boundary


# @MODELS.register_module()
# class MMSegBevFovSegmenter(_MMSegmenter):
#     MODE = "semseg"

#     def __init__(
#         self,
#         model="unet",
#         dataset="carla",
#         gpu=0,
#         iteration="latest",
#         *args,
#         **kwargs,
#     ):
#         super().__init__(model=model, dataset=dataset, gpu=gpu, iteration=iteration, **kwargs)

#     def _execute(self, data: Union[np.ndarray, ImageData], **kwargs) -> SemSegImageData:
#         from mmseg.apis import inference_model

#         d_in = data if isinstance(data, np.ndarray) else data.data
#         result = inference_model(self.model, d_in)
#         return result

#     @staticmethod
#     def parse_mm_model_from_checkpoint(model, dataset, iteration):
#         # set the dataset strings
#         dataset = dataset.lower()
#         iter_str = "iter_{}.pth".format(iteration) if iteration != "latest" else "last_checkpoint"

#         # parse the model
#         if model == "unet":
#             if dataset == "carla":
#                 config_file = "work_dirs/unet_fov/unet_fov.py"
#                 checkpoint_file = f"work_dirs/unet_fov/{iter_str}"
#             else:
#                 raise NotImplementedError(f"{model}, {dataset} not compatible yet")
#         else:
#             raise NotImplementedError(model)

#         return config_file, checkpoint_file
