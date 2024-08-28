from typing import Any, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from avstack.geometry import Polygon
    from avstack.sensors import LidarData


import numpy as np
from scipy.interpolate import make_smoothing_spline

from avstack.config import MODELS
from avstack.geometry import Polygon
from avstack.modules import BaseModule
from avstack.utils.decorators import apply_hooks


class _LidarFovEstimator(BaseModule):
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)


@MODELS.register_module()
class ConcaveHullLidarFOVEstimator(_LidarFovEstimator):
    def __init__(
        self,
        concavity: int = 1,
        length_threshold: float = 4,
        max_height: float = np.inf,
        name: str="fov_estimator",
        *args,
        **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)
        self.concavity = concavity
        self.length_threshold = length_threshold
        self.max_height = max_height

    @apply_hooks
    def __call__(self, pc: "LidarData", in_global: bool = False, *args: Any, **kwds: Any) -> "Polygon":
        fov = pc.concave_hull_bev(
            concavity=self.concavity,
            length_threshold=self.length_threshold,
            in_global=in_global,
            max_height=self.max_height,
        )
        return fov


@MODELS.register_module()
class RayTraceBevLidarFovEstimator(_LidarFovEstimator):
    def __init__(
        self,
        z_min: float = -3.0,
        z_max: float = 3.0,
        az_bin: float = 0.05,
        rng_bin: float = 0.50,
        rng_max: float = 100,
        name: str="fov_estimator",
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
        self.az_bin = az_bin
        self.rng_bin = rng_bin
        self.rng_max = rng_max

    @apply_hooks
    def __call__(self, pc: "LidarData", in_global: bool = False, *args: Any, **kwds: Any) -> "Polygon":
        # project into BEV
        if in_global:
            raise NotImplementedError("Global transform not implemented yet")
        pc_bev = pc.project_to_2d_bev(z_min=self.z_min, z_max=self.z_max)

        # transform to polar coordinates
        # azimuth defined here as zero along positive x axis
        pc_bev_azimuth = np.arctan2(pc_bev.data.x[:,1], pc_bev.data.x[:,0])
        pc_bev_range = np.linalg.norm(pc_bev.data.x, axis=1)

        # NOTE: arctan2 output is on [-pi, pi] so ensure az queries are the same
        # make edges for the bins
        num_az = int(2*np.pi // self.az_bin)
        az_edges = np.linspace(-np.pi, np.pi, num=num_az)
        num_rng = int(self.rng_max // self.rng_bin)
        rng_edges = np.linspace(0, self.rng_max, num=num_rng)

        # populate a grid
        col_indices = np.array([list(range(len(rng_edges)-1))] * (len(az_edges)-1))
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
        boundary = np.concatenate((x_out[:,None], y_out[:,None]), axis=1)
        fov = Polygon(
            boundary=boundary,
            reference=pc_bev.reference,
            frame=pc.frame,
            timestamp=pc.timestamp
        )
        return fov
    


@MODELS.register_module()
class RayTraceBevLidarFovEstimator(_LidarFovEstimator):
    def __init__(
        self,
        z_min: float = -3.0,
        z_max: float = 3.0,
        az_tolerance: float = 0.05,
        smoothing: Union[None, float] = None,
        name: str="fov_estimator",
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
        self.smoothing = smoothing
        self.az_tol = az_tolerance

    @apply_hooks
    def __call__(self, pc: "LidarData", in_global: bool = False, *args: Any, **kwds: Any) -> "Polygon":
        # project into BEV
        if in_global:
            raise NotImplementedError("Global transform not implemented yet")
        pc_bev = pc.project_to_2d_bev(z_min=self.z_min, z_max=self.z_max)

        # transform to polar coordinates
        # azimuth defined here as zero along positive x axis
        pc_bev_azimuth = np.arctan2(pc_bev.data.x[:,1], pc_bev.data.x[:,0])
        pc_bev_range = np.linalg.norm(pc_bev.data.x, axis=1)

        # NOTE: arctan2 output is on [-pi, pi] so ensure az queries are the same
        az_query = np.linspace(-np.pi, np.pi, num=100)
        az_out = []
        rng_out = []
        for az_q in az_query:
            # NOTE: this currently doesn't handle the wrap around condition
            # look for azimuths within the tolerance
            idx_az_valid = (pc_bev_azimuth >= az_q-self.az_tol) & \
                (pc_bev_azimuth <= az_q+self.az_tol)
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
        if self.smoothing is not None:
            idx_sort = x_out.argsort()
            spline = make_smoothing_spline(
                x_out[idx_sort],
                y_out[idx_sort],
                lam=self.smoothing
            )
            y_out = spline(x_out)

        # Construct the fov model
        boundary = np.concatenate((x_out[:,None], y_out[:,None]), axis=1)
        fov = Polygon(
            boundary=boundary,
            reference=pc_bev.reference,
            frame=pc.frame,
            timestamp=pc.timestamp
        )
        return fov
    

# query the spline at regular azimuths to get the polygon
# construct a smoothing spline interpolation of rng = f(az)
# we must sort the data for the input to the spline

# az_out = az_edges
# rng_out = rng_edges[np.argmax]

# az_query = np.linspace(-np.pi, np.pi, num=100)
# az_out = []
# rng_out = []
# for az_q in az_query:
#     # NOTE: this currently doesn't handle the wrap around condition
#     # look for azimuths within the tolerance
#     idx_az_valid = (pc_bev_azimuth >= az_q-self.az_tol) & \
#         (pc_bev_azimuth <= az_q+self.az_tol)
#     if sum(idx_az_valid) == 0:
#         continue
#     # take the max range over these azimuths
#     rng_out.append(max(pc_bev_range[idx_az_valid]))
#     az_out.append(az_q)
# az_out = np.asarray(az_out)
# rng_out = np.asarray(rng_out)

# # Construct a polygon model
# # add small amount of noise to prevent for duplicate x


# # If we have a smoothing parameter, run a B-spline on the output
# if self.smoothing is not None:
#     idx_sort = x_out.argsort()
#     spline = make_smoothing_spline(
#         x_out[idx_sort],
#         y_out[idx_sort],
#         lam=self.smoothing
#     )
#     y_out = spline(x_out)