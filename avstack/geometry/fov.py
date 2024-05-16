import itertools
from typing import Dict, FrozenSet, Union

import numpy as np

from avstack.config import GEOMETRY
from avstack.geometry.utils import in_hull


def points_in_fov(points, fov: Union["Shape", np.ndarray]):
    try:
        in_fov = fov.check_point(points)
    except AttributeError:
        in_fov = in_hull(points, fov)
    return in_fov


def get_disjoint_fov_subsets(fovs: Dict[int, "Shape"]) -> Dict[FrozenSet[int], "Shape"]:
    """compute all disjoint FOV subsets from all agents

    A disjoint subsets are subsets that have no overlap with each other.

    Computing all subsets amounts to looping over the 2^n number of
    distinct agent combinations and checking for FOV overlap. Some
    optimizations and caching can be done for efficiency.

    Reducing all subsets to "maximal" disjoint subsets amounts to
    pruning the list of subsets if the list of agent IDs is a strict subset
    of a larger list of agent IDs with overlap
    """
    agent_IDs = list(fovs.keys())
    fov_subsets = {
        frozenset({ID}): fov for ID, fov in fovs.items()
    }  # dict of agent ID sets to the subset
    overlap_ID_combos = {(ID,) for ID in agent_IDs}
    n = len(fovs)  # choose from all agents
    k = 2  # choose pairs of agents

    # Compute all disjoint subsets of overlap
    while True:
        # check all unique combinations of k agents for overlap
        found_a_combo = False
        agent_combos = itertools.combinations(range(n), k)
        for combo in agent_combos:
            IDs = frozenset([agent_IDs[idx] for idx in combo])

            # check if the km1 combos are valid
            all_subsets_valid = True
            all_km1_combos = list(itertools.combinations(IDs, k - 1))
            for km1_combo in all_km1_combos:
                if km1_combo not in overlap_ID_combos:
                    all_subsets_valid = False
                    break

            # check overlaps only if all subsets are valid
            if all_subsets_valid:
                # get starting point and remaining IDs
                IDs_use = all_km1_combos[0]  # just pick first arbitrarily
                overlap = fov_subsets[frozenset(IDs_use)]
                IDs_rem = IDs.difference(IDs_use)

                # check the remaining ID FOVs for overlap
                for ID in IDs_rem:
                    overlap = overlap.intersection(fovs[ID])
                    if overlap is None:
                        break
                else:
                    # modify the km1 subsets to reflect the disjoint nature
                    for km1_combo in all_km1_combos:
                        fov_subsets[frozenset(km1_combo)] = fov_subsets[
                            frozenset(km1_combo)
                        ].difference(overlap)
                    # add the new set to the dict
                    found_a_combo = True
                    fov_subsets[IDs] = overlap

        # check if we need to increment k and keep going
        if not found_a_combo:
            break  # we can't possibly find a more maximal one
        else:
            k += 1

    return fov_subsets


class Shape:
    @property
    def area(self):
        raise NotImplementedError

    def check_point(self, point: np.ndarray) -> bool:
        raise NotImplementedError


@GEOMETRY.register_module()
class Wedge(Shape):
    def __init__(self, radius: float, angle_start: float, angle_stop: float) -> None:
        """Define a wedge as a part of a circle

        Assumes unit circle convention where:
            (1, 0) is at 0
            (0, 1) is at pi/2
            (-1,0) is at pi or -pi
            (0,-1) is at -pi/2
        """
        self.radius = radius
        self.angle_start = angle_start
        self.angle_stop = angle_stop

    @property
    def angle_range(self):
        return (self.angle_stop % (2 * np.pi)) - (self.angle_start % (2 * np.pi))

    @property
    def area(self):
        return self.radius**2 * self.angle_range / 2

    def check_point(self, point: np.ndarray):
        if len(point.shape) == 1:
            point = point[:, None]
        flag_rng = np.linalg.norm(point, axis=0) <= self.radius
        az = np.arctan2(point[1, :], point[0, :])
        flag_az_1 = self.angle_start <= az
        flag_az_2 = az <= self.angle_stop
        return flag_rng & flag_az_1 & flag_az_2


@GEOMETRY.register_module()
class Circle(Shape):
    def __init__(self, radius: float, center: np.ndarray = np.zeros((2,))) -> None:
        self.radius = radius
        self.center = center

    def check_point(self, point: np.ndarray):
        return np.linalg.norm(point - self.center, axis=0) <= self.radius

    def intersection(self, other: "Shape") -> Union["Shape", None]:
        if isinstance(other, (Sphere, Circle)):
            d = np.linalg.norm(self.center - other.center)
            if (d + self.radius) < other.radius:
                return self
            elif (d + other.radius) < self.radius:
                return other
            elif d < (self.radius + other.radius):
                overlap = Vesica([self, other])
            else:
                overlap = None
        elif isinstance(other, Vesica):
            overlap = other.intersection(self)
        else:
            raise NotImplementedError
        return overlap

    def difference(self, other: "Shape") -> Union["Shape", None]:
        raise NotImplementedError

    def change_reference(self, reference):
        pass


@GEOMETRY.register_module()
class Sphere(Circle):
    def __init__(self, radius: float, center: np.ndarray = np.zeros((3,))) -> None:
        self.radius = radius
        self.center = center


@GEOMETRY.register_module()
class Vesica(Shape):
    def __init__(self, circles):
        """Shape formed by intersection of circles"""
        self.circles = circles

    def intersection(self, other: "Shape") -> Union["Shape", None]:
        if isinstance(other, Circle):
            for circle in self.circles:
                if np.linalg.norm(circle.center - other.center) >= max(
                    circle.radius, other.radius
                ):
                    overlap = None
                    break
            else:
                overlap = Vesica(self.circles + [other])
        elif isinstance(other, Vesica):
            raise NotImplementedError
        else:
            raise NotImplementedError
        return overlap

    def difference(self, other: "Shape") -> Union["Shape", None]:
        raise NotImplementedError


# class FieldOfView:
#     def check_point(self, point: np.ndarray):
#         raise NotImplementedError

#     def intersection(self, other: "FieldOfView") -> Union["FieldOfView", None]:
#         raise NotImplementedError

#     def difference(self, other: "FieldOfView") -> Union["FieldOfView", None]:
#         raise NotImplementedError


# class ParametricFieldOfView(FieldOfView):
#     def __init__(self) -> None:
#         """Uses a particular shape to form a field of view"""
#         super().__init__()

#     def check_point(self, point: np.ndarray):
#         return any([shape.check_point(point) for shape in self.shapes])


# def fov_from_radar(range_doppler):
#     raise NotImplementedError


# def fov_from_lidar(pc: LidarData):
#     """Use LiDAR data to estimate the field of view"""
#     raise NotImplementedError
