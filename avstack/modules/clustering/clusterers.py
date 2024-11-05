from typing import Dict, Union

import numpy as np

from avstack.config import MODELS
from avstack.datastructs import DataContainer
from avstack.utils.decorators import apply_hooks

from ..base import BaseModule
from .types import Cluster, ClusterSet


class _BaseClustering(BaseModule):
    def __init__(self, name="clustering", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)


@MODELS.register_module()
class NoClustering(_BaseClustering):
    """Each track is its own cluster"""

    @apply_hooks
    def __call__(
        self, objects: Dict[int, DataContainer], frame: int, timestamp: int
    ) -> ClusterSet:
        clusters = ClusterSet(
            frame=frame, timestamp=timestamp, data=[], source_identifier="no-clustering"
        )
        for agent_ID, tracks in objects.items():
            for track in tracks:
                clusters.append(Cluster((agent_ID, track)))
        return clusters


@MODELS.register_module()
class BasicClusterer(_BaseClustering):
    """Cluster the objects in a list based on threshold"""

    def __init__(self, assign_radius: float = 1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.assign_radius = assign_radius

    @apply_hooks
    def __call__(
        self,
        objects: Union[list, DataContainer],
        agent_ID: int,
        frame: int = None,
        timestamp: float = None,
        check_reference: bool = False,
        *args,
        **kwargs,
    ) -> ClusterSet:
        clusters = ClusterSet(
            frame=frame if frame is not None else objects.frame,
            timestamp=timestamp if timestamp is not None else objects.timestamp,
            data=[],
            source_identifier="assignment-clustering",
        )

        # Check object to cluster distances
        for obj in objects:
            distances = []
            for clust in clusters:
                distances.append(clust.distance(obj, check_reference=check_reference))
            if any(np.array(distances) <= self.assign_radius):
                idx_min = np.argmin(distances)
                clusters[idx_min].append((agent_ID, obj))
            else:
                clusters.append(Cluster((agent_ID, obj)))
        return clusters


@MODELS.register_module()
class SampledAssignmentClusterer(_BaseClustering):
    """Run assignment by sampling one object from a cluster

    Assumes each sublist does not contain duplicates
    """

    def __init__(self, assign_radius: float = 8.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.assign_radius = assign_radius

    @apply_hooks
    def __call__(
        self,
        objects: Dict[int, DataContainer],
        frame: int,
        timestamp: float,
        check_reference: bool = True,
        *args,
        **kwargs,
    ) -> ClusterSet:
        """Perform clustering

        input:
            list_objects -- list of list where list elements are each set of objects
                from e.g. an agent and each sublist are the objects
                e.g., objects = {agent_1_ID: objects_agent_1, agent_2_ID: objects_agent_2, ... }
                    objects_agent_1 = [ object_1, object_2 ]
                    objects_agent_2 = [ object_3, object_4 ]

        returns:
            clusters -- list of list where list elements are each cluster and
                sublist elements are all objects belonging to cluster
                e.g., clusters = [ cluster_1, cluster_2, ... ]
                      cluster_1 = [ object_1, object_3, ... ]
                      cluster_2 = [ object_2, object_4, ... ]

            object_to_cluster_map -- dict mapping agent idx and object idx to cluster idx
                e.g., object_to_cluster_map = {1:{1:1, 2:2}, 2:{1:1, 2:2}}

        """
        return self.cluster(
            objects, frame, timestamp, self.assign_radius, check_reference
        )

    @staticmethod
    def cluster(objects, frame, timestamp, assign_radius, check_reference):
        assert len(objects) > 0

        clusters = ClusterSet(
            frame=frame,
            timestamp=timestamp,
            data=[],
            source_identifier="assignment-clustering",
        )

        # check out other objects
        for agent_ID, objs in objects.items():
            for obj in objs:
                # if not clusters.contains(agent_ID, obj):
                distances = np.array(
                    [
                        clust.distance(obj, check_reference=check_reference)
                        for clust in clusters
                    ]
                )
                try:
                    if any(distances <= assign_radius):
                        idx_min = np.argmin(distances)
                        clusters[idx_min].append((agent_ID, obj))
                    else:
                        clusters.append(Cluster((agent_ID, obj)))
                except ValueError:
                    import pdb

                    pdb.set_trace()
                    raise
        return clusters


@MODELS.register_module()
class HierarchicalAssignmentClustering(_BaseClustering):
    """Run assignment pairwise from binary tree for efficiency"""
