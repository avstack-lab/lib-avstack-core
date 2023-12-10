from typing import Dict

import numpy as np

from avstack.config import ALGORITHMS
from avstack.datastructs import DataContainer

from .types import Cluster, ClusterSet


@ALGORITHMS.register_module()
class NoClustering:
    """Each track is its own cluster"""

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


@ALGORITHMS.register_module()
class SampledAssignmentClusterer:
    """Run assignment by sampling one object from a cluster

    Assumes each sublist does not contain duplicates
    """

    def __init__(self, assign_radius: float = 8.0) -> None:
        self.assign_radius = assign_radius

    def __call__(
        self,
        objects: Dict[int, DataContainer],
        frame: int,
        timestamp: float,
        check_reference: bool = True,
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
        assert len(objects) > 0

        clusters = ClusterSet(
            frame=frame,
            timestamp=timestamp,
            data=[],
            source_identifier="assignment-clustering",
        )

        # check out other objects
        for agent_ID, tracks in objects.items():
            for track in tracks:
                if not clusters.contains(agent_ID, track):
                    distances = np.array(
                        [
                            clust.distance(track, check_reference=check_reference)
                            for clust in clusters
                        ]
                    )
                    try:
                        if any(distances <= self.assign_radius):
                            idx_min = np.argmin(distances)
                            clusters[idx_min].append((agent_ID, track))
                        else:
                            clusters.append(Cluster((agent_ID, track)))
                    except ValueError:
                        import pdb

                        pdb.set_trace()
                        raise

        return clusters


@ALGORITHMS.register_module()
class HierarchicalAssignmentClustering:
    """Run assignment pairwise from binary tree for efficiency"""
