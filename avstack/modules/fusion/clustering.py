import itertools
from typing import Dict, List, Union

import numpy as np


class Cluster:
    id_iter = itertools.count()

    def __init__(self, *args: List[tuple]) -> None:
        """A cluster of tracks representing an object

        inputs:
        tracks - a dictionary where keys are agent ID and values are tracks
        """
        self.ID = next(self.id_iter)
        self.agent_IDs = []
        self.track_IDs = []
        self.tracks = []
        for arg in args:
            if not isinstance(arg, tuple):
                raise TypeError("Input arguments must be tuples")
            else:
                self.agent_IDs.append(arg[0])
                self.track_IDs.append(arg[1].ID)
                self.tracks.append(arg[1])

    def __str__(self) -> str:
        return "Cluster ({}, {} elements)".format(self.ID, len(self.tracks))

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.tracks)

    def append(self, atrack: tuple) -> None:
        if not isinstance(atrack, tuple):
            raise TypeError("Input arguments must be tuples")
        else:
            self.agent_IDs.append(atrack[0])
            self.track_IDs.append(atrack[1].ID)
            self.tracks.append(atrack[1])

    def centroid(self):
        return np.mean([trk.position.x for trk in self.tracks], axis=0)

    def contains(self, agent_ID, track) -> bool:
        a_idxs = [i for i, ID in enumerate(self.agent_IDs) if i == agent_ID]
        t_idxs = [i for i, trk in enumerate(self.tracks) if trk.ID == track.ID]
        return any([t_idx in a_idxs for t_idx in t_idxs])

    def distance(self, track) -> float:
        return track.distance(self.centroid())

    def get_tracks_by_agent_ID(self, ID: int) -> list:
        return [
            trk for trk, agent_ID in zip(self.tracks, self.agent_IDs) if agent_ID == ID
        ]


class ClusterSet(list):
    def contains(self, agent_ID: int, track) -> Union[bool, int]:
        for i, clust in enumerate(self):
            if clust.contains(agent_ID, track):
                return i
        return False


class SampledAssignmentClustering:
    """Run assignment by sampling one object from a cluster

    Assumes each sublist does not contain duplicates
    """

    def __init__(self, assign_radius: float = 8.0) -> None:
        self.assign_radius = assign_radius

    def __call__(self, objects: Dict[int, list]) -> dict:
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

        clusters = ClusterSet()

        # check out other objects
        for agent_ID, tracks in objects.items():
            for track in tracks:
                if not clusters.contains(agent_ID, track):
                    distances = np.array([clust.distance(track) for clust in clusters])
                    if any(distances <= self.assign_radius):
                        idx_min = np.argmin(distances)
                        clusters[idx_min].append((agent_ID, track))
                    else:
                        clusters.append(Cluster((agent_ID, track)))

        return clusters


class HierarchicalAssignmentClustering:
    """Run assignment pairwise from binary tree for efficiency"""
