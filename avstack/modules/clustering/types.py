import itertools
from typing import List, Union

import numpy as np

from avstack.datastructs import DataContainer
from avstack.geometry import PassiveReferenceFrame, Position


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
        self.reference = None
        for arg in args:
            if not isinstance(arg, tuple):
                raise TypeError("Input arguments must be tuples")
            else:
                self.agent_IDs.append(arg[0])
                self.track_IDs.append(arg[1].ID)
                self.tracks.append(arg[1])
                if self.reference is None:
                    self.reference = arg[1].reference
                else:
                    if isinstance(self.reference, PassiveReferenceFrame):
                        assert self.reference == args[1].reference
                    else:
                        assert self.reference.allclose(arg[1].reference)

    def __str__(self) -> str:
        return "Cluster ({}, {} elements)".format(self.ID, len(self.tracks))

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, key):
        return self.tracks[key]

    def append(self, atrack: tuple) -> None:
        if not isinstance(atrack, tuple):
            raise TypeError("Input arguments must be tuples")
        else:
            self.agent_IDs.append(atrack[0])
            self.track_IDs.append(atrack[1].ID)
            self.tracks.append(atrack[1])
            if self.reference is None:
                self.reference = atrack[1].reference
            else:
                if isinstance(self.reference, PassiveReferenceFrame):
                    assert self.reference == atrack[1].reference
                else:
                    assert self.reference.allclose(atrack[1].reference)

    def centroid(self):
        """Ensure all are of the same reference"""
        x_mean = np.mean([trk.position.x for trk in self.tracks], axis=0)
        return Position(x_mean, reference=self.reference)

    def contains(self, agent_ID, track) -> bool:
        a_idxs = [i for i, ID in enumerate(self.agent_IDs) if i == agent_ID]
        t_idxs = [i for i, trk in enumerate(self.tracks) if trk.ID == track.ID]
        return any([t_idx in a_idxs for t_idx in t_idxs])

    def distance(self, track, check_reference: bool=True) -> float:
        return track.distance(self.centroid(), check_reference=check_reference)

    def get_tracks_by_agent_ID(self, ID: int) -> list:
        return [
            trk for trk, agent_ID in zip(self.tracks, self.agent_IDs) if agent_ID == ID
        ]


class ClusterSet(DataContainer):
    def contains(self, agent_ID: int, track) -> Union[bool, int]:
        for i, clust in enumerate(self):
            if clust.contains(agent_ID, track):
                return i
        return False
