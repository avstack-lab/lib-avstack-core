import itertools
from typing import List, Union

import numpy as np

from avstack.datastructs import DataContainer
from avstack.geometry import PassiveReferenceFrame, Position


class Cluster:
    id_iter = itertools.count()

    def __init__(self, *args: List[tuple]) -> None:
        """A cluster of objects representing a single obj

        inputs:
        objects - a dictionary where keys are agent ID and values are objects
        """
        self.ID = next(self.id_iter)
        self.agent_IDs = []
        self.obj_IDs = []
        self.objects = []
        self.reference = None
        for arg in args:
            if not isinstance(arg, tuple):
                raise TypeError("Input arguments must be tuples")
            else:
                self.agent_IDs.append(arg[0])
                try:
                    self.obj_IDs.append(arg[1].ID)
                except AttributeError:
                    pass
                self.objects.append(arg[1])
                if self.reference is None:
                    self.reference = arg[1].reference
                else:
                    if isinstance(self.reference, PassiveReferenceFrame):
                        assert self.reference == args[1].reference
                    else:
                        assert self.reference.allclose(arg[1].reference)

    def __str__(self) -> str:
        return "Cluster ({}, {} elements)".format(self.ID, len(self.objects))

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, key):
        return self.objects[key]

    def append(self, an_obj: tuple) -> None:
        if not isinstance(an_obj, tuple):
            raise TypeError("Input arguments must be tuples")
        else:
            self.agent_IDs.append(an_obj[0])
            try:
                self.obj_IDs.append(an_obj[1].ID)
            except AttributeError:
                pass
            self.objects.append(an_obj[1])
            if self.reference is None:
                self.reference = an_obj[1].reference
            else:
                if isinstance(self.reference, PassiveReferenceFrame):
                    assert self.reference == an_obj[1].reference
                else:
                    assert self.reference.allclose(an_obj[1].reference)

    def centroid(self):
        """Ensure all are of the same reference"""
        x_mean = np.mean([trk.position.x for trk in self.objects], axis=0)
        return Position(x_mean, reference=self.reference)

    def contains(self, agent_ID, obj) -> bool:
        try:
            a_idxs = [i for i, ID in enumerate(self.agent_IDs) if i == agent_ID]
            t_idxs = [i for i, trk in enumerate(self.objects) if trk.ID == obj.ID]
            return any([t_idx in a_idxs for t_idx in t_idxs])
        except AttributeError:
            return False

    def distance(self, obj, check_reference: bool = True) -> float:
        centroid = self.centroid()
        try:
            d = obj.distance(centroid, check_reference=check_reference)
        except AttributeError:
            d = obj.position.distance(centroid, check_reference=check_reference)
        return d

    def get_objects_by_agent_ID(self, ID: int) -> list:
        return [
            trk for trk, agent_ID in zip(self.objects, self.agent_IDs) if agent_ID == ID
        ]

    def sample(self):
        return np.random.choice(self.objects, size=1, replace=False)[0]


class ClusterSet(DataContainer):
    def contains(self, agent_ID: int, obj) -> Union[bool, int]:
        for i, clust in enumerate(self):
            if clust.contains(agent_ID, obj):
                return i
        return False
