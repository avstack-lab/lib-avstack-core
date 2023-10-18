from typing import List

import numpy as np
from avstack.modules import assignment


class SampledAssignmentClustering:
    """Run assignment by sampling one object from a cluster

    Assumes each sublist does not contain duplicates
    """

    def __init__(self, assign_radius: float = 8.0) -> None:
        self.assign_radius = assign_radius

    def __call__(self, list_objects: List[list]) -> dict:
        assert len(list_objects) > 0
        object_clusters = [[obj] for obj in list_objects[0]]
        for objects in list_objects[1:]:
            # obtain cost matrix
            A = np.zeros((len(objects), len(object_clusters)))
            for i, obj1 in enumerate(objects):
                for j, tclust in enumerate(object_clusters):
                    # sample one object from cluster randomly
                    obj2 = np.random.choice(tclust)
                    # Compute cost
                    dist = obj1.position.distance(obj2.position)
                    cost = dist - self.assign_radius
                    A[i, j] = cost
            # perform assignment
            assign_sol = assignment.gnn_single_frame_assign(
                A, all_assigned=False, cost_threshold=self.assign_radius
            )
            # add assignments to clusters and start new clusters with lone dets
            for obj_idx, clust_idx in assign_sol.iterate_over("rows").items():
                object_clusters[list(clust_idx.keys())[0]].append(objects[obj_idx])
            for obj_idx in assign_sol.unassigned_rows:
                object_clusters.append([objects[obj_idx]])
        return object_clusters


class HierarchicalAssignmentClustering:
    """Run assignment pairwise from binary tree for efficiency"""
