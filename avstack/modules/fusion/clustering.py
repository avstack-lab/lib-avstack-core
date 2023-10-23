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
        """Perform clustering

        input:
            list_objects -- list of list where list elements are each set of objects 
                from e.g. an agent and each sublist are the objects
                e.g., list_objects = [ objects_agent_1, objects_agent_2, ... ]
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
        assert len(list_objects) > 0
        clusters = [[obj] for obj in list_objects[0]]

        object_to_cluster_map = {0:{i:i for i in range(len(list_objects[0]))}}

        for idx_agent, objects in enumerate(list_objects[1:]):
            idx_agent += 1  # bc zero index start
            # obtain cost matrix
            A = np.zeros((len(objects), len(clusters)))
            for i, obj1 in enumerate(objects):
                for j, tclust in enumerate(clusters):
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
                clust_id = list(clust_idx.keys())[0]
                clusters[clust_id].append(objects[obj_idx])
                if idx_agent not in object_to_cluster_map:
                    object_to_cluster_map[idx_agent] = {obj_idx:clust_id}
                else:
                    object_to_cluster_map[idx_agent][obj_idx] = clust_id
            for obj_idx in assign_sol.unassigned_rows:
                clusters.append([objects[obj_idx]])
                if idx_agent not in object_to_cluster_map:
                    object_to_cluster_map[idx_agent] = {obj_idx:len(clusters)}
                else:
                    object_to_cluster_map[idx_agent][obj_idx] = len(clusters)

        return clusters, object_to_cluster_map


class HierarchicalAssignmentClustering:
    """Run assignment pairwise from binary tree for efficiency"""
