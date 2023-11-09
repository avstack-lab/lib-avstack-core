import sys


sys.path.append("tests/")
from utilities import get_object_global


def get_longitudinal_clusters(n_objects, n_agents):
    return {
        i: [get_object_global(seed=i) for i in range(n_objects)]
        for i in range(n_agents)
    }


# def test_cluster_tracker():
#     clustering = clusterers.SampledAssignmentClustering(assign_radius=2)
#     tracker = tracker3d
#     platform = GlobalOrigin3D
#     ctracker = managers.ClusterTracker(clustering, tracker, platform)
#     cdata =  get_longitudinal_clusters(n_objects=n_objects, n_agents=n_agents, n_frames=n_frames)
#     for clusters in cdata:
#         pass
