import sys

from avstack.modules.fusion import clustering

sys.path.append("tests/")
from utilities import get_object_global


def test_clustering_1_agent():
    n_objects = 10
    n_agents = 1
    objects = [[get_object_global(seed=i) for i in range(n_objects)] for _ in range(n_agents)]
    clusterer = clustering.SampledAssignmentClustering(assign_radius=4)
    clusters, object_to_cluster_map = clusterer(objects)
    assert len(clusters) == n_objects
    assert object_to_cluster_map == {k:{i:i for i in range(n_objects)} for k in range(n_agents)}


def test_clustering_2_agents():
    n_objects = 10
    n_agents = 2
    objects = [[get_object_global(seed=i) for i in range(n_objects)] for _ in range(n_agents)]
    clusterer = clustering.SampledAssignmentClustering(assign_radius=4)
    clusters, object_to_cluster_map = clusterer(objects)
    assert len(clusters) == n_objects
    assert object_to_cluster_map == {k:{i:i for i in range(n_objects)} for k in range(n_agents)}


def test_clustering_3_agents():
    n_objects = 10
    n_agents = 3
    objects = [[get_object_global(seed=i) for i in range(n_objects)] for _ in range(n_agents)]
    clusterer = clustering.SampledAssignmentClustering(assign_radius=4)
    clusters, object_to_cluster_map = clusterer(objects)
    assert len(clusters) == n_objects
    assert object_to_cluster_map == {k:{i:i for i in range(n_objects)} for k in range(n_agents)}