import sys

from avstack.modules.clustering import clusterers


sys.path.append("tests/")
from utilities import get_object_global


def get_objects(n_objects, n_agents):
    return {
        i: [get_object_global(seed=i) for i in range(n_objects)]
        for i in range(n_agents)
    }


def test_no_clustering():
    n_objects = 10
    n_agents = 2
    objects = get_objects(n_objects, n_agents)
    clusterer = clusterers.NoClustering()
    clusters = clusterer(objects, frame=0, timestamp=0)
    assert len(clusters) == n_objects * n_agents


def test_clustering_1_agent():
    n_objects = 10
    n_agents = 1
    objects = get_objects(n_objects, n_agents)
    clusterer = clusterers.SampledAssignmentClustering(assign_radius=1)
    clusters = clusterer(objects, frame=0, timestamp=0)
    assert len(clusters) == n_objects


def test_clustering_2_agents():
    n_objects = 10
    n_agents = 2
    objects = get_objects(n_objects, n_agents)
    clusterer = clusterers.SampledAssignmentClustering(assign_radius=1)
    clusters = clusterer(objects, frame=0, timestamp=0)
    assert len(clusters) == n_objects


def test_clustering_3_agents():
    n_objects = 10
    n_agents = 3
    objects = get_objects(n_objects, n_agents)
    clusterer = clusterers.SampledAssignmentClustering(assign_radius=1)
    clusters = clusterer(objects, frame=0, timestamp=0)
    assert len(clusters) == n_objects
