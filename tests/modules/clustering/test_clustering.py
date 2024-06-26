import sys

from avstack.modules.clustering import clusterers


sys.path.append("tests/")
from utilities import get_object_global


def get_objects(n_objects, n_agents):
    return {
        i_agent: [get_object_global(seed=j_object) for j_object in range(n_objects)]
        for i_agent in range(n_agents)
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
    clusterer = clusterers.SampledAssignmentClusterer(assign_radius=1)
    clusters = clusterer(objects, frame=0, timestamp=0)
    assert len(clusters) == n_objects


def test_clustering_2_agents():
    n_objects = 10
    n_agents = 2
    objects = get_objects(n_objects, n_agents)
    clusterer = clusterers.SampledAssignmentClusterer(assign_radius=1)
    clusters = clusterer(objects, frame=0, timestamp=0)
    assert len(clusters) == n_objects


def test_clustering_3_agents():
    n_objects = 10
    n_agents = 3
    objects = get_objects(n_objects, n_agents)
    clusterer = clusterers.SampledAssignmentClusterer(assign_radius=1)
    clusters = clusterer(objects, frame=0, timestamp=0)
    assert len(clusters) == n_objects


def test_basic_clusterer():
    n_objects = 5
    n_agents = 1
    objects = get_objects(n_objects, n_agents)[0]
    objects.append(objects[0])
    objects.append(objects[1])
    clusterer = clusterers.BasicClusterer(assign_radius=0.1)
    clusters = clusterer(objects, agent_ID=0, frame=0, timestamp=0)
    assert len(clusters) == len(objects) - 2
    assert len(objects) == n_objects + 2
