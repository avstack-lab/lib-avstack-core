import sys

from avstack.modules.clustering import clusterers


sys.path.append("tests/")
from utilities import get_object_global


def get_objects(n_objects, n_agents):
    return {
        i: [get_object_global(seed=i) for i in range(n_objects)]
        for i in range(n_agents)
    }


def test_cluster_data_structure():
    agent_ID = 1
    n_objects = 10
    objects = get_objects(n_objects=n_objects, n_agents=1)
    cluster = clusterers.Cluster(
        *[(agent_ID, obj) for obj in list(objects.values())[0]]
    )
    assert len(cluster) == n_objects
    assert len(set(cluster.agent_IDs)) == 1
    assert len(cluster.get_objects_by_agent_ID(agent_ID)) == n_objects
