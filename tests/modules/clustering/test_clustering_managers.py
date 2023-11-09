import sys

from avstack.geometry import GlobalOrigin3D
from avstack.modules.clustering import clusterers, managers
from avstack.modules.tracking import tracker3d


sys.path.append("tests/")
from utilities import get_object_global


def get_longitudinal_objects(n_objects, n_agents, n_frames, dt):
    obj_data = []
    for i_frame in range(n_frames):
        objs = {
            i_agent: [get_object_global(seed=j_object) for j_object in range(n_objects)]
            for i_agent in range(n_agents)
        }
        obj_data.append([i_frame, dt * i_frame, objs])
    return obj_data


def test_cluster_tracker():
    clustering = clusterers.SampledAssignmentClustering(assign_radius=2)
    tracker = tracker3d.BasicXyzTracker(threshold_confirmed=3, threshold_coast=3)
    platform = GlobalOrigin3D
    ctracker = managers.ClusterTracker(clustering, tracker, platform)

    n_objects = 8
    n_agents = 3
    n_frames = 10
    dt = 0.10
    obj_data = get_longitudinal_objects(
        n_objects=n_objects, n_agents=n_agents, n_frames=n_frames, dt=dt
    )

    for frame, timestamp, objs in obj_data:
        ctracks = ctracker(objects=objs, frame=frame, timestamp=timestamp)

    assert len(ctracks) == n_objects
