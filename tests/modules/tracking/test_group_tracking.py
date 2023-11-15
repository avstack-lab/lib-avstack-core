import sys

from avstack.geometry import GlobalOrigin3D
from avstack.modules.clustering import Cluster, ClusterSet
from avstack.modules.fusion import CovarianceIntersectionFusion
from avstack.modules.tracking import grouptrack, tracker3d, tracks


sys.path.append("tests/")
from utilities import get_object_global


def make_longitudinal_cluster_data(n_objects=4, n_agents=3, dt=0.1, n_frames=10):
    clusters_all = []
    for frame in range(n_frames):
        timestamp = dt * frame
        clusters = [
            Cluster(
                *[
                    (
                        i_agent,
                        tracks.XyzFromXyzTrack(
                            t0=timestamp,
                            xyz=get_object_global(seed=i_object).position.x,
                            reference=GlobalOrigin3D,
                            obj_type=None,
                        ),
                    )
                    for i_agent in range(n_agents)
                ]
            )
            for i_object in range(n_objects)
        ]
        clusters_all.append(
            ClusterSet(
                frame=frame,
                timestamp=timestamp,
                data=clusters,
                source_identifier="clusters",
            )
        )
    return clusters_all


def test_grouptracking_clusters():
    n_objects = 5
    n_agents = 3
    clusters_sets = make_longitudinal_cluster_data(
        n_objects=n_objects, n_agents=n_agents
    )
    group_tracker = grouptrack.GroupTrackerWrapper(
        fusion=CovarianceIntersectionFusion(),
        tracker=tracker3d.BasicXyzTracker(),
    )
    for clusters in clusters_sets:
        group_tracks = group_tracker(
            clusters,
            frame=clusters.frame,
            timestamp=clusters.timestamp,
            platform=GlobalOrigin3D,
        )
    assert len(group_tracks) == n_objects
    for gtracks in group_tracks:
        assert len(gtracks.members) == n_agents
