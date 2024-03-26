from copy import deepcopy

import numpy as np

from avstack.geometry import BoundingBox3D, BoxSize, Pose, Rotation, Vector, WorldFrame
from avstack.modules import clustering, fusion, tracking


def get_n_tracks(n_tracks=2):
    t0 = 0.0
    obj_type = "car"
    pos = Vector(np.array([-1, -2, -3]), WorldFrame)
    rot = Rotation(np.quaternion(1), WorldFrame)
    pose = Pose(pos, rot)
    box_size = BoxSize(2, 2, 4)
    box1 = BoundingBox3D(pose, box_size)
    tracks = [
        tracking.tracker3d.BasicBoxTrack3D(t0, deepcopy(box1), WorldFrame, obj_type)
        for _ in range(n_tracks)
    ]
    return tracks


def test_no_fusion():
    fuser = fusion.track_to_track.NoFusion()
    clusterer = clustering.NoClustering()
    tracks = get_n_tracks(n_tracks=2)
    clusters = clusterer({0: tracks}, frame=0, timestamp=0)
    tracks_fused = [fuser(cluster) for cluster in clusters]
    assert [tracks_fused[0][0], tracks_fused[1][0]] == tracks


def test_ci_fusion_base_naive_bayes():
    n_dim = 9
    n_agents = 5
    x = np.random.randn(n_dim)
    P = 20 * np.eye(9) + np.random.rand(n_dim, n_dim)
    xs = [x for _ in range(n_agents)]
    Ps = [P for _ in range(n_agents)]
    x_f, P_f = fusion.track_to_track.ci_fusion(xs, Ps, w_method="naive_bayes")
    assert np.allclose(x_f, x)
    assert np.allclose(P_f, P)


def test_fusion_into_boxtrack():
    tracks = get_n_tracks(n_tracks=4)
    fuser = fusion.CovarianceIntersectionFusionToBox()
    fused_out = fuser(tracks)
    assert isinstance(fused_out, tracking.tracker3d.BasicBoxTrack3D)
