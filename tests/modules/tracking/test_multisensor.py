import numpy as np

from avstack.datastructs import DataContainer
from avstack.geometry import GlobalOrigin3D, ReferenceFrame, Sphere
from avstack.modules.perception import detections
from avstack.modules.tracking import MeasurementBasedMultiTracker, tracker3d


def test_msmt_based_multitracker():
    # make platform
    n_platforms = 4
    platforms = []
    fovs = []
    for i in range(n_platforms):
        platforms.append(
            ReferenceFrame(
                x=np.array([i, 0, 0]), q=np.quaternion(1), reference=GlobalOrigin3D
            )
        )
        if i % 2:
            # fov as shape
            fovs.append(Sphere(radius=1))
        else:
            # fov as hull
            fovs.append(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]))

    # make detections
    n_frames = 5
    dt = 0.1
    n_dets = [3, 2, 2, 3]
    dets_all = []
    for frame in range(n_frames):
        timestamp = frame * dt
        dets_frame = {}
        for i in range(n_platforms):
            dets = []
            for j in range(n_dets[i]):
                dets.append(
                    detections.CentroidDetection(
                        source_identifier=f"platform-{i}",
                        centroid=np.array([j / 2, 0, 0]),
                        reference=platforms[i],
                    )
                )
            dets_frame[i] = DataContainer(
                frame=frame,
                timestamp=timestamp,
                data=dets,
                source_identifier=f"platform-{i}",
            )
        dets_all.append(dets_frame)

    # run tracker
    tracker = MeasurementBasedMultiTracker(
        tracker=tracker3d.BasicXyzTracker(assign_radius=0.1)
    )
    for frame in range(n_frames):
        tracker(
            detections=dets_all[frame],
            fovs=fovs,
            platforms=platforms,
        )
    assert n_dets[0] < len(tracker.tracker.tracks_active) < sum(n_dets)
