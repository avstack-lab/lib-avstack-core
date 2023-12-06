from typing import Dict

import numpy as np

from avstack.config import ALGORITHMS, ConfigDict
from avstack.datastructs import DataContainer
from avstack.geometry import ReferenceFrame
from avstack.modules.perception.detections import BoxDetection, CentroidDetection
from avstack.modules.tracking.tracks import BasicBoxTrack3D, GroupTrack


@ALGORITHMS.register_module()
class GroupTrackerWrapper:
    def __init__(self, fusion, tracker):
        self.fusion = (
            ALGORITHMS.build(fusion) if isinstance(fusion, ConfigDict) else fusion
        )
        self.tracker = (
            ALGORITHMS.build(tracker) if isinstance(tracker, ConfigDict) else tracker
        )

    def __call__(
        self,
        clusters: Dict[int, DataContainer],
        platform: ReferenceFrame,
        frame: int,
        timestamp: float,
    ) -> DataContainer:

        # Track on the detections
        fused_dets = []
        for cluster in clusters:
            fuse_out = self.fusion(cluster)
            if isinstance(fuse_out, BasicBoxTrack3D):
                fuse_as_det = BoxDetection(
                    source_identifier="cluster",
                    box=fuse_out.box3d,
                    reference=cluster.reference,
                    obj_type=fuse_out.obj_type,
                    score=None,
                )
            elif isinstance(fuse_out, tuple):
                if isinstance(fuse_out[0], np.ndarray):
                    fuse_as_det = CentroidDetection(
                        source_identifier="cluster",
                        centroid=fuse_out[0][: len(cluster[0].position)],
                        reference=cluster[0].reference,
                        obj_type=cluster[0].obj_type,
                        score=None,
                    )
                else:
                    raise NotImplemented(fuse_out)
            else:
                raise NotImplementedError(fuse_out)
            fused_dets.append(fuse_as_det)

        clusters_fused = DataContainer(
            frame=frame,
            timestamp=timestamp,
            data=fused_dets,
            source_identifier="cluster-dets",
        )
        cluster_tracks = self.tracker(
            t=timestamp,
            frame=frame,
            detections=clusters_fused,
            platform=platform,
        )

        # Convert to a group track datastructure to keep track of members
        group_tracks = DataContainer(
            frame=frame,
            timestamp=timestamp,
            data=[],
            source_identifier="cluster-groups",
        )
        for i, ct in enumerate(cluster_tracks):
            if i in self.tracker.last_assignment.unassigned_cols:
                members = []
            else:
                r_assigns = self.tracker.last_assignment.assigns_by_col(i)
                if len(r_assigns) > 1:
                    import pdb

                    pdb.set_trace()
                else:
                    members = clusters[r_assigns[0]]
            group_tracks.append(GroupTrack(ct, members))

        return group_tracks
