from typing import Dict

from avstack.config import ALGORITHMS, ConfigDict
from avstack.datastructs import DataContainer
from avstack.geometry import ReferenceFrame
from avstack.modules.perception.detections import CentroidDetection
from avstack.modules.tracking.tracks import GroupTrack


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
        fuseds = [
            CentroidDetection(
                source_identifier="cluster",
                centroid=self.fusion(cluster)[0][: len(cluster[0].position)],
                reference=cluster[0].reference,
                obj_type=cluster[0].obj_type,
                score=None,
            )
            for cluster in clusters
        ]
        clusters_fused = DataContainer(
            frame=frame,
            timestamp=timestamp,
            data=fuseds,
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
