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
        dets_as_tracks = DataContainer(
            frame=frame,
            timestamp=timestamp,
            data=[self.fusion(cluster) for cluster in clusters],
            source_identifier="cluster-dets",
        )
        # HACK: to handle reference frame transformation
        dets_as_dets = DataContainer(
            frame=frame,
            timestamp=timestamp,
            data=[
                CentroidDetection(
                    source_identifier="centroid",
                    centroid=track.x[:3],
                    reference=track.reference,
                    obj_type=track.obj_type,
                    score=track.probability,
                )
                for track in dets_as_tracks
            ],
            source_identifier="cluster-dets",
        )
        cluster_tracks = self.tracker(
            t=timestamp,
            frame=frame,
            detections=dets_as_dets,
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


# class _BaseClusterManager:
#     def __call__(
#         self, t: float, frame: int, objects: Dict[int, DataContainer], **kwargs: Any
#     ):
#         cluster_data = self.cluster(
#             timestamp=t, frame=frame, objects=objects, **kwargs
#         )  # should already return datacontainer
#         return cluster_data


# class NoClusterTracker(_BaseClusterManager):
#     def __init__(self) -> None:
#         self.clusterer = NoClustering()

#     def cluster(
#         self, objects: Dict[int, DataContainer], frame: int, timestamp: float
#     ) -> ClusterSet:
#         clusters = self.clusterer(objects=objects, frame=frame, timestamp=timestamp)
#         return clusters


# class ClusterTracker(_BaseClusterManager):
#     def __init__(self, clusterer, fusion, tracker, platform) -> None:
#         """Perform tracking on the cluster centroids

#         Helps to maintain consistent cluster IDs
#         """
#         self.clusterer = clusterer
#         self.fusion = fusion
#         self.tracker = tracker
#         self.platform = platform

#     def cluster(
#         self, objects: Dict[int, DataContainer], frame: int, timestamp: float
#     ) -> DataContainer:
#         """Get centroids and track"""
#         # Perform clustering across the agents
#         clusters = self.clusterer(objects=objects, frame=frame, timestamp=timestamp)

#         # Fuse the cluster information to get a "detection"
#         detections = self.fusion(clusters)
#         # centroids = clusters.apply_and_return("centroid")
#         # centroid_dets = [
#         #     CentroidDetection(
#         #         source_identifier="ct", centroid=cent.x, reference=cent.reference
#         #     )
#         #     for cent in centroids
#         # ]
#         # detections = DataContainer(
#         #     frame=frame, timestamp=timestamp, data=centroid_dets, source_identifier="ct"
#         # )

#         # Tracking on cluster centroids
#         cluster_tracks = self.tracker(
#             t=timestamp,
#             frame=frame,
#             detections=detections,
#             platform=self.platform,
#         )
