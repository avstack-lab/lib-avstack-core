from typing import Dict

from avstack.datastructs import DataContainer
from avstack.modules.perception.detections import CentroidDetection

from .types import ClusterSet


class ClusterTracker:
    def __init__(self, clustering, tracker, platform) -> None:
        """Perform tracking on the cluster centroids

        Helps to maintain consistent cluster IDs
        """
        self.clustering = clustering
        self.tracker = tracker
        self.platform = platform

    def __call__(
        self, objects: Dict[int, list], frame: int, timestamp: float
    ) -> ClusterSet:
        """Get centroids and track"""

        clusters = self.clustering(objects=objects, frame=frame, timestamp=timestamp)
        centroids = clusters.apply_and_return("centroid")
        centroid_dets = [
            CentroidDetection(
                source_identifier="ct", centroid=cent.x, reference=cent.reference
            )
            for cent in centroids
        ]
        detections = DataContainer(
            frame=frame, timestamp=timestamp, data=centroid_dets, source_identifier="ct"
        )
        cluster_tracks = self.tracker(
            t=timestamp,
            frame=frame,
            detections=detections,
            platform=self.platform,
        )
        # TODO: maintain some idea of which clusters go with which tracks

        return cluster_tracks
