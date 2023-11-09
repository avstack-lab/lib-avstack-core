from typing import Dict

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
        cluster_tracks = self.tracker(
            t=centroids.timestamp,
            frame=centroids.frame,
            detections=centroids,
            platform=self.platform,
        )
