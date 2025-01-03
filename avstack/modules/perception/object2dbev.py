from itertools import combinations

import numpy as np
import scipy.spatial.distance

from avstack import maskfilters
from avstack.config import MODELS

from .base import _PerceptionAlgorithm
from .detections import CentroidDetection


try:
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
except ModuleNotFoundError as e:
    print("Cannot use sklearn methods")


@MODELS.register_module()
class Lidar2dCentroidDetector(_PerceptionAlgorithm):
    """Detects object centroids from 2D lidar data"""

    MODE = "object 2d bev"

    def __init__(
        self,
        min_range=0,
        max_range=25,
        eps=0.5,
        p=2,
        min_samples=50,
        pct_sample_keep=1.0,
        pca_ratio_ignore=3,
        merge_dist_thresh=3,
    ):
        self.eps = eps
        self.p = p
        self.pct_sample_keep = pct_sample_keep
        self.min_samples = min_samples
        self.min_range = min_range
        self.max_range = max_range
        self.pca_ratio_ignore = pca_ratio_ignore
        self.merge_dist_thresh = merge_dist_thresh

    def __call__(self, data, platform, alg_name, **kwargs):
        data = data.data
        assert data.shape[1] == 2, "Data must be in 2D format"
        return self._wrap(
            self._merge(self._detect(self._filter(data))), platform, alg_name
        )

    def _filter(self, data):
        pts_range = data[
            maskfilters.filter_points_range(data, self.min_range, self.max_range), :
        ]
        n = int(np.round(pts_range.shape[0] * self.pct_sample_keep))
        pts_samp = data[np.random.choice(pts_range.shape[0], n, replace=False), :]
        return pts_samp

    def _detect(self, data):
        dbscan = DBSCAN(
            eps=self.eps, p=self.p, min_samples=self.min_samples, leaf_size=20
        )
        pca = PCA()
        # remove near-duplicates and add sample weight TBD
        clustering = dbscan.fit_predict(data)
        centroids = []
        for i in set(clustering):
            pts = data[clustering == i, :]
            pca.fit(pts)
            if (
                pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]
                <= self.pca_ratio_ignore
            ):
                centroids.append(np.mean(pts, axis=0))
        return centroids

    def _merge(self, centroids):
        dists = scipy.spatial.distance.pdist(np.asarray(centroids), "euclidean")
        idx_close = np.argwhere(dists < self.merge_dist_thresh)
        combos = list(combinations(range(len(centroids)), 2))
        if len(idx_close) > 0:
            for idx in idx_close[0]:
                i0 = combos[idx][0]
                i1 = combos[idx][1]
                centroids[i0] = np.mean(
                    np.vstack((centroids[i0], centroids[i1])), axis=0
                )
                del centroids[i1]
        return centroids

    def _wrap(self, centroids, reference, alg_name):
        dets = []
        for cent in centroids:
            dets.append(
                CentroidDetection(
                    data=cent,
                    noise=np.array([1] * len(cent)),
                    source_identifier=alg_name,
                    reference=reference,
                )
            )
        return dets
