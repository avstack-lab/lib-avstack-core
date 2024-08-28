from typing import List, NamedTuple

import numpy as np

from avstack.modules.assignment import gnn_single_frame_assign


class SingleFrameMetrics(NamedTuple):
    """Container for tracking metrics data"""

    n_truths: int
    n_tracks: int
    n_assign: int
    precision: float
    recall: float
    ospa: float
    timestamp: float = 0.0


class ConfusionMatrix(NamedTuple):
    """Container for confusion matrix for assignment"""
    n_true_positives: int
    n_true_negatives: int
    n_false_positives: int
    n_false_negatives: int

    @property
    def _ntp(self):
        return self.n_true_positives

    @property
    def _ntn(self):
        return self.n_true_negatives

    @property
    def _nfp(self):
        return self.n_false_positives

    @property
    def _nfn(self):
        return self.n_false_negatives

    @property
    def matrix(self) -> np.ndarray:
        return np.ndarray([[self._ntp, self._nfp], [self._nfn, self._ntn]])

    @property
    def precision(self) -> float:
        if (self._ntp + self._nfp) == 0:
            return 0.0
        else:
            return self._ntp / (self._ntp + self._nfp)

    @property
    def recall(self) -> float:
        if (self._ntp + self._nfn) == 0:
            return 0.0
        else:
            return self._ntp / (self._ntp + self._nfn)

    def __getitem__(self, idx0: int, idx1: int) -> int:
        return self.matrix[idx0, idx1]


class OspaMetric:
    @staticmethod
    def _cost(
        list_shorter: list, list_longer: list, p: float = 1.0, c: float = 1.0
    ) -> float:
        n = len(list_longer)
        m = len(list_shorter)
        if n == 0:
            return 0.0
        A = np.array(
            [[np.linalg.norm(l1 - l2) for l1 in list_shorter] for l2 in list_longer]
        )
        assign = gnn_single_frame_assign(A, cost_threshold=c)
        rows, cols = assign.rows_and_cols()
        c_una = c * len(assign.unassigned_rows)
        distance = (
            1 / n * (np.sum(A[rows, cols]) + c_una) ** p + (n - m) * c**p
        ) ** 1 / p
        return distance

    @staticmethod
    def cost(tracks: list, truths: list, p: float = 1.0, c: float = 1.0) -> float:
        if len(tracks) <= len(truths):
            return OspaMetric._cost(tracks, truths, p=p, c=c)
        else:
            return OspaMetric._cost(truths, tracks, p=p, c=c)


def get_instantaneous_metrics(
    tracks: List,
    truths: List,
    assign_radius: float = 2.0,
    timestamp: float = 0.0,
) -> SingleFrameMetrics:
    # convert to avstack types
    trks_position = [track.position.x for track in tracks]
    trus_position = [
        truth.position.x for truth in truths if "static" not in truth.obj_type
    ]

    # compute metrics
    A = np.array(
        [[np.linalg.norm(l1 - l2) for l1 in trus_position] for l2 in trks_position]
    )
    if len(A.shape) < 2:
        A = A[:, None]
    assign = gnn_single_frame_assign(A, cost_threshold=assign_radius)

    # get metrics
    confusion = ConfusionMatrix(
        n_true_positives=len(assign),
        n_true_negatives=0,  # because assignment problem
        n_false_positives=len(assign.unassigned_rows),
        n_false_negatives=len(assign.unassigned_cols),
    )
    ospa = OspaMetric.cost(trks_position, trus_position)

    # package up
    tracking_metrics = SingleFrameMetrics(
        ospa=ospa,
        n_truths=len(truths),
        n_tracks=len(tracks),
        n_assign=len(assign),
        precision=confusion.precision,
        recall=confusion.recall,
        timestamp=timestamp,
    )

    return tracking_metrics
