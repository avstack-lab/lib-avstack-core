import numpy as np

from avstack.metrics.assignment.instantaneous import OspaMetric


def test_ospa_metric_one_to_one():
    tracks = [np.random.randn(3)]
    truths = [np.random.randn(3)]
    cost = OspaMetric.cost(tracks, truths)
    assert cost > 0


def test_ospa_metric_two_to_one():
    tracks = [np.random.randn(3) * i**3 for i in range(2)]
    truths = [np.random.randn(3) for _ in range(1)]
    cost1 = OspaMetric.cost(tracks[:1], truths)
    cost2 = OspaMetric.cost(tracks, truths)
    assert 0 < cost1 < cost2
