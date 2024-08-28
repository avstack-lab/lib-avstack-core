"""
Some sampling algorithms take inspiration from code at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
"""

import numpy as np


def n_eff(weights):
    return 1.0 / np.sum(np.square(weights))


def multinomial_sampling(weights: np.ndarray, n: int) -> np.ndarray:
    """Draw samples from multinomial distribution

    For a more efficient implementation, see
    https://en.wikipedia.org/wiki/Alias_method
    """
    weights /= sum(weights)  # normalize
    w_cum = np.cumsum(weights)  # CDF
    w_cum[-1] = 1.0  # avoid round-off error
    rands = np.random.rand(n)  # get points on cdf curve to sample
    indices = np.searchsorted(w_cum, rands)
    return indices


def residual_sampling(weights: np.ndarray, n: int) -> np.ndarray:
    """Use the residual method to get samples

    Uses multinomial sampling on the residuals
    """
    if n != len(weights):
        raise NotImplementedError(
            "Not sure how to implement for n less than weights len"
        )
    indices = np.zeros(n, dtype=int)

    # take floor(N*w copies of each weight)
    num_copies = (n * np.asarray(weights)).astype(int)
    k = 0
    for i in range(n):
        for _ in range(num_copies[i]):  # make n copies
            indices[k] = i
            k += 1

    # use multinomial sampling on residuals for the rest
    residual = weights - num_copies  # fractional part
    indices[k:n] = multinomial_sampling(residual, n - k)

    return indices


def stratified_sampling(weights: np.ndarray, n: int) -> np.ndarray:
    """Uses stratified resampling

    Guarantees each sample is between 0 and 2/n apart
    """
    # make subdivisions and choose a random position
    positions = (np.random.rand(n) + range(n)) / n
    indices = _do_spaced_sampling(positions, weights, n)
    return indices


def systematic_sampling(weights: np.ndarray, n: int) -> np.ndarray:
    """Uses systematic resampling

    Guarantees each sample is exactly 0 and 1/n apart
    """
    # make subdivisions and choose a random consistent offset
    positions = (np.arange(n) + np.random.rand()) / n
    indices = _do_spaced_sampling(positions, weights, n)
    return indices


def _do_spaced_sampling(positions: np.ndarray, weights: np.ndarray, n: int):
    if n != len(weights):
        raise NotImplementedError(
            f"Not sure how to implement for n less than weights len, {n} vs {len(weights)}"
        )

    indices = np.zeros(n, dtype=int)
    cum_sum = np.cumsum(weights)  # CDF
    i, j = 0, 0
    while i < n:
        if positions[i] < cum_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices
