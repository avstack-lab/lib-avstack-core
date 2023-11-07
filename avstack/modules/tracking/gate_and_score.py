import math

import numpy as np


def rectangular_gate(y, S, Kg=4.0):
    """
    Coarse, rectangular gate

    ---inputs
    y - residual
    S - innovation covariance (Pzz)
    Kg - the number of sigmas

    ---returns
    passed - boolean
    """
    return np.all(np.squeeze(y) ** 2 < Kg * np.diag(S))


def ellipsoidal_gate(d2, y, S, CHI2_TABLE, method="sigmas"):
    """
    Finer, ellipsoidal gate

    --inputs
    d2 - normalized distance given by prior calculation of y.T @ S^-1 @ y
    y - residual
    S - innovation covariance (Pzz)
    method - method to compute gate itself

    --returns
    passed - boolean
    """
    if method == "sigmas":
        return d2 < CHI2_TABLE[len(y)]
    elif method == "max_likelihood":
        raise NotImplementedError
    else:
        raise NotImplementedError


def get_score(d2, S, PD, BETA_FT):
    M = S.shape[0]
    return (
        1 / 2 * d2
        - np.log(PD / BETA_FT)
        + M / 2 * np.log(2 * math.pi)
        + 1 / 2 * np.log(np.linalg.det(S))
    )


def get_assignment_cost_from_gate(
    y,
    S,
    Sinv,
    n_updates,
    PD,
    BETA_FT,
    N_SIGMA_GATE,
    N_UPDATES_SCORE_PENALTY,
    CHI2_TABLE,
):
    """
    Define the assignment costs when assigning detections and tracks

    Depends on the type of measurements
    FLAG -- 0: passed all
            1: failed score gate
            2: failed maneuver gate
            3: failed ellipsoidal gate
            4: failed rectangular gate
    """
    # --- rectangular gate
    if not rectangular_gate(y, S, N_SIGMA_GATE):
        return np.inf, None, np.inf, 4

    # --- ellipsoidal gate
    d2 = y.T @ Sinv @ y
    if not ellipsoidal_gate(d2, y, S, CHI2_TABLE):
        return np.inf, None, d2, 3

    # -- maneuver gate
    # TBD

    # --- score gate (only improves the score)
    score = get_score(d2, S, PD, BETA_FT)
    if score > 0:
        return np.inf, None, d2, 1

    # --- apply score penalty, if applicable
    cost = score * min(1, (n_updates + 1) / N_UPDATES_SCORE_PENALTY)
    return cost, score, d2, 0
