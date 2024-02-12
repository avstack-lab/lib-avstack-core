import numpy as np


def kalman_linear_predict(x, P, F_func, Q_func, dt):
    F = F_func(dt)
    Q = Q_func(dt)
    assert Q.shape == P.shape
    return F @ x, F @ P @ F.T + Q


def kalman_linear_update(xp, Pp, z, H, R):
    z = np.squeeze(z)
    assert R.shape == (len(z), len(z))
    assert H.shape == (len(z), len(xp))
    y = z - H @ xp
    S = H @ Pp @ H.T + R
    K = Pp @ H.T @ np.linalg.inv(S)
    return xp + K @ y, (np.eye(Pp.shape[0]) - K @ H) @ Pp


def kalman_extended_predict(x, P, f_func, F_func, Q_func, dt):
    F = F_func(x, dt)
    Q = Q_func(dt)
    return f_func(x, dt), F @ P @ F.T + Q


def kalman_extended_update(xp, Pp, z, h_func, H_func, R):
    H = H_func(xp)
    y = z - h_func(xp)
    S = H @ Pp @ H.T + R
    K = Pp @ H.T @ np.linalg.inv(S)
    return xp + K @ y, (np.eye(Pp.shape[0]) - K @ H) @ Pp


def compute_sigma_points(x, P, alpha=1e-3, kappa=0, beta=2):
    L = len(x)
    lam = alpha**2 * (L + kappa) - L
    LlPsqrt = np.linalg.cholesky((L + lam) * P)
    x_sigma = [x]
    x_sigma.extend([x + LlPsqrt[:, i] for i in range(L)])
    x_sigma.extend([x - LlPsqrt[:, i] for i in range(L)])
    Wm_sigma = [lam / (L + lam)]
    Wm_sigma.extend([1 / (2 * (L + lam)) for _ in range(2 * L)])
    Wc_sigma = [lam / (L + lam) + 1 - alpha**2 + beta]
    Wc_sigma.extend([1 / (2 * (L + lam)) for _ in range(2 * L)])
    return x_sigma, Wm_sigma, Wc_sigma


def unscented_transform(x_sigma, Wm_sigma, Wc_sigma):
    x = np.sum(
        [Wm_sigma_ * x_sigma_ for Wm_sigma_, x_sigma_ in zip(Wm_sigma, x_sigma)], axis=0
    )
    P = np.sum(
        [
            Wc_sigma_ * np.outer(x_sigma_ - x, x_sigma_ - x)
            for Wc_sigma_, x_sigma_ in zip(Wc_sigma, x_sigma)
        ],
        axis=0,
    )
    return x, P


def kalman_unscented_predict(x, P, f_func, Q_func, dt):
    x_sigma, Wm_sigma, Wc_sigma = compute_sigma_points(x, P)
    xp_sigma = [f_func(x_sigma_, dt) for x_sigma_ in x_sigma]
    xp, Pp = unscented_transform(xp_sigma, Wm_sigma, Wc_sigma)
    Q = Q_func(dt)
    return xp, Pp + Q


def kalman_unscented_update(xp, Pp, z, h_func, R):
    xp_sigma, Wm_sigma, Wc_sigma = compute_sigma_points(xp, Pp)
    z_sigma = [h_func(xp_sigma_) for xp_sigma_ in xp_sigma]
    mz, Pzz = unscented_transform(z_sigma, Wm_sigma, Wc_sigma)
    Pzz = Pzz + R
    y = z - mz
    Pxz = np.sum(
        [
            Wc_sigma_ * np.outer(xp_sigma_ - xp, z_sigma_ - mz)
            for Wc_sigma_, xp_sigma_, z_sigma_ in zip(Wc_sigma, xp_sigma, z_sigma)
        ],
        axis=0,
    )
    # import pdb; pdb.set_trace()
    # K = np.linalg.solve(Pzz, Pxz)
    K = Pxz @ np.linalg.inv(Pzz)
    x = xp + K @ y
    P = Pp - K @ Pzz @ K.T
    return x, P
