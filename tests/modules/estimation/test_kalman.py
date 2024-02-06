import numpy as np

from avstack.modules import estimation


def test_linear_predict():
    ndim = 10
    x = np.random.rand(ndim)
    P = 4 * np.eye(ndim) + np.random.rand(ndim, ndim)
    P = (P + P.T) / 2
    F = 2 * np.eye(ndim)
    Q = np.random.rand(ndim, ndim)
    xp, Pp = estimation.kalman_linear_predict(x, P, F, Q)
    assert np.allclose(xp, F @ x)
    assert np.allclose(Pp, F @ P @ F.T + Q)


def test_linear_update():
    ndim = 10
    xp = np.random.rand(ndim)
    Pp = 4 * np.eye(ndim)
    z = np.random.rand(ndim)
    H = np.eye(ndim)
    R = np.eye(ndim)
    x, P = estimation.kalman_linear_update(xp, Pp, z, H, R)
    assert all(((xp <= x) & (x <= z)) | ((z <= x) * (x <= xp)))


def test_extended_predict():
    dt = 0.1
    ndim = 10
    x = np.random.rand(ndim)
    P = 4 * np.eye(ndim) + np.random.rand(ndim, ndim)
    P = (P + P.T) / 2
    f_func = lambda x, dt: x**2
    F_func = lambda x, dt: 2 * np.diag(x)
    Q = np.random.rand(ndim, ndim)
    xp, Pp = estimation.kalman_extended_predict(x, P, f_func, F_func, Q, dt)
    F = F_func(x, dt)
    assert not np.allclose(xp, F @ x)
    assert np.allclose(Pp, F @ P @ F.T + Q)


def test_extended_update():
    ndim = 10
    xp = np.random.rand(ndim)
    Pp = 4 * np.eye(ndim)
    z = np.random.rand(ndim)
    h_func = lambda x: x**2
    H_func = lambda x: 2 * np.diag(x)
    R = np.eye(ndim)
    x, P = estimation.kalman_extended_update(xp, Pp, z, h_func, H_func, R)


def test_compute_sigma():
    ndim = 10
    x = np.random.rand(ndim)
    P = 4 * np.eye(ndim)
    x_sigma, Wm_sigma, Wc_sigma = estimation.compute_sigma_points(x, P)
    assert len(x_sigma) == 2 * ndim + 1
    assert len(Wm_sigma) == 2 * ndim + 1
    assert len(Wc_sigma) == 2 * ndim + 1


def test_unscented_predict():
    dt = 0.1
    ndim = 10
    x = np.random.rand(ndim)
    P = 4 * np.eye(ndim) + np.random.rand(ndim, ndim)
    P = (P + P.T) / 2
    f_func = lambda x, dt: x**2
    Q = np.random.rand(ndim, ndim)
    x_sigma, Wm_sigma, Wc_sigma = estimation.kalman.compute_sigma_points(x, P)
    xp, Pp, xp_sigma = estimation.kalman_unscented_predict(
        x_sigma, Wm_sigma, Wc_sigma, f_func, Q, dt
    )
    assert xp.shape == x.shape
    assert Pp.shape == P.shape
    assert len(xp_sigma) == 2 * ndim + 1


def test_unscented_update():
    ndim = 10
    xp = np.random.rand(ndim)
    Pp = 4 * np.eye(ndim)
    z = np.random.rand(ndim)
    h_func = lambda x: x**2
    H_func = lambda x: 2 * np.diag(x)
    R = np.eye(ndim)
    xp_sigma, Wm_sigma, Wc_sigma = estimation.kalman.compute_sigma_points(xp, Pp)
    x, P, z_sigma = estimation.kalman_unscented_update(
        xp, Pp, z, xp_sigma, Wm_sigma, Wc_sigma, h_func, R
    )
    assert x.shape == xp.shape
    assert P.shape == P.shape
    assert len(z_sigma) == 2 * ndim + 1
