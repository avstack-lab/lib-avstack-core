import numpy as np

from avstack.modules import estimation


def test_linear_predict():
    ndim = 10
    x = np.random.rand(ndim)
    P = 4 * np.eye(ndim) + np.random.rand(ndim, ndim)
    P = (P + P.T) / 2
    F_func = lambda dt: 2 * np.eye(ndim)
    Q_func = lambda dt: dt * np.eye(ndim)
    xp, Pp = estimation.kalman_linear_predict(x, P, F_func, Q_func, dt=0.01)
    F = F_func(None)
    Q = Q_func(0.01)
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
    Q_func = lambda dt: dt * np.eye(ndim)
    xp, Pp = estimation.kalman_extended_predict(x, P, f_func, F_func, Q_func, dt)
    F = F_func(x, dt)
    Q = Q_func(dt)
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
    Q_func = lambda dt: dt * np.eye(ndim)
    xp, Pp = estimation.kalman_unscented_predict(x, P, f_func, Q_func, dt)
    assert xp.shape == x.shape
    assert Pp.shape == P.shape
    assert np.all(np.linalg.eigvals(Pp) > 0)


def test_unscented_update():
    ndim = 10
    xp = np.random.rand(ndim)
    Pp = 4 * np.eye(ndim)
    z = np.random.rand(ndim)
    h_func = lambda x: x**2
    R = np.eye(ndim)
    x, P = estimation.kalman_unscented_update(xp, Pp, z, h_func, R)
    assert x.shape == xp.shape
    assert P.shape == Pp.shape
    assert np.all(np.linalg.eigvals(P) > 0)


def test_all_kalmans_on_linear_problem():
    """state is [x, xdot, y, ydot]"""
    ndim = 4
    F_func = lambda dt: np.kron(np.eye(2), np.array([[1, dt], [0, 1]]))
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    sigma = 0.02
    Q_func = lambda dt: sigma**2 * np.kron(
        np.eye(2),
        np.array([[1 / 4 * dt**4, 1 / 2 * dt**3], [1 / 2 * dt**3, dt**2]]),
    )
    std_x = std_y = 0.5
    R = np.array([[std_x**2, 0], [0, std_y**2]])

    # make measurements
    np.random.seed(1234)
    n_steps = 100
    dt = 1.0
    zs = [
        np.array([i + np.random.randn() * std_x, i + np.random.randn() * std_y])
        for i in range(100)
    ]

    #  linear kalman
    x = np.zeros((ndim,))
    P = np.eye(ndim)
    for i in range(n_steps):
        xp, Pp = estimation.kalman_linear_predict(x, P, F_func, Q_func, dt)
        x, P = estimation.kalman_linear_update(xp, Pp, zs[i], H, R)
    x_linear, P_linear = x, P

    # extended kalman
    x = np.zeros((ndim,))
    P = np.eye(ndim)
    f_func_ex = lambda x, dt: F_func(dt) @ x
    F_func_ex = lambda x, dt: F_func(dt)
    h_func_ex = lambda x: H @ x
    H_func_ex = lambda x: H
    for i in range(n_steps):
        xp, Pp = estimation.kalman_extended_predict(
            x, P, f_func_ex, F_func_ex, Q_func, dt
        )
        x, P = estimation.kalman_extended_update(xp, Pp, zs[i], h_func_ex, H_func_ex, R)
    x_extended, P_extended = x, P

    # unscented kalman
    x = np.zeros((ndim,))
    P = np.eye(ndim)
    f_func_ut = lambda x, dt: F_func(dt) @ x
    h_func_ut = lambda x: H @ x
    for i in range(n_steps):
        xp, Pp = estimation.kalman_unscented_predict(x, P, f_func_ut, Q_func, dt)
        x, P = estimation.kalman_unscented_update(xp, Pp, zs[i], h_func_ut, R)
    x_unscented, P_unscented = x, P

    # evaluate results
    # -- extended reduces exactly
    assert np.allclose(x_linear, x_extended)
    assert np.allclose(P_linear, P_extended)
    assert np.allclose(x_linear, x_unscented)
    assert np.allclose(P_linear, P_unscented)
