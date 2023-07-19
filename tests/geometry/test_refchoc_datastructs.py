import numpy as np

from avstack.calibration import Calibration, CameraCalibration
from avstack.geometry import (
    Acceleration,
    AngularVelocity,
    Attitude,
    GlobalOrigin3D,
    PointMatrix3D,
    Position,
    ReferenceFrame,
    Velocity,
    q_mult_vec,
    q_stan_to_cam,
    transform_orientation,
)
from avstack.geometry import transformations as tforms


def x_rand():
    return np.random.randn(3)


def q_rand():
    return transform_orientation(np.random.rand(3), "euler", "quat")


def get_random_frame():
    return ReferenceFrame(
        x=x_rand(),
        v=x_rand(),
        acc=x_rand(),
        q=q_rand(),
        ang=q_rand(),
        reference=GlobalOrigin3D,
    )


# ================================
# STATES
# ================================


def test_position():
    ref = get_random_frame()
    p = Position(x_rand(), ref)


def test_velocity():
    ref = get_random_frame()
    v = Velocity(x_rand(), ref)


def test_acceleration():
    ref = get_random_frame()
    acc = Acceleration(x_rand(), ref)


def test_attitude():
    ref = get_random_frame()
    att = Attitude(q_rand(), ref)


def test_angularvelocity():
    ref = get_random_frame()
    ang = AngularVelocity(q_rand(), ref)


# ================================
# MATRICES
# ================================


def get_calib_cam():
    # -- set up camera calibration -- from FOV of 90 degrees
    x1 = np.array([0, 0, 0])
    q1 = q_stan_to_cam
    O_cam = ReferenceFrame(x1, q1, GlobalOrigin3D)
    P = np.array(
        [[800.0, 0.0, 800.0, 0.0], [0.0, 800.0, 450.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    img_shape = [900, 1600, 3]
    calib_cam = CameraCalibration(O_cam, P, img_shape)
    return calib_cam


def test_point_matrix_3d():
    calib1 = Calibration(GlobalOrigin3D)
    pm = PointMatrix3D(np.random.rand(100, 3), calibration=calib1)
    cf1 = ReferenceFrame(x_rand(), q_rand(), GlobalOrigin3D)
    pm_cf1 = pm.change_reference(cf1)
    assert np.allclose(pm_cf1.x, q_mult_vec(cf1.q, pm.x - cf1.x))


def test_project_3d_to_2d():
    calib1 = Calibration(GlobalOrigin3D)
    pm_3d = PointMatrix3D(np.random.rand(100, 3), calibration=calib1)
    calib_cam = get_calib_cam()
    pm_2d = pm_3d.project_to_2d(calib_cam)
    assert len(pm_2d) == len(pm_3d)


def test_project_3d_to_2d_2():
    calib1 = Calibration(GlobalOrigin3D)
    pm_3d = PointMatrix3D(np.array([[100, 0, 0]]), calibration=calib1)
    calib_cam = get_calib_cam()
    pm_2d = pm_3d.project_to_2d(calib_cam)
    assert np.allclose(pm_2d.x, np.array([[calib_cam.width / 2, calib_cam.height / 2]]))


def test_point_matrix_2d_angles():
    R1 = ReferenceFrame(np.zeros((3,)), np.quaternion(1), reference=GlobalOrigin3D)
    R2 = ReferenceFrame(np.zeros((3,)), q_stan_to_cam, reference=GlobalOrigin3D)
    P = np.array(
        [[800.0, 0.0, 800.0, 0.0], [0.0, 800.0, 450.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    img_shape = [900, 1600, 3]
    fov_h = 90 * np.pi / 180
    calib_cam = CameraCalibration(
        R2, P, img_shape, fov_horizontal=fov_h, fov_vertical=None, square_pixels=True
    )
    x_razel = np.array([[50, 0.4, -0.3]])
    x_cart = PointMatrix3D(tforms.matrix_spherical_to_cartesian(x_razel), R1)
    x_pixels = x_cart.project_to_2d(calib_cam)
    x_angles = x_pixels.angles
    assert np.allclose(x_angles, x_razel[:, 1:], atol=0.05)


def test_project_to_camera_halves_same_ReferenceFrame():
    calib_cam = get_calib_cam()
    x1 = np.array([0, 0, 0])
    q2 = np.quaternion(1)
    O_pts = ReferenceFrame(x1, q2, GlobalOrigin3D)

    # -- make all points (in front only)
    pts_all = 2 * (np.random.rand(1000, 3) - 0.5) * np.array([10, 10, 2]) + np.array(
        [40, 0, 0]
    )

    # -- points on left half of image
    pts_left = PointMatrix3D(pts_all[pts_all[:, 1] >= 0], calibration=calib_cam)
    pts_proj = pts_left.project_to_2d(calib_cam)
    assert len(pts_proj) == len(pts_left) < len(pts_all)
    pts_proj_left = pts_proj[0 <= (pts_proj[:, 0] <= calib_cam.img_shape[1] / 2), :]
    assert len(pts_proj_left) == len(pts_left)

    # -- points on right half of image
    pts_right = PointMatrix3D(pts_all[pts_all[:, 1] <= 0], calibration=calib_cam)
    pts_proj = pts_right.project_to_2d(calib_cam)
    assert len(pts_proj) == len(pts_right) < len(pts_all)
    pts_proj_right = pts_proj[
        calib_cam.img_shape[1] > (pts_proj[:, 0] >= calib_cam.img_shape[1] / 2), :
    ]
    assert len(pts_proj_right) == len(pts_right)

    # -- points on top half of image
    pts_top = PointMatrix3D(pts_all[pts_all[:, 2] >= 0], calibration=calib_cam)
    pts_proj = pts_top.project_to_2d(calib_cam)  # same origin
    assert len(pts_proj) == len(pts_top) < len(pts_all)
    pts_proj_top = pts_proj[0 <= (pts_proj[:, 1] <= calib_cam.img_shape[0] / 2), :]
    assert len(pts_proj_top) == len(pts_top)

    # -- points on bottom half of image
    pts_bot = PointMatrix3D(pts_all[pts_all[:, 2] <= 0], calibration=calib_cam)
    pts_proj = pts_bot.project_to_2d(calib_cam)  # same origin
    assert len(pts_proj) == len(pts_bot) < len(pts_all)
    pts_proj_bot = pts_proj[
        calib_cam.img_shape[0] > (pts_proj[:, 1] >= calib_cam.img_shape[0] / 2), :
    ]
    assert len(pts_proj_bot) == len(pts_bot)


def test_filter_point_matrix():
    calib1 = Calibration(GlobalOrigin3D)
    pm_3d = PointMatrix3D(np.random.rand(100, 3), calibration=calib1)
    mask = np.random.rand(100) < 0.25
    pm_3d_2 = pm_3d.filter(mask)
    assert len(pm_3d_2) < len(pm_3d)
